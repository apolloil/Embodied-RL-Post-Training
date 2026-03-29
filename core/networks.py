import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal

def batch_norm_to_group_norm(model):
    """Recursively replace all BatchNorm2d layers with GroupNorm.

    Args:
        model: nn.Module whose BatchNorm layers will be replaced in-place.

    Returns:
        The same model with GroupNorm substituted for BatchNorm.
    """
    # Channel-count to group-count lookup table
    GROUP_NORM_LOOKUP = {
        16: 2, 32: 4, 64: 8, 128: 8, 256: 16, 512: 32, 1024: 32, 2048: 32,
    }

    # Iterate over direct children (named_children does not recurse)
    for name, child in model.named_children():
        
        # Case A: This is a BN layer -> replace it
        if isinstance(child, torch.nn.BatchNorm2d):
            num_channels = child.num_features
            num_groups = GROUP_NORM_LOOKUP[num_channels]
            
            # Safe to setattr because name is a direct child attribute
            setattr(model, name, torch.nn.GroupNorm(num_groups, num_channels))
        
        # Case B: Not a BN layer (e.g. Sequential, BasicBlock) -> recurse
        else:
            batch_norm_to_group_norm(child)
            
    return model

class TruncatedNormal(pyd.Normal):
    """
    Truncated Normal Distribution.
    Source: baku/utils.py
    """
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)

class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # Numeric stability adaptation
        return self.atanh(y.clamp(-0.99999997, 0.99999997))

    def log_abs_det_jacobian(self, x, y):
        # Formula: 2 * (log(2) - x - softplus(-2x))
        # More numerically stable than log(1 - tanh(x)^2)
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))

class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale
        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        # Approximate mean of tanh(Normal) is tanh(loc)
        # Accurate enough for inference/eval
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu
# =============================================================================
# 1. Vision Encoder (ResNet based, GroupNorm, No FiLM)
# =============================================================================

class SpatialSoftmax(nn.Module):
    def __init__(self, in_c, in_h, in_w, num_kp):
        super().__init__()
        self._spatial_conv = nn.Conv2d(in_c, num_kp, kernel_size=1)

        pos_x, pos_y = torch.meshgrid(
            torch.linspace(-1, 1, in_w).float(),
            torch.linspace(-1, 1, in_h).float(),
            indexing='xy'
        )
        
        pos_x = pos_x.reshape(1, in_w * in_h)
        pos_y = pos_y.reshape(1, in_w * in_h)
        self.register_buffer("pos_x", pos_x)
        self.register_buffer("pos_y", pos_y)

        self._num_kp = num_kp
        self._in_c = in_c
        self._in_w = in_w
        self._in_h = in_h

    def forward(self, x):
        h = x
        if self._num_kp != self._in_c:
            h = self._spatial_conv(h)
        h = h.contiguous().view(-1, self._in_h * self._in_w)

        attention = F.softmax(h, dim=-1)
        
        keypoint_x = (self.pos_x * attention).sum(1, keepdim=True).view(-1, self._num_kp)
        keypoint_y = (self.pos_y * attention).sum(1, keepdim=True).view(-1, self._num_kp)
        
        keypoints = torch.cat([keypoint_x, keypoint_y], dim=1)
        return keypoints


class VisionEncoder(nn.Module):
    def __init__(self, input_shape, output_size, pretrained=False, freeze=False):
        super().__init__()
        
        # input_shape: (C, H, W)
        c, h, w = input_shape
        assert c == 3, "input shape must be (3, H, W)"
        
        # Load ResNet18
        resnet = torchvision.models.resnet18(pretrained=pretrained)
        
        # Replace BatchNorm with GroupNorm (Essential for BC/RL stability)
        resnet = batch_norm_to_group_norm(resnet)
        
        # Remove avgpool and fc
        layers = list(resnet.children())[:-2]
        self.backbone = nn.Sequential(*layers)
        
        # Freeze weights if requested
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Calculate output shape
        with torch.no_grad():
            dummy_input = torch.zeros(1, c, h, w)
            dummy_output = self.backbone(dummy_input)
            feat_c, feat_h, feat_w = dummy_output.shape[1:]
            
        # Spatial Projection
        # num_kp is explicit, no default None
        self.num_kp = output_size // 2
        self.spatial_softmax = SpatialSoftmax(feat_c, feat_h, feat_w, num_kp=self.num_kp)
        self.projection = nn.Linear(self.num_kp * 2, output_size)
        
    def forward(self, x):
        h = self.backbone(x) 
        h = self.spatial_softmax(h) 
        h = self.projection(h) 
        return h


# =============================================================================
# 2. Task Embedding
# =============================================================================

class TaskEmbedding(nn.Module):
    def __init__(self, num_tasks, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_tasks, embed_dim)
    
    def forward(self, task_ids):
        x =self.embedding(task_ids)
        return F.normalize(x, p=2, dim=-1)


# =============================================================================
# 3. GPT Backbone
# =============================================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                                     .view(1, 1, block_size, block_size))
        self.n_head = n_head
        self.n_embd = n_embd

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.c_proj = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.c_fc(x)
        # New GELU implementation (identical to OpenAI GPT)
        x = 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPTBackbone(nn.Module):
    def __init__(self, input_dim, output_dim, n_layer, n_head, n_embd, block_size, dropout):
        super().__init__()
        self.block_size = block_size
        self.n_layer = n_layer
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Linear(input_dim, n_embd),
            wpe = nn.Embedding(block_size, n_embd),
            drop = nn.Dropout(dropout),
            h = nn.ModuleList([Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]),
            ln_f = nn.LayerNorm(n_embd),
        ))
        
        self.lm_head = nn.Linear(n_embd, output_dim, bias=False)

        # Init weights
        self.apply(self._init_weights)
        
        # Special scaled init for residual projections (Missed in previous iteration)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer)
                )

        # Report parameters
        n_params = sum(p.numel() for p in self.parameters())
        print("GPT number of parameters: %.2fM" % (n_params / 1e6,))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, x):
        # x: (B, T, input_dim)
        device = x.device
        b, t, d = x.size()
        assert t <= self.block_size, f"Sequence length {t} exceeds block size {self.block_size}"
        
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)

        tok_emb = self.transformer.wte(x) # (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        for block in self.transformer.h:
            x = block(x)
            
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, output_dim)
        
        return logits


# =============================================================================
# 4. Stochastic Policy Head
# =============================================================================

class StochasticActorHead(nn.Module):
    """
    Stochastic Actor Head for SAC (Online RL).
    Outputs a Squashed Normal Distribution (mu, std -> tanh).
    """
    def __init__(self, input_size, output_size, hidden_size, num_layers, 
                 log_std_min, log_std_max):
        super().__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # 1. Backbone MLP (Shared features)
        sizes = [input_size] + [hidden_size] * num_layers
        layers = []
        for i in range(num_layers):
            layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU()]
        self.backbone = nn.Sequential(*layers)
        
        # 2. Output Projection (Projects to 2 * action_dim)
        # JAXRL typically uses a single layer that splits into mu and log_std
        self.proj = nn.Linear(sizes[-1], 2 * output_size)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # Orthogonal init is common in RL to prevent vanishing gradients
            nn.init.orthogonal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x):
        # x: (B, input_size) or (B, T, input_size)
        feat = self.backbone(x)
        
        # Split into mean and log_std
        mu_log_std = self.proj(feat)
        mu, log_std = mu_log_std.chunk(2, dim=-1)
        
        # Tanh squash for log_std (Soft clip)
        # This is often more stable than hard torch.clamp during training
        # Formula: min + 0.5 * (max - min) * (tanh(x) + 1)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
            self.log_std_max - self.log_std_min
        ) * (log_std + 1)
        
        std = log_std.exp()
        
        # Create Squashed Normal Distribution
        # .rsample() will be available for reparameterization trick (SAC requirement)
        dist = SquashedNormal(mu, std)
        
        return dist
    
# =============================================================================
# 5. BRC Critic Architectures (Distributional RL + Ensemble)
# =============================================================================

class BronetBlock(nn.Module):
    """
    BroNet Residual Block adapted from BRC.
    Structure: Dense -> LN -> Act -> Dense -> LN -> Residual Add
    """
    def __init__(self, hidden_dim, activation=nn.ReLU()):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.act = activation
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # Orthogonal init is standard for these RL networks
            nn.init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x):
        # x: (B, hidden_dim)
        identity = x
        
        out = self.fc1(x)       # (B, hidden_dim)
        out = self.ln1(out)     # (B, hidden_dim)
        out = self.act(out)     # (B, hidden_dim)
        
        out = self.fc2(out)     # (B, hidden_dim)
        out = self.ln2(out)     # (B, hidden_dim)
        
        return out + identity   # Residual connection


class BroNet(nn.Module):
    """
    BroNet Backbone.
    Structure: Input Project -> [BronetBlock] * depth -> Output Project
    """
    def __init__(self, input_dim, hidden_dim, depth, output_dim, activation):
        super().__init__()
        
        # Initial projection
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.ln_in = nn.LayerNorm(hidden_dim)
        self.act = activation
        
        # Residual Blocks
        self.blocks = nn.ModuleList([
            BronetBlock(hidden_dim, activation) for _ in range(depth)
        ])
        
        # Final Output Layer (optional)
        self.output_dim = output_dim
        self.fc_out = nn.Linear(hidden_dim, output_dim)
            
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x):
        # x: (B, input_dim)
        x = self.fc_in(x)   # (B, hidden_dim)
        x = self.ln_in(x)   # (B, hidden_dim)
        x = self.act(x)     # (B, hidden_dim)
        
        for block in self.blocks:
            x = block(x)    # (B, hidden_dim)
            
        x = self.fc_out(x) # (B, output_dim)
            
        return x


class DistributionalCritic(nn.Module):
    """
    Ensemble Distributional Critic for BRC (C51-style).
    Encapsulates multiple BroNets (default 2) for Double Q-Learning.
    """
    def __init__(self, repr_dim, action_dim, task_emb_dim, hidden_dim, depth, num_bins, ensemble_size):
        super().__init__()
        
        # Concatenated input dimension: Image Feature + Action Chunk + Task Embedding
        input_dim = repr_dim + action_dim + task_emb_dim
        
        # Ensemble of Critics (typically 2)
        # Each critic outputs 'num_bins' logits
        self.critics = nn.ModuleList([
            BroNet(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                depth=depth,
                output_dim=num_bins, 
                activation=nn.ReLU()
            ) for _ in range(ensemble_size)
        ])

    def forward(self, obs_feat, action, task_emb):
        """
        Args:
            obs_feat: (B, repr_dim) - Vision Encoder output (gradients detached usually)
            action:   (B, action_dim * chunk_size) - Flattened action chunk
            task_emb: (B, task_emb_dim) - Task Embedding
        Returns:
            logits:   (ensemble_size,B,num_bins) - Unnormalized Log-probabilities for each critic
        """
        # 1. Concatenate Inputs
        # Shape: (B, repr_dim) + (B, act_dim_flat) + (B, task_emb_dim) -> (B, total_input_dim)
        x = torch.cat([obs_feat, action, task_emb], dim=-1)
        
        # 2. Forward pass through each critic in the ensemble
        # Each model returns: (B, num_bins)
        outputs = [critic(x) for critic in self.critics]
        
        # 3. Stack outputs to keep ensemble dimension
        # List[(B, num_bins)] -> (ensemble_size,B,num_bins)
        logits = torch.stack(outputs, dim=0)
        
        return logits