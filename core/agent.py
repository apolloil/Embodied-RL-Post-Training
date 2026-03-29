import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import copy

from .networks import (
    VisionEncoder, 
    TaskEmbedding, 
    GPTBackbone, 
    StochasticActorHead, 
    MLP,
    DistributionalCritic
)

# =====================================================================
# Actor (GPT-based Stochastic Policy)
# =====================================================================

class Actor(nn.Module):
    def __init__(
        self,
        repr_dim,
        act_dim,        
        act_chunk_size,     
        act_hidden_dim,   
        act_num_layers,
        act_log_std_min,
        act_log_std_max,
        gpt_output_dim,
        gpt_layer,
        gpt_head,
        gpt_embd,        
        gpt_block,
        gpt_dropout  
    ):
        super().__init__()
        self._action_token = nn.Parameter(torch.randn(1, 1, repr_dim))

        self._policy = GPTBackbone(
            input_dim=repr_dim,
            output_dim=gpt_output_dim, 
            n_layer=gpt_layer,
            n_head=gpt_head,
            n_embd=gpt_embd,
            block_size=gpt_block,
            dropout=gpt_dropout
        )

        self._action_head = StochasticActorHead(
            input_size=gpt_output_dim,
            output_size=act_dim * act_chunk_size,
            hidden_size=act_hidden_dim,
            num_layers=act_num_layers,
            log_std_min=act_log_std_min,
            log_std_max=act_log_std_max
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight.data)
            if hasattr(m.bias, 'data'):
                m.bias.data.fill_(0.0)

    def forward(self, task_emb, img_feat):
        """
        Args:
            task_emb: (B, repr_dim)
            img_feat: (B, repr_dim)
        Returns:
            dist: SquashedNormal — sample shape (B, act_chunk_size * act_dim)
        """
        B = task_emb.shape[0]
        task_token   = task_emb.unsqueeze(1)                          # (B, 1, D)
        img_token    = img_feat.unsqueeze(1)                          # (B, 1, D)
        action_token = self._action_token.expand(B, -1, -1)          # (B, 1, D)

        features = self._policy(torch.cat([task_token, img_token, action_token], dim=1))
        return self._action_head(features[:, -1, :])


# =====================================================================
# BCAgent (Behavior Cloning — used by old_scripts/train_bc.py)
# =====================================================================

class BCAgent:
    def __init__(self, device, cfg):
        self.device = device
        self.cfg = cfg
        self.use_tb = cfg.train.use_tb
        self.act_chunk_size = cfg.agent.action_chunking.num_queries
        self.act_dim = 4

        self.task_embedding = TaskEmbedding(
            num_tasks=50, embed_dim=cfg.agent.repr_dim
        ).to(device)

        self.encoder = VisionEncoder(
            input_shape=(3, cfg.env.img_size, cfg.env.img_size),
            output_size=cfg.agent.repr_dim,
            pretrained=cfg.agent.vision_encoder.pretrained,
            freeze=cfg.agent.vision_encoder.freeze
        ).to(device)

        self.actor = Actor(
            repr_dim=cfg.agent.repr_dim,
            act_dim=self.act_dim,
            act_chunk_size=cfg.agent.action_chunking.num_queries,
            act_hidden_dim=cfg.agent.actor.hidden_size,
            act_num_layers=cfg.agent.actor.num_layers,
            act_log_std_min=cfg.agent.actor.log_std_min,
            act_log_std_max=cfg.agent.actor.log_std_max,
            gpt_output_dim=cfg.agent.gpt.output_dim,
            gpt_layer=cfg.agent.gpt.n_layer,
            gpt_head=cfg.agent.gpt.n_head,
            gpt_embd=cfg.agent.gpt.n_embd,
            gpt_block=cfg.agent.gpt.block_size,
            gpt_dropout=cfg.agent.gpt.dropout
        ).to(device)

        params = (
            list(self.task_embedding.parameters()) +
            list(self.encoder.parameters()) +
            list(self.actor.parameters())
        )
        self.optimizer = torch.optim.AdamW(
            params, lr=cfg.train.optimizer.lr,
            weight_decay=cfg.train.optimizer.weight_decay
        )
        self.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.task_embedding.train(training)
        self.actor.train(training)

    def update(self, loader_iter, step=None):
        metrics = {}
        batch = next(loader_iter)
        images   = batch['pixels'].to(self.device)
        actions  = batch['actions'].to(self.device)
        task_ids = batch['task_id'].to(self.device).long().squeeze(-1)

        img_feat = self.encoder(images)
        task_emb = self.task_embedding(task_ids)
        dist = self.actor(task_emb, img_feat)

        flat_actions = actions.view(actions.size(0), -1)
        loss = -dist.log_prob(flat_actions).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.use_tb:
            metrics['train/loss'] = loss.item()
            metrics['train/std'] = dist.scale.mean().item()
        return metrics

    @torch.no_grad()
    def act(self, obs, task_id, eval_mode=True):
        self.train(False)
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float().to(self.device)
        obs = obs.unsqueeze(0) / 255.0
        task_ids = torch.tensor([task_id], device=self.device).long()

        img_feat = self.encoder(obs)
        task_emb = self.task_embedding(task_ids)
        dist = self.actor(task_emb, img_feat)
        flat_action = dist.mean if eval_mode else dist.sample()
        action_chunk = flat_action.view(self.act_chunk_size, self.act_dim).cpu().numpy()
        self.train(True)
        return action_chunk

    def save_snapshot(self):
        return {
            'actor': self.actor.state_dict(),
            'encoder': self.encoder.state_dict(),
            'task_embedding': self.task_embedding.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }

    def load_snapshot(self, payload):
        self.actor.load_state_dict(payload['actor'])
        self.encoder.load_state_dict(payload['encoder'])
        self.task_embedding.load_state_dict(payload['task_embedding'])
        self.optimizer.load_state_dict(payload['optimizer'])


# =====================================================================
# RLAgent (SAC + Distributional Critic + optional TD3BC)
# =====================================================================

class RLAgent:
    def __init__(self, device, cfg):
        self.device = device
        self.cfg = cfg
        self.act_chunk_size = cfg.agent.action_chunking.num_queries
        self.raw_act_dim = 4
        self.flat_act_dim = self.raw_act_dim * self.act_chunk_size

        # RL hyper-parameters
        self.gamma    = cfg.train.rl.discount ** self.act_chunk_size
        self.tau      = cfg.train.rl.tau
        self.num_bins = cfg.train.rl.critic.num_bins
        self.v_min    = cfg.train.rl.critic.v_min
        self.v_max    = cfg.train.rl.critic.v_max
        self.support  = torch.linspace(self.v_min, self.v_max, self.num_bins).to(device)

        self.freeze_encoder = cfg.train.rl.encoder_freeze

        # Target policy smoothing (default off; ON_SO2 sets sigma>0)
        self.target_noise_sigma = 0.0
        self.target_noise_clip  = 0.0

        # Ensemble aggregation: min (pessimistic) vs mean
        self.use_min_q = True

        # --- Build everything ---
        self._init_modules(cfg)
        if self.freeze_encoder:
            self.encoder.requires_grad_(False)
            self.task_embedding.requires_grad_(False)
        self._init_optimizers(cfg)

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_modules(self, cfg):
        """Create all network modules."""
        self.task_embedding = TaskEmbedding(
            num_tasks=50, embed_dim=cfg.agent.repr_dim
        ).to(self.device)

        self.encoder = VisionEncoder(
            input_shape=(3, cfg.env.img_size, cfg.env.img_size),
            output_size=cfg.agent.repr_dim,
            pretrained=cfg.agent.vision_encoder.pretrained,
            freeze=cfg.agent.vision_encoder.freeze
        ).to(self.device)

        self.actor = Actor(
            repr_dim=cfg.agent.repr_dim,
            act_dim=self.raw_act_dim,
            act_chunk_size=self.act_chunk_size,
            act_hidden_dim=cfg.agent.actor.hidden_size,
            act_num_layers=cfg.agent.actor.num_layers,
            act_log_std_min=cfg.agent.actor.log_std_min,
            act_log_std_max=cfg.agent.actor.log_std_max,
            gpt_output_dim=cfg.agent.gpt.output_dim,
            gpt_layer=cfg.agent.gpt.n_layer,
            gpt_head=cfg.agent.gpt.n_head,
            gpt_embd=cfg.agent.gpt.n_embd,
            gpt_block=cfg.agent.gpt.block_size,
            gpt_dropout=cfg.agent.gpt.dropout
        ).to(self.device)

        self.critic = DistributionalCritic(
            repr_dim=cfg.agent.repr_dim,
            action_dim=self.flat_act_dim,
            task_emb_dim=cfg.agent.repr_dim,
            hidden_dim=cfg.train.rl.critic.hidden_dim,
            depth=cfg.train.rl.critic.depth,
            num_bins=self.num_bins,
            ensemble_size=cfg.train.rl.critic.ensemble_size
        ).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)

        self.log_alpha = torch.tensor(
            np.log(cfg.train.rl.init_temperature)
        ).to(self.device)
        self.log_alpha.requires_grad = True

        te = cfg.train.rl.target_entropy
        self.target_entropy = -self.flat_act_dim if te is None else te

    def _init_optimizers(self, cfg):
        """Initialize optimizers for actor, critic, and alpha.

        Encoder and TaskEmbedding parameters are included in the critic
        optimizer so that the critic loss back-propagates through the encoder.
        The actor loss uses detached features, leaving the encoder unaffected.
        """
        rl = cfg.train.rl

        # Actor optimizer (only actor params)
        self.actor_opt = torch.optim.AdamW(
            self.actor.parameters(),
            lr=rl.optimizer.lr_actor,
            weight_decay=rl.optimizer.weight_decay,
        )

        # Critic optimizer (+ encoder/task_emb when not frozen)
        critic_params = list(self.critic.parameters())
        if not self.freeze_encoder:
            critic_params += list(self.encoder.parameters())
            critic_params += list(self.task_embedding.parameters())
        self.critic_opt = torch.optim.AdamW(
            critic_params,
            lr=rl.optimizer.lr_critic,
            weight_decay=rl.optimizer.weight_decay,
        )

        # Alpha optimizer
        self.alpha_opt = torch.optim.Adam(
            [self.log_alpha], lr=rl.optimizer.lr_temp, betas=(0.5, 0.999))

    # ------------------------------------------------------------------
    # Properties & mode switching
    # ------------------------------------------------------------------

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def train(self, training=True):
        self.training = training
        if self.freeze_encoder:
            self.encoder.eval()
            self.task_embedding.eval()
        else:
            self.encoder.train(training)
            self.task_embedding.train(training)
        self.actor.train(training)
        self.critic.train(training)

    # ------------------------------------------------------------------
    # Shared: C51 Distributional Critic Update
    # ------------------------------------------------------------------

    def _update_critic(self, obs, action, reward, next_obs, done, task_emb,
                       return_dists=False, dist_sample_idx=0):
        """
        C51 critic update.  obs / task_emb **may** carry encoder gradients —
        if they do, encoder will be updated via critic_opt.

        Args:
            return_dists:    if True, metrics will contain '_current_probs' and
                             '_target_probs' for Q-distribution visualisation.
            dist_sample_idx: which sample in the batch to snapshot (for vis).
        """
        metrics = {}

        # ---------- Target computation (no grad) ----------
        with torch.no_grad():
            next_dist  = self.actor(task_emb, next_obs)
            next_action   = next_dist.sample()
            # Target policy smoothing (optional, controlled by sigma > 0)
            if self.target_noise_sigma > 0:
                noise = (torch.randn_like(next_action) * self.target_noise_sigma
                         ).clamp(-self.target_noise_clip, self.target_noise_clip)
                next_action = (next_action + noise).clamp(-1.0, 1.0)
            next_log_prob = next_dist.log_prob(next_action).sum(-1, keepdim=True)

            metrics['q_next'] = self._get_expected_q(
                next_obs, next_action, task_emb).mean().item()

            next_logits = self.critic_target(next_obs, next_action, task_emb)
            next_probs_all = F.softmax(next_logits, dim=-1)            # (E, B, bins)
            if self.use_min_q:
                sup_t = self.support.view(1, 1, -1)
                eq = (next_probs_all * sup_t).sum(-1)                  # (E, B)
                min_idx = eq.argmin(dim=0)                             # (B,)
                B_t = next_logits.size(1)
                next_probs = next_probs_all[min_idx, torch.arange(B_t, device=next_logits.device)]
            else:
                next_probs = next_probs_all.mean(dim=0)                # (B, bins)

            alpha_val      = self.alpha.detach()
            next_support   = self.support.unsqueeze(0) - alpha_val * next_log_prob
            target_support = reward + self.gamma * next_support
            target_support = target_support.clamp(self.v_min, self.v_max)

            # C51 projection
            delta_z = float(self.v_max - self.v_min) / (self.num_bins - 1)
            b = torch.clamp((target_support - self.v_min) / delta_z, 0, self.num_bins - 1)
            l = b.floor().long()
            u = torch.clamp(l + 1, max=self.num_bins - 1)
            d = b - l.float()

            target_dist = torch.zeros_like(next_probs)
            target_dist.scatter_add_(1, l, next_probs * (1.0 - d))
            target_dist.scatter_add_(1, u, next_probs * d)

        # ---------- Critic loss ----------
        current_logits = self.critic(obs, action, task_emb)

        with torch.no_grad():
            metrics['q_data'] = self._get_expected_q(obs, action, task_emb).mean().item()

        # Capture distributions for visualisation BEFORE backward
        if return_dists:
            with torch.no_grad():
                cur_probs = F.softmax(current_logits, dim=-1).mean(dim=0)  # (B, bins)
                idx = dist_sample_idx
                metrics['_current_probs'] = cur_probs[idx].cpu().numpy()
                metrics['_target_probs']  = target_dist[idx].cpu().numpy()

        current_log_probs = F.log_softmax(current_logits, dim=-1)
        critic_loss = -(target_dist.unsqueeze(0) * current_log_probs).sum(-1).mean()

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        metrics['critic_loss'] = critic_loss.item()
        return metrics

    # ------------------------------------------------------------------
    # Offline SAC Update  (accepts raw pixels)
    # ------------------------------------------------------------------

    def update(self, batch, bc_alpha=2.5, return_dists=False,
               dist_sample_idx=0, use_bc_loss=True, bc_batch=None,
               bc_weight=1.0):
        """
        Offline SAC update.  Accepts raw pixels.

        Gradient design:
            Critic loss  →  encoder / task_emb  (trainable when not frozen)
            Actor  loss  →  actor only           (features are detached)

        use_bc_loss=True  (BC):  actor_loss = bc_weight * MSE_bc + λ·SAC_loss
        use_bc_loss=False (Pure SAC): actor_loss = SAC_loss

        bc_batch: optional separate (pixels, actions, _, _, _, task_ids) for
                  BC loss.  When provided, BC loss uses this data instead of
                  the main ``batch``.  Backward-compatible: None → old behaviour.
        bc_weight: scalar in [0, 1] that multiplies the BC loss term (default 1.0).
        """
        metrics = {}
        pixels, action, reward, next_pixels, done, task_ids = batch
        metrics['reward'] = reward.mean().item()

        # --- Encode ---
        if self.freeze_encoder:
            with torch.no_grad():
                img_feat      = self.encoder(pixels)
                next_img_feat = self.encoder(next_pixels)
                task_emb      = self.task_embedding(task_ids)
        else:
            # Current state WITH grad → encoder updated through critic
            img_feat = self.encoder(pixels)
            task_emb = self.task_embedding(task_ids)
            # Next state NO grad (target path only)
            with torch.no_grad():
                next_img_feat = self.encoder(next_pixels)

        # ===== 1. Critic =====
        critic_metrics = self._update_critic(
            img_feat, action, reward, next_img_feat, done, task_emb,
            return_dists=return_dists, dist_sample_idx=dist_sample_idx)
        metrics.update(critic_metrics)

        # ===== 2. Soft Update =====
        self._soft_update(self.critic, self.critic_target, self.tau)

        # ===== 3. Actor (detached img, fresh task_emb from updated params) =====
        img_feat_d = img_feat.detach()
        task_emb_fresh = self.task_embedding(task_ids)

        dist       = self.actor(task_emb_fresh, img_feat_d)
        new_action = dist.rsample()
        new_log_prob = dist.log_prob(new_action).sum(-1, keepdim=True)

        q_value  = self._get_expected_q(img_feat_d, new_action, task_emb_fresh)
        sac_loss = (self.alpha.detach() * new_log_prob - q_value).mean()

        if use_bc_loss:
            if bc_batch is not None:
                bc_pix, bc_act, _, _, _, bc_tid = bc_batch
                with torch.no_grad():
                    bc_feat = self.encoder(bc_pix).detach()
                    bc_temb = self.task_embedding(bc_tid).detach()
                bc_dist = self.actor(bc_temb, bc_feat)
                bc_loss = F.mse_loss(bc_dist.mean, bc_act)
            else:
                bc_loss = F.mse_loss(dist.mean, action)
            lmbda   = bc_alpha / (q_value.abs().mean().detach() + 1e-8)
            actor_loss = bc_weight * bc_loss + lmbda * sac_loss
            metrics['bc_loss']   = bc_loss.item()
            metrics['bc_weight'] = bc_weight
            metrics['lmbda']     = lmbda.item()
        else:
            actor_loss = sac_loss
            metrics['bc_loss']   = 0.0
            metrics['bc_weight'] = 0.0
            metrics['lmbda']     = 0.0

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        metrics['actor_loss'] = actor_loss.item()
        metrics['sac_loss']   = sac_loss.item()
        metrics['q_pi']       = q_value.mean().item()

        # ===== 4. Alpha =====
        alpha_loss = (self.alpha * (-new_log_prob - self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        metrics['alpha_loss'] = alpha_loss.item()
        metrics['alpha']      = self.alpha.item()

        return metrics

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _soft_update(self, source, target, tau):
        for p, tp in zip(source.parameters(), target.parameters()):
            tp.data.copy_(tau * p.data + (1 - tau) * tp.data)

    def _get_expected_q(self, obs_feat, action, task_emb):
        """Distributional → scalar expected Q.

        Aggregation over ensemble controlled by ``self.use_min_q``:
            True  (default) → min  over ensemble members (pessimistic, for offline)
            False           → mean over ensemble members (for online)
        """
        logits = self.critic(obs_feat, action, task_emb)          # (E, B, bins)
        probs  = F.softmax(logits, dim=-1)                        # (E, B, bins)
        support = self.support.view(1, 1, -1)                     # (1, 1, bins)
        q_per_member = (probs * support).sum(-1)                  # (E, B)
        if self.use_min_q:
            return q_per_member.min(0).values.unsqueeze(-1)       # (B, 1)
        return q_per_member.mean(0, keepdim=True).T               # (B, 1)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def act(self, obs, task_id, eval_mode=True):
        """Standard interface for raw observations (used by Eval.py)."""
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float().to(self.device)
        obs = obs.unsqueeze(0) / 255.0
        task_ids = torch.tensor([task_id], device=self.device).long()

        img_feat = self.encoder(obs)
        task_emb = self.task_embedding(task_ids)
        dist = self.actor(task_emb, img_feat)

        flat_action = dist.mean if eval_mode else dist.sample()
        return flat_action.view(self.act_chunk_size, self.raw_act_dim).cpu().numpy()

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save_snapshot(self):
        return {
            'actor':          self.actor.state_dict(),
            'critic':         self.critic.state_dict(),
            'critic_target':  self.critic_target.state_dict(),
            'encoder':        self.encoder.state_dict(),
            'task_embedding': self.task_embedding.state_dict(),
            'log_alpha':      self.log_alpha,
            'actor_opt':      self.actor_opt.state_dict(),
            'critic_opt':     self.critic_opt.state_dict(),
            'alpha_opt':      self.alpha_opt.state_dict(),
        }

    def load_snapshot(self, payload):
        self.actor.load_state_dict(payload['actor'])
        self.critic.load_state_dict(payload['critic'])
        self.critic_target.load_state_dict(payload['critic_target'])
        self.encoder.load_state_dict(payload['encoder'])
        self.task_embedding.load_state_dict(payload['task_embedding'])
        self.log_alpha = payload['log_alpha'].to(self.device)
        self.actor_opt.load_state_dict(payload['actor_opt'])
        self.critic_opt.load_state_dict(payload['critic_opt'])
        self.alpha_opt.load_state_dict(payload['alpha_opt'])
