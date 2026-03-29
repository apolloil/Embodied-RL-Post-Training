"""
Plot_Snapshot.py — Comprehensive snapshot visualization for offline RL agents.

For each sampled frame (s, a_data) a sub-folder {idx}/ is created under the
output directory, containing five images:
  {idx}a.png — Observation image
  {idx}b.png — dQ/da gradient vector field with Q-value contours
  {idx}c.png — Actor probability density heatmap (marginal over dims 0-1)
  {idx}d.png — C51 distribution variance heatmap over the action plane
  {idx}e.png — Q-distribution bar chart comparing a_data vs a_ood

Output is saved to the Plot/<OUTPUT_NAME>/ directory.
"""

import os
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"

import sys
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
CONFIG_DIR = str(REPO_ROOT / "conf")

import hydra
import torch
import torch.nn.functional as F
import numpy as np
import random
from pathlib import Path
from omegaconf import DictConfig

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from core.agent import RLAgent
from core.data import BCDataset
from utils.repo_paths import REPO_ROOT

# =====================================================================
# User configuration (edit here)
# =====================================================================
IS_PIXEL_BASED = os.environ.get("IS_PIXEL_BASED", "True").lower() in ("true", "1", "yes")
SNAPSHOT_PATH  = os.environ.get("SNAPSHOT_PATH")
OUTPUT_NAME    = os.environ.get("OUTPUT_NAME", "Serial_TF")
SEED           = int(os.environ.get("SEED", "42"))  # Fixed seed for reproducible frame sampling
N             = 20           # Grid density NxN for the vector field
NUM_PLOTS     = 16
OOD_MIN_DIST  = 0.4         # Minimum Euclidean distance between a_data and a_ood

# State-based agent imports (loaded lazily only when IS_PIXEL_BASED=False)
if not IS_PIXEL_BASED:
    from experiments.state.Serial_TF import StateRLAgent
    from experiments.state.Serial_Bro import FaithfulBRCAgent


# =====================================================================
# Helper functions
# =====================================================================

def load_agent_weights(agent, payload):
    """Load network weights only (skip optimizers), compatible with all snapshot types."""
    agent.actor.load_state_dict(payload['actor'])
    agent.critic.load_state_dict(payload['critic'])
    agent.critic_target.load_state_dict(payload['critic_target'])
    if 'encoder' in payload and hasattr(agent, 'encoder'):
        agent.encoder.load_state_dict(payload['encoder'])
    agent.task_embedding.load_state_dict(payload['task_embedding'])
    if 'log_alpha' in payload:
        agent.log_alpha = payload['log_alpha'].to(agent.device)
        agent.log_alpha.requires_grad = False


def sample_one_frame(dataset):
    """Randomly sample a single frame from BCDataset.

    Returns:
        obs_uint8: Raw uint8 image for display, shape (H, W, 3).
        obs_tensor: Normalized float tensor for inference, shape (3, H, W).
        flat_action: Flattened action chunk, shape (flat_act_dim,).
        task_id: Integer task identifier.
        task_name: Human-readable task name string.
        t: Sampled time step index within the trajectory.
        completion_rate: Estimated task completion ratio at time step t.
        state_vector: numpy float32 (state_dim,) if global_state exists, else None.
    """
    path_idx = random.choice(list(dataset._episodes.keys()))
    episode = random.choice(dataset._episodes[path_idx])

    observations = episode["observation"]
    actions = episode["action"]
    task_id = episode["task_id"]

    T = len(actions)
    k = dataset._chunk_size

    if T < k:
        t = 0
    else:
        t = random.randint(0, T - k)

    obs_uint8 = observations["pixels"][t]
    obs_tensor = dataset.aug(obs_uint8)

    state_vector = None
    if "global_state" in observations:
        state_vector = observations["global_state"][t].astype(np.float32)

    if T >= t + k:
        action_chunk = actions[t: t + k]
    else:
        raw_act_dim = actions.shape[-1]
        action_chunk = np.zeros((k, raw_act_dim), dtype=np.float32)
        end_idx = min(T, t + k)
        action_chunk[:end_idx - t] = actions[t:end_idx]

    flat_action = action_chunk.reshape(-1).astype(np.float32)

    inv_map = {v: k for k, v in dataset.task_map.items()}
    task_name = inv_map.get(task_id, f"task_{task_id}")

    reward = episode.get("reward", None)
    if reward is not None:
        traj_r = np.asarray(reward, dtype=np.float32)
        goal_steps = np.where(traj_r >= 10.0 - 1e-3)[0]
        goal_step = int(goal_steps[0]) if len(goal_steps) > 0 else len(traj_r)
    else:
        goal_step = T
    completion_rate = 1.0 if t >= goal_step else float(t) / max(goal_step, 1)

    return obs_uint8, obs_tensor, flat_action, task_id, task_name, t, completion_rate, state_vector


def encode_observation(agent, obs_tensor, task_id, device,
                       state_vector=None, is_pixel=True):
    """Encode observation and task embedding once for reuse across plots.

    For pixel-based agents: encodes the image through agent.encoder.
    For state-based agents with encoder (StateRLAgent): encodes state through MLP.
    For state-based agents without encoder (FaithfulBRC): uses raw state directly.

    Returns:
        obs_feat: Encoded feature, shape (1, feat_dim).
        task_emb: Task embedding, shape (1, emb_dim).
    """
    tid = torch.tensor([task_id], device=device, dtype=torch.long)
    with torch.no_grad():
        task_emb = agent.task_embedding(tid)
        if is_pixel:
            pixel = obs_tensor.unsqueeze(0).to(device)
            obs_feat = agent.encoder(pixel)
        elif hasattr(agent, 'encoder'):
            state_t = torch.from_numpy(state_vector).float().unsqueeze(0).to(device)
            obs_feat = agent.encoder(state_t)
        else:
            obs_feat = torch.from_numpy(state_vector).float().unsqueeze(0).to(device)
    return obs_feat, task_emb


# =====================================================================
# Compute functions
# =====================================================================

def compute_grad_field(agent, img_feat, task_emb, flat_action, n_grid, device):
    """Compute dQ/da gradient field over the first two action dimensions.

    Returns:
        ax_grid, ay_grid: Meshgrid arrays, shape (N, N).
        U, V: Gradient components, shape (N, N).
        q_surface: Expected Q values on the grid, shape (N, N).
        q_data: Scalar Q(s, a_data).
    """
    a_data_t = torch.from_numpy(flat_action).float().unsqueeze(0).to(device)
    with torch.no_grad():
        q_data = agent._get_expected_q(img_feat, a_data_t, task_emb).item()

    ax_vals = np.linspace(-1.0, 1.0, n_grid)
    ay_vals = np.linspace(-1.0, 1.0, n_grid)
    ax_grid, ay_grid = np.meshgrid(ax_vals, ay_vals)

    n_total = n_grid * n_grid
    base_action = np.tile(flat_action, (n_total, 1))
    base_action[:, 0] = ax_grid.ravel()
    base_action[:, 1] = ay_grid.ravel()

    all_actions = torch.from_numpy(base_action).float().to(device)
    all_actions.requires_grad_(True)

    img_feat_rep = img_feat.detach().expand(n_total, -1)
    task_emb_rep = task_emb.detach().expand(n_total, -1)

    logits = agent.critic(img_feat_rep, all_actions, task_emb_rep)  # (E, B, bins)
    probs = F.softmax(logits, dim=-1)
    support = agent.support.view(1, 1, -1)
    q_per_member = (probs * support).sum(-1)   # (E, B)
    q_mean = q_per_member.mean(0)              # (B,)
    q_mean.sum().backward()

    grad = all_actions.grad[:, :2].cpu().numpy()
    U = grad[:, 0].reshape(n_grid, n_grid)
    V = grad[:, 1].reshape(n_grid, n_grid)
    q_surface = q_mean.detach().cpu().numpy().reshape(n_grid, n_grid)

    return ax_grid, ay_grid, U, V, q_surface, q_data


def _get_actor_dist(agent, obs_feat, task_emb, is_faithful_brc=False):
    """Get actor distribution, handling different agent actor interfaces."""
    if is_faithful_brc:
        actor_input = torch.cat([obs_feat, task_emb], dim=-1)
        return agent.actor(actor_input, temperature=1.0)
    else:
        return agent.actor(task_emb, obs_feat)


def compute_actor_density(agent, obs_feat, task_emb, flat_action, n_grid, device,
                          is_faithful_brc=False):
    """Compute actor marginal probability density over the first two action dims.

    Returns:
        ax_grid, ay_grid: Meshgrid arrays, shape (N, N).
        density_surface: Marginal density values, shape (N, N).
        actor_mean: Actor's mean action, shape (flat_act_dim,) numpy array.
    """
    ax_vals = np.linspace(-0.999, 0.999, n_grid)
    ay_vals = np.linspace(-0.999, 0.999, n_grid)
    ax_grid, ay_grid = np.meshgrid(ax_vals, ay_vals)

    n_total = n_grid * n_grid
    base_action = np.tile(flat_action, (n_total, 1))
    base_action[:, 0] = ax_grid.ravel()
    base_action[:, 1] = ay_grid.ravel()
    all_actions = torch.from_numpy(base_action).float().to(device)

    with torch.no_grad():
        dist = _get_actor_dist(agent, obs_feat, task_emb, is_faithful_brc)
        actor_mean = dist.mean.squeeze(0).cpu().numpy()
        log_prob = dist.log_prob(all_actions)      # (n_total, flat_act_dim)
        log_density = log_prob[:, :2].sum(-1)      # marginal over dims 0-1
        density = log_density.exp()

    density_surface = density.cpu().numpy().reshape(n_grid, n_grid)
    return ax_grid, ay_grid, density_surface, actor_mean


@torch.no_grad()
def compute_q_variance(agent, img_feat, task_emb, flat_action, n_grid, device):
    """Compute C51 distribution variance over the first two action dimensions.

    The variance of a C51 distribution is Var = E[Z^2] - (E[Z])^2,
    averaged across ensemble members.

    Returns:
        ax_grid, ay_grid: Meshgrid arrays, shape (N, N).
        var_surface: Distribution variance on the grid, shape (N, N).
    """
    ax_vals = np.linspace(-1.0, 1.0, n_grid)
    ay_vals = np.linspace(-1.0, 1.0, n_grid)
    ax_grid, ay_grid = np.meshgrid(ax_vals, ay_vals)

    n_total = n_grid * n_grid
    base_action = np.tile(flat_action, (n_total, 1))
    base_action[:, 0] = ax_grid.ravel()
    base_action[:, 1] = ay_grid.ravel()

    all_actions = torch.from_numpy(base_action).float().to(device)
    img_feat_rep = img_feat.expand(n_total, -1)
    task_emb_rep = task_emb.expand(n_total, -1)

    logits = agent.critic(img_feat_rep, all_actions, task_emb_rep)  # (E, B, bins)
    probs = F.softmax(logits, dim=-1)                               # (E, B, bins)
    support = agent.support.view(1, 1, -1)                          # (1, 1, bins)

    eq = (probs * support).sum(-1)                                  # (E, B)
    eq2 = (probs * support ** 2).sum(-1)                            # (E, B)
    var_per_member = eq2 - eq ** 2                                  # (E, B)
    var_mean = var_per_member.mean(0)                                # (B,)

    var_surface = var_mean.cpu().numpy().reshape(n_grid, n_grid)
    return ax_grid, ay_grid, var_surface


@torch.no_grad()
def compute_q_dist_pair(agent, img_feat, task_emb, flat_action, device,
                        min_dist=OOD_MIN_DIST):
    """Compute C51 Q-distributions for a_data and a randomly sampled a_ood.

    a_ood is uniformly sampled from [-1,1]^d with ||a_data - a_ood||_2 >= min_dist.

    Returns:
        support_np: 1-D numpy array of bin centers, shape (num_bins,).
        probs_data: Ensemble-averaged distribution for a_data, shape (num_bins,).
        probs_ood:  Ensemble-averaged distribution for a_ood, shape (num_bins,).
        a_ood: The sampled OOD action, shape (flat_act_dim,) numpy array.
        q_data: Scalar expected Q(s, a_data).
        q_ood:  Scalar expected Q(s, a_ood).
    """
    act_dim = len(flat_action)
    a_ood = None
    for _ in range(10000):
        candidate = np.random.uniform(-1.0, 1.0, size=act_dim).astype(np.float32)
        if np.linalg.norm(candidate - flat_action) >= min_dist:
            a_ood = candidate
            break
    if a_ood is None:
        a_ood = np.random.uniform(-1.0, 1.0, size=act_dim).astype(np.float32)

    a_data_t = torch.from_numpy(flat_action).float().unsqueeze(0).to(device)
    a_ood_t = torch.from_numpy(a_ood).float().unsqueeze(0).to(device)
    both = torch.cat([a_data_t, a_ood_t], dim=0)  # (2, act_dim)

    img_feat_rep = img_feat.expand(2, -1)
    task_emb_rep = task_emb.expand(2, -1)

    logits = agent.critic(img_feat_rep, both, task_emb_rep)  # (E, 2, bins)
    probs = F.softmax(logits, dim=-1)                        # (E, 2, bins)
    probs_avg = probs.mean(0)                                # (2, bins)

    support_np = agent.support.cpu().numpy()
    probs_data = probs_avg[0].cpu().numpy()
    probs_ood = probs_avg[1].cpu().numpy()

    q_data = (probs_avg[0] * agent.support).sum().item()
    q_ood = (probs_avg[1] * agent.support).sum().item()

    return support_np, probs_data, probs_ood, a_ood, q_data, q_ood


# =====================================================================
# Plot functions (each produces a single standalone figure)
# =====================================================================

def plot_observation(obs_uint8, task_name, timestep, completion_rate, output_path):
    """Save observation image as a standalone figure."""
    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
    ax.imshow(obs_uint8)
    ax.set_axis_off()
    if completion_rate is not None:
        ax.set_title(f'{task_name}  (t={timestep}, {completion_rate:.0%})', fontsize=12)
    else:
        ax.set_title(f'{task_name}  (t={timestep})', fontsize=12)
    fig.tight_layout()
    fig.savefig(str(output_path), bbox_inches='tight')
    plt.close(fig)


def plot_gradient_field(ax_grid, ay_grid, U, V, q_surface, q_data,
                        flat_action, a_ood, output_path):
    """Save gradient vector field plot as a standalone figure."""
    fig, ax = plt.subplots(figsize=(8, 7), dpi=120)

    levels = 30
    cf = ax.contourf(ax_grid, ay_grid, q_surface, levels=levels,
                     cmap='viridis', alpha=0.8)
    ax.contour(ax_grid, ay_grid, q_surface, levels=levels,
               colors='k', linewidths=0.3, alpha=0.3)

    magnitude = np.sqrt(U ** 2 + V ** 2)
    mag_max = magnitude.max()
    if mag_max > 1e-8:
        U_norm = U / mag_max
        V_norm = V / mag_max
    else:
        U_norm, V_norm = U, V

    ax.quiver(ax_grid, ay_grid, U_norm, V_norm, magnitude,
              cmap='hot', scale=25, width=0.004, alpha=0.9,
              headwidth=4, headlength=5)

    a0, a1 = flat_action[0], flat_action[1]
    ax.plot(a0, a1, 'r*', markersize=18, markeredgecolor='white',
            markeredgewidth=0.8, zorder=10,
            label=f'a_data ({a0:.2f}, {a1:.2f})  Q={q_data:.2f}')
    ax.annotate(f'Q={q_data:.2f}', xy=(a0, a1),
                xytext=(8, 8), textcoords='offset points',
                fontsize=9, fontweight='bold', color='red',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))

    ax.plot(a_ood[0], a_ood[1], '^', color='cyan', markersize=14,
            markeredgecolor='white', markeredgewidth=0.8, zorder=10,
            label=f'a_ood ({a_ood[0]:.2f}, {a_ood[1]:.2f})')

    ax.set_xlabel('Action dim 0 (Ax)', fontsize=10)
    ax.set_ylabel('Action dim 1 (Ay)', fontsize=10)
    ax.set_title(r'$\nabla_a \, \mathbb{E}[Q(s, a)]$  Gradient Field',
                 fontsize=12, fontweight='bold')
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect('equal')
    ax.legend(loc='upper left', fontsize=8)
    fig.colorbar(cf, ax=ax, shrink=0.85, aspect=20, pad=0.03, label='E[Q(s, a)]')

    fig.tight_layout()
    fig.savefig(str(output_path), bbox_inches='tight')
    plt.close(fig)


def plot_actor_density(ax_grid, ay_grid, density_surface, actor_mean,
                       flat_action, a_ood, output_path):
    """Save actor density heatmap as a standalone figure."""
    fig, ax = plt.subplots(figsize=(8, 7), dpi=120)

    levels = 30
    cf = ax.contourf(ax_grid, ay_grid, density_surface, levels=levels,
                     cmap='magma', alpha=0.9)
    ax.contour(ax_grid, ay_grid, density_surface, levels=levels,
               colors='w', linewidths=0.3, alpha=0.3)

    a0, a1 = flat_action[0], flat_action[1]
    ax.plot(a0, a1, 'r*', markersize=18, markeredgecolor='white',
            markeredgewidth=0.8, zorder=10,
            label=f'a_data ({a0:.2f}, {a1:.2f})')

    mu0, mu1 = actor_mean[0], actor_mean[1]
    ax.plot(mu0, mu1, '*', color='lime', markersize=18, markeredgecolor='white',
            markeredgewidth=0.8, zorder=10,
            label=f'a_actor ({mu0:.2f}, {mu1:.2f})')

    ax.plot(a_ood[0], a_ood[1], '^', color='cyan', markersize=14,
            markeredgecolor='white', markeredgewidth=0.8, zorder=10,
            label=f'a_ood ({a_ood[0]:.2f}, {a_ood[1]:.2f})')

    ax.set_xlabel('Action dim 0 (Ax)', fontsize=10)
    ax.set_ylabel('Action dim 1 (Ay)', fontsize=10)
    ax.set_title(r'$\pi(a \mid s)$  Density', fontsize=12, fontweight='bold')
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect('equal')
    ax.legend(loc='upper left', fontsize=8)
    fig.colorbar(cf, ax=ax, shrink=0.85, aspect=20, pad=0.03, label='Density')

    fig.tight_layout()
    fig.savefig(str(output_path), bbox_inches='tight')
    plt.close(fig)


def plot_variance_heatmap(ax_grid, ay_grid, var_surface, flat_action, a_ood,
                          output_path):
    """Save C51 distribution variance heatmap as a standalone figure."""
    fig, ax = plt.subplots(figsize=(8, 7), dpi=120)

    levels = 30
    cf = ax.contourf(ax_grid, ay_grid, var_surface, levels=levels,
                     cmap='inferno', alpha=0.9)
    ax.contour(ax_grid, ay_grid, var_surface, levels=levels,
               colors='w', linewidths=0.3, alpha=0.3)

    a0, a1 = flat_action[0], flat_action[1]
    ax.plot(a0, a1, 'r*', markersize=18, markeredgecolor='white',
            markeredgewidth=0.8, zorder=10,
            label=f'a_data ({a0:.2f}, {a1:.2f})')

    ax.plot(a_ood[0], a_ood[1], '^', color='cyan', markersize=14,
            markeredgecolor='white', markeredgewidth=0.8, zorder=10,
            label=f'a_ood ({a_ood[0]:.2f}, {a_ood[1]:.2f})')

    ax.set_xlabel('Action dim 0 (Ax)', fontsize=10)
    ax.set_ylabel('Action dim 1 (Ay)', fontsize=10)
    ax.set_title(r'$\mathrm{Var}[Q(s, a)]$  Distribution Variance',
                 fontsize=12, fontweight='bold')
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect('equal')
    ax.legend(loc='upper left', fontsize=8)
    fig.colorbar(cf, ax=ax, shrink=0.85, aspect=20, pad=0.03, label='Variance')

    fig.tight_layout()
    fig.savefig(str(output_path), bbox_inches='tight')
    plt.close(fig)


def plot_q_dist(support, probs_data, probs_ood, flat_action, a_ood,
                q_data, q_ood, output_path):
    """Save Q-distribution bar chart (a_data vs a_ood) as a standalone figure."""
    fig, ax = plt.subplots(figsize=(10, 5), dpi=120)

    n_bins = len(support)
    base_width = (support[-1] - support[0]) / n_bins * 0.85
    bw = base_width / 2

    ax.bar(support - bw / 2, probs_data, width=bw, alpha=0.85,
           color='#1B4F72',
           label=f'Q(s, a_data)  E={q_data:.2f}')
    ax.bar(support + bw / 2, probs_ood, width=bw, alpha=0.85,
           color='#922B21',
           label=f'Q(s, a_ood)  E={q_ood:.2f}')

    max_p = max(probs_data.max(), probs_ood.max())
    ax.set_xlabel('Q-value', fontsize=11)
    ax.set_ylabel('Probability', fontsize=11)
    ax.set_xlim(support[0] - 0.5, support[-1] + 0.5)
    ax.set_ylim(0, max_p * 1.15 + 1e-6)
    ax.legend(loc='upper right', fontsize=10)

    a0, a1 = flat_action[0], flat_action[1]
    o0, o1 = a_ood[0], a_ood[1]
    dist_val = np.linalg.norm(flat_action - a_ood)
    info = (f'a_data=({a0:.2f}, {a1:.2f}, ...)\n'
            f'a_ood =({o0:.2f}, {o1:.2f}, ...)\n'
            f'||a_data - a_ood||={dist_val:.2f}')
    ax.text(0.02, 0.95, info, transform=ax.transAxes, fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))

    ax.set_title('Q-Distribution: a_data vs a_ood', fontsize=12, fontweight='bold')

    fig.tight_layout()
    fig.savefig(str(output_path), bbox_inches='tight')
    plt.close(fig)


# =====================================================================
# Main
# =====================================================================

@hydra.main(config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig):

    device = torch.device(cfg.device)

    # ----- 1. Load snapshot & create Agent -----
    snapshot_path_cfg = SNAPSHOT_PATH or cfg.eval.snapshot_path
    if not snapshot_path_cfg:
        raise ValueError(
            "No snapshot path provided. Set SNAPSHOT_PATH=/path/to/snapshot.pt "
            "or pass eval.snapshot_path=/path/to/snapshot.pt."
        )
    snapshot_path = Path(snapshot_path_cfg).expanduser()
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Snapshot not found: {snapshot_path}")

    print(f"Loading snapshot: {snapshot_path}")
    payload = torch.load(snapshot_path, map_location=device)

    is_faithful_brc = False
    if IS_PIXEL_BASED:
        print("Creating RLAgent (pixel-based) ...")
        agent = RLAgent(device, cfg)
    elif 'encoder' in payload:
        print("Creating StateRLAgent (state-based with MLP encoder) ...")
        agent = StateRLAgent(device, cfg)
    else:
        print("Creating FaithfulBRCAgent (state-based, no encoder) ...")
        agent = FaithfulBRCAgent(device, num_tasks=50)
        is_faithful_brc = True

    load_agent_weights(agent, payload)
    agent.train(False)
    print("Snapshot loaded (network weights only, optimizers skipped).")

    # ----- 2. Load dataset and sample a frame -----
    data_root = REPO_ROOT / cfg.data.root_dir.lstrip('./')

    if cfg.env.task_names == "MT30":
        task_names = cfg.env.MT30
    elif cfg.env.task_names == "MT50":
        task_names = cfg.env.MT50
    else:
        raise ValueError(f"Invalid task names: {cfg.env.task_names}")

    print("Loading dataset for sampling ...")
    dataset = BCDataset(
        path=str(data_root),
        suite="Expert_Goal_Observable",
        scenes=task_names,
        num_demos_per_task=cfg.data.num_demos_per_task,
        chunk_size=cfg.agent.action_chunking.num_queries,
        discount=cfg.train.rl.discount,
    )

    # ----- 3. Output directory -----
    output_dir = REPO_ROOT / "Plot" / OUTPUT_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    act_dim = getattr(agent, 'flat_act_dim', 4)
    print(f"IS_PIXEL_BASED={IS_PIXEL_BASED}, is_faithful_brc={is_faithful_brc}, "
          f"act_dim={act_dim}")
    print(f"Generating {NUM_PLOTS} snapshot folders into {output_dir}\n")

    # ----- 4. Fix random seed for reproducible frame sampling -----
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    print(f"Random seed set to {SEED} — same seed = same sampled frames across snapshots.\n")

    # ----- 5. Sample & plot loop -----
    for idx in range(1, NUM_PLOTS + 1):
        frame_dir = output_dir / str(idx)
        frame_dir.mkdir(parents=True, exist_ok=True)

        obs_uint8, obs_tensor, flat_action, task_id, task_name, \
            timestep, completion_rate, state_vector = sample_one_frame(dataset)

        print(f"[{idx}/{NUM_PLOTS}] task={task_name}, t={timestep}, "
              f"a_data[:4]={np.array2string(flat_action[:4], precision=3)}")

        obs_feat, task_emb = encode_observation(
            agent, obs_tensor, task_id, device,
            state_vector=state_vector, is_pixel=IS_PIXEL_BASED)

        # (b) Gradient field
        ax_grid, ay_grid, U, V, q_surface, q_data = compute_grad_field(
            agent, obs_feat, task_emb, flat_action, N, device)

        # (c) Actor density
        ax_grid_d, ay_grid_d, density_surface, actor_mean = compute_actor_density(
            agent, obs_feat, task_emb, flat_action, N, device,
            is_faithful_brc=is_faithful_brc)

        # (d) Variance
        ax_grid_v, ay_grid_v, var_surface = compute_q_variance(
            agent, obs_feat, task_emb, flat_action, N, device)

        # (e) Q-dist pair
        support_np, probs_data, probs_ood, a_ood, q_data_dist, q_ood = \
            compute_q_dist_pair(agent, obs_feat, task_emb, flat_action, device)

        grad_mag = np.sqrt(U ** 2 + V ** 2)
        print(f"  Q range: [{q_surface.min():.3f}, {q_surface.max():.3f}], "
              f"Q(s,a_data)={q_data:.3f}, "
              f"|grad| range: [{grad_mag.min():.3f}, {grad_mag.max():.3f}]")
        print(f"  Density range: [{density_surface.min():.6f}, {density_surface.max():.6f}], "
              f"a_actor[:2]=({actor_mean[0]:.3f}, {actor_mean[1]:.3f})")
        print(f"  Var range: [{var_surface.min():.3f}, {var_surface.max():.3f}]")
        print(f"  Q_dist: a_data E[Q]={q_data_dist:.3f}, a_ood E[Q]={q_ood:.3f}, "
              f"||diff||={np.linalg.norm(flat_action - a_ood):.3f}")

        plot_observation(obs_uint8, task_name, timestep, completion_rate,
                         frame_dir / f"{idx}a.png")

        plot_gradient_field(ax_grid, ay_grid, U, V, q_surface, q_data,
                            flat_action, a_ood, frame_dir / f"{idx}b.png")

        plot_actor_density(ax_grid_d, ay_grid_d, density_surface, actor_mean,
                           flat_action, a_ood, frame_dir / f"{idx}c.png")

        plot_variance_heatmap(ax_grid_v, ay_grid_v, var_surface, flat_action,
                              a_ood, frame_dir / f"{idx}d.png")

        plot_q_dist(support_np, probs_data, probs_ood, flat_action, a_ood,
                    q_data_dist, q_ood, frame_dir / f"{idx}e.png")

        print(f"  Saved 5 plots to {frame_dir}/")

    del dataset
    print(f"\nDone. {NUM_PLOTS} snapshot folders saved to {output_dir}")


if __name__ == '__main__':
    main()
