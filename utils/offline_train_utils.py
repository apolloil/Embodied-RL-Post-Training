"""
Shared utilities for offline training scripts (OFF_BRC_BC, OFF_TD3BC, OFF_CalQL, OFF_EDAC)
and online training scripts (Serial_Bro, Serial_TF, Parallel_Bro, Parallel_TF).

Extracted from the four offline training scripts to eliminate code duplication.
Includes: task palette setup, reward normalizer warm-up, P/N data loader creation,
batch mixing, Q-distribution visualization, video flushing, offline evaluation,
vis sample selection, dist frame collection, and checkpoint saving.
"""

import pickle
import yaml
import hydra
import torch
import torch.nn.functional as F
import numpy as np
import imageio
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.logger import Logger
from utils.common import VideoRecorder, Every, Timer
from utils.repo_paths import REPO_ROOT, CONF_DIR
from core.wrapper_new import make
from core.data import BCDataset, make_loader
from core.normalizer import RewardNormalizer
from tools.Eval import evaluate


# =====================================================================
# Shared Offline Constants
# =====================================================================
LOG_FREQ        = 1000
EVAL_FREQ       = 25000
SAVE_FREQ       = 25000
DIST_FRAME_FREQ = 200
DIST_VIDEO_FREQ = 10000
SAC_P_RATIO     = 0.25
P_SUITE         = "Expert_Goal_Observable"
N_SUITE         = "Medium_Goal_Observable"


# =====================================================================
# 1. Task Palette & ID Mappings
# =====================================================================

def setup_task_palette(cfg, device):
    """Load task_palette.yaml and build task-name list with ID mappings.

    Reads the global task map from ``conf/task_palette.yaml`` and constructs
    local indices for the selected task set (MT30 or MT50).

    Args:
        cfg:    Hydra DictConfig (must contain ``cfg.env.task_names``).
        device: torch device for the lookup tensor.

    Returns:
        train_task_names: list[str] -- ordered task name strings.
        global_to_local:  dict {global_task_id: local_idx}.
        gid_to_name:      dict {global_task_id: task_name}.
        g2l_tensor:       Tensor of shape ``(max_gid,)`` on *device*,
                          mapping global task IDs to local indices.
    """
    with open(CONF_DIR / "task_palette.yaml", 'r') as f:
        full_task_map = yaml.safe_load(f)

    if cfg.env.task_names == "MT30":
        train_task_names = cfg.env.MT30
    elif cfg.env.task_names == "MT50":
        train_task_names = cfg.env.MT50
    else:
        raise ValueError(f"Invalid task names: {cfg.env.task_names}")

    global_to_local = {}
    gid_to_name = {}
    for local_idx, name in enumerate(train_task_names):
        gid = full_task_map[name]
        global_to_local[gid] = local_idx
        gid_to_name[gid] = name

    max_global_id = max(full_task_map.values()) + 1
    g2l_tensor = torch.zeros(max_global_id, dtype=torch.long, device=device)
    for g_id, l_idx in global_to_local.items():
        g2l_tensor[g_id] = l_idx

    return train_task_names, global_to_local, gid_to_name, g2l_tensor


# =====================================================================
# 1b. State-based offline data loading (for online training scripts)
# =====================================================================

def load_state_offline_buffer(data_root, p_suite, n_suite, task_names, device,
                              task_palette=None, state_dim=39, act_dim=4):
    """Load state-based P + N offline data into a FeatureReplayBuffer.

    Unified version used by Serial_Bro, Serial_TF, Parallel_Bro, Parallel_TF.

    Args:
        data_root:     Path to datasets/ directory.
        p_suite:       Expert dataset subdirectory name (e.g. "new_P").
        n_suite:       Sub-optimal dataset subdirectory name (e.g. "new_N").
        task_names:    List of task name strings.
        device:        torch device.
        task_palette:  Optional dict {task_name: global_task_id}. If None,
                       loaded from conf/task_palette.yaml.
        state_dim:     State observation dimension (default 39 for MetaWorld).
        act_dim:       Action dimension (default 4 for MetaWorld).

    Returns:
        buf:           FeatureReplayBuffer with all loaded transitions.
        reward_seqs:   dict {global_task_id: [reward_array, ...]} for normalizer warmup.
    """
    from core.replay_buffer import FeatureReplayBuffer

    if task_palette is None:
        with open(CONF_DIR / "task_palette.yaml", 'r') as f:
            task_palette = yaml.safe_load(f)

    # Count total transitions
    total = 0
    for suite in [p_suite, n_suite]:
        for name in task_names:
            path = data_root / suite / f"{name}.pkl"
            if not path.exists():
                continue
            with open(path, 'rb') as f:
                data = pickle.load(f)
            for i in range(len(data['observations'])):
                total += len(data['actions'][i])

    buf = FeatureReplayBuffer(
        capacity=total, repr_dim=state_dim,
        act_dim=act_dim, device=device)

    reward_seqs = {}
    for suite in [p_suite, n_suite]:
        for name in task_names:
            path = data_root / suite / f"{name}.pkl"
            if not path.exists():
                print(f"  [WARN] pkl not found, skip: {path}")
                continue
            with open(path, 'rb') as f:
                data = pickle.load(f)
            task_id = task_palette[name]
            for i in range(len(data['observations'])):
                states = data['observations'][i]['global_state']
                actions = data['actions'][i]
                rewards = data['rewards'][i]
                for t in range(len(actions)):
                    terminated = bool(rewards[t] > 20 - 1e-3)
                    buf.add(states[t], actions[t], rewards[t],
                            states[t + 1], terminated, task_id)
                reward_seqs.setdefault(task_id, []).append(rewards)

    print(f"  Offline buffer loaded: {buf.size} transitions "
          f"from {len(task_names)} tasks (P={p_suite}, N={n_suite})")
    return buf, reward_seqs


# =====================================================================
# 2. Reward Normalizer Warm-up
# =====================================================================

def warmup_normalizer(data_root, cfg, train_task_names, global_to_local, agent, p_suite, n_suite):
    """Create a RewardNormalizer and warm it up with the P + N datasets.

    Loads every trajectory from both datasets, feeds the per-episode
    reward sequences into the normalizer so that ``returns_max`` is correctly
    calibrated before training begins.

    Args:
        data_root:        Path to the dataset root directory.
        cfg:              Hydra DictConfig.
        train_task_names: list of task name strings.
        global_to_local:  dict mapping global task IDs to local indices.
        agent:            agent instance (used for ``flat_act_dim``).
        p_suite:          suite name string for the expert dataset.
        n_suite:          suite name string for the suboptimal dataset.

    Returns:
        normalizer: RewardNormalizer (already warmed up).
    """
    normalizer = RewardNormalizer(
        num_tasks=len(train_task_names),
        action_dim=agent.flat_act_dim,
        discount=cfg.train.rl.discount,
        v_max=cfg.train.rl.critic.v_max,
        target_entropy=agent.target_entropy,
    )

    print("Pre-updating Reward Normalizer with P + N datasets...")
    for suite in [p_suite, n_suite]:
        warmup_ds = BCDataset(
            path=str(data_root),
            suite=suite,
            scenes=train_task_names,
            num_demos_per_task=cfg.data.num_demos_per_task,
            chunk_size=cfg.agent.action_chunking.num_queries,
            discount=cfg.train.rl.discount,
        )
        for gid, trajectories in warmup_ds.dump_rewards().items():
            for r_seq in trajectories:
                # BRC original: always bootstrap (no success distinction)
                normalizer.update(global_to_local[gid], r_seq)
        del warmup_ds
    print(f"  returns_max = {normalizer.returns_max}")
    print("Reward Normalizer ready.\n")

    return normalizer


# =====================================================================
# 2b. Setup Offline Infrastructure
# =====================================================================

def setup_offline_infra(cfg, device, train_task_names, global_to_local, agent,
                        p_suite=P_SUITE, n_suite=N_SUITE, sac_p_ratio=SAC_P_RATIO):
    """Create normalizer and P/N data loaders for offline training.

    Returns:
        (normalizer, p_iter, n_iter, B_p, B_n, batch_size)
    """
    data_root = REPO_ROOT / cfg.data.root_dir.lstrip('./')

    normalizer = warmup_normalizer(
        data_root, cfg, train_task_names, global_to_local, agent, p_suite, n_suite)

    batch_size = cfg.train.rl.batch_size
    p_iter, n_iter, B_p, B_n = create_pn_loaders(
        data_root, cfg, train_task_names, batch_size, sac_p_ratio, p_suite, n_suite)

    return normalizer, p_iter, n_iter, B_p, B_n, batch_size


# =====================================================================
# 3. P/N Data Loaders
# =====================================================================

def create_pn_loaders(data_root, cfg, train_task_names, batch_size,
                      sac_p_ratio, p_suite, n_suite):
    """Create P (expert) + N (suboptimal) dual data loaders and batch-split sizes.

    Args:
        data_root:        Path to dataset root.
        cfg:              Hydra DictConfig.
        train_task_names: list of task name strings.
        batch_size:       total batch size per training step.
        sac_p_ratio:      fraction of P data in the mixed SAC/Critic batch.
        p_suite:          suite name string for the expert dataset.
        n_suite:          suite name string for the suboptimal dataset.

    Returns:
        p_iter: iterator over the P (expert) data loader.
        n_iter: iterator over the N (suboptimal) data loader.
        B_p:    int -- number of P samples per mixed batch.
        B_n:    int -- number of N samples per mixed batch.
    """
    p_loader = make_loader(
        path=str(data_root),
        suite=p_suite,
        scenes=train_task_names,
        num_demos_per_task=cfg.data.num_demos_per_task,
        chunk_size=cfg.agent.action_chunking.num_queries,
        batch_size=batch_size,
        num_workers=cfg.train.num_workers,
        discount=cfg.train.rl.discount,
    )
    n_loader = make_loader(
        path=str(data_root),
        suite=n_suite,
        scenes=train_task_names,
        num_demos_per_task=cfg.data.num_demos_per_task,
        chunk_size=cfg.agent.action_chunking.num_queries,
        batch_size=batch_size,
        num_workers=cfg.train.num_workers,
        discount=cfg.train.rl.discount,
    )
    B_p = int(batch_size * sac_p_ratio)
    B_n = batch_size - B_p
    return iter(p_loader), iter(n_loader), B_p, B_n


# =====================================================================
# 4. P/N Batch Mixing
# =====================================================================

def mix_pn_batch(batch_p, batch_n, B_p, B_n, batch_size, device):
    """Mix P and N batches by ratio, and extract a pure-P BC batch.

    Constructs a mixed SAC/Critic batch (``B_p`` samples from P, ``B_n`` from N)
    and a separate pure-P batch for the BC loss.

    Args:
        batch_p:    dict from the P DataLoader.
        batch_n:    dict from the N DataLoader.
        B_p:        number of P samples in the mixed batch.
        B_n:        number of N samples in the mixed batch.
        batch_size: total batch size (used for reshaping actions).
        device:     torch device.

    Returns:
        sac:      dict with keys ``'pixels'``, ``'next_pixels'``, ``'flat_actions'``,
                  ``'rewards'``, ``'mc_returns'``, ``'dones'``, ``'task_ids'``,
                  ``'completion'`` -- all on *device*.
        bc_batch: tuple ``(pixels, flat_actions, None, None, None, task_ids)``
                  -- pure P data for BC loss, on *device*.
    """
    sac_pixels      = torch.cat([batch_p['pixels'][:B_p],      batch_n['pixels'][:B_n]]).to(device, non_blocking=True)
    sac_next_pixels = torch.cat([batch_p['next_pixels'][:B_p], batch_n['next_pixels'][:B_n]]).to(device, non_blocking=True)
    sac_actions     = torch.cat([batch_p['actions'][:B_p],      batch_n['actions'][:B_n]]).to(device, non_blocking=True)
    sac_rewards     = torch.cat([batch_p['rewards'][:B_p],      batch_n['rewards'][:B_n]]).to(device, non_blocking=True)
    sac_mc_returns  = torch.cat([batch_p['mc_returns'][:B_p],   batch_n['mc_returns'][:B_n]]).to(device, non_blocking=True)
    sac_dones       = torch.cat([batch_p['dones'][:B_p],        batch_n['dones'][:B_n]]).to(device, non_blocking=True)
    sac_task_ids    = torch.cat([batch_p['task_id'][:B_p],      batch_n['task_id'][:B_n]]).to(device, non_blocking=True).long().squeeze(-1)
    sac_completion  = torch.cat([batch_p['completion_rate'][:B_p], batch_n['completion_rate'][:B_n]])

    sac = dict(
        pixels=sac_pixels, next_pixels=sac_next_pixels,
        flat_actions=sac_actions.view(batch_size, -1),
        rewards=sac_rewards, mc_returns=sac_mc_returns,
        dones=sac_dones, task_ids=sac_task_ids,
        completion=sac_completion,
    )

    bc_pixels       = batch_p['pixels'].to(device, non_blocking=True)
    bc_flat_actions = batch_p['actions'].to(device, non_blocking=True).view(batch_size, -1)
    bc_task_ids     = batch_p['task_id'].to(device, non_blocking=True).long().squeeze(-1)
    bc_batch = (bc_pixels, bc_flat_actions, None, None, None, bc_task_ids)

    return sac, bc_batch


# =====================================================================
# 5. Q(s, a_random) Computation
# =====================================================================

def compute_random_q_probs(agent, pixels, task_ids, idx, device):
    """Compute Q(s, a_random) distribution for visualization.

    Samples a uniform random action and evaluates it through the critic ensemble.

    Args:
        agent:    the training agent.
        pixels:   batch of pixel observations (already on device).
        task_ids: batch of task IDs (already on device).
        idx:      index of the sample to evaluate.
        device:   torch device.

    Returns:
        Numpy array of shape ``(num_bins,)`` -- averaged Q-distribution probabilities.
    """
    with torch.no_grad():
        feat = agent.encoder(pixels[idx:idx+1])
        temb = agent.task_embedding(task_ids[idx:idx+1])
        a_rand = torch.empty(1, agent.flat_act_dim, device=device).uniform_(-1, 1)
        logits = agent.critic(feat, a_rand, temb)
        return F.softmax(logits, dim=-1).mean(0).squeeze(0).cpu().numpy()


# =====================================================================
# 6. Q-Distribution Visualization
# =====================================================================

def render_q_dist_frame(support, current_probs, target_probs, step,
                        obs_img=None, task_name=None, mc_return=None,
                        random_probs=None, completion_rate=None):
    """Render a single Q-distribution visualization frame for offline training.

    Layout: left half shows the observation image (optional), right half shows
    bar charts of the current Q(s, a_data) distribution, the Bellman target
    distribution, and optionally Q(s, a_random).

    Args:
        support:         1-D numpy array of Q-value bin centers, shape ``(num_bins,)``.
        current_probs:   numpy array ``(num_bins,)`` -- current critic distribution.
        target_probs:    numpy array ``(num_bins,)`` -- Bellman target distribution.
        step:            training step number (displayed in annotation).
        obs_img:         optional HWC uint8 observation image.
        task_name:       optional task name string for the title.
        mc_return:       optional float -- MC return value (drawn as a dashed line).
        random_probs:    optional numpy array ``(num_bins,)`` -- Q(s, a_random) distribution.
        completion_rate: optional float -- task completion rate; negative means failure.

    Returns:
        RGB image as a numpy array, shape ``(H, W, 3)``, uint8.
    """
    if obs_img is not None:
        fig, (ax_img, ax_dist) = plt.subplots(
            1, 2, figsize=(14, 4), dpi=80,
            gridspec_kw={'width_ratios': [1, 2.5]})
        ax_img.imshow(obs_img)
        ax_img.set_axis_off()
        if task_name and completion_rate is not None:
            if completion_rate >= 0:
                title = f'{task_name}  ({completion_rate:.0%})'
            else:
                title = f'{task_name}  (Not Success)'
        elif task_name:
            title = f'Observation ({task_name})'
        else:
            title = 'Observation'
        ax_img.set_title(title, fontsize=10)
    else:
        fig, ax_dist = plt.subplots(figsize=(10, 4), dpi=80)

    n_bins = len(support)
    base_width = (support[-1] - support[0]) / n_bins * 0.85

    if random_probs is not None:
        bw = base_width / 3
        ax_dist.bar(support - bw, current_probs, width=bw,
                    alpha=0.7, color='#4C72B0', label='Q(s, a_data)')
        ax_dist.bar(support, target_probs, width=bw,
                    alpha=0.7, color='#DD8452', label='Target Dist')
        ax_dist.bar(support + bw, random_probs, width=bw,
                    alpha=0.6, color='#FF6B6B', label='Q(s, a_rand)')
        max_p = max(current_probs.max(), target_probs.max(), random_probs.max())
    else:
        ax_dist.bar(support - base_width / 4, current_probs, width=base_width / 2,
                    alpha=0.7, color='#4C72B0', label='Q(s, a_data)')
        ax_dist.bar(support + base_width / 4, target_probs, width=base_width / 2,
                    alpha=0.7, color='#DD8452', label='Target Dist')
        max_p = max(current_probs.max(), target_probs.max())

    if mc_return is not None:
        ax_dist.axvline(x=mc_return, color='#2CA02C', linestyle='--', linewidth=2,
                        label=f'MC Return = {mc_return:.2f}')

    ax_dist.set_xlabel('Q-value', fontsize=10)
    ax_dist.set_ylabel('Probability', fontsize=10)
    ax_dist.legend(loc='upper right', fontsize=9)
    ax_dist.set_xlim(support[0] - 0.5, support[-1] + 0.5)
    ax_dist.set_ylim(0, max_p * 1.15 + 1e-6)

    ax_dist.text(0.02, 0.95, f'Critic Update #{step}',
                 transform=ax_dist.transAxes, fontsize=11, fontweight='bold',
                 verticalalignment='top',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))

    fig.tight_layout()
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3).copy()
    plt.close(fig)
    return img


# =====================================================================
# 7. Dist Video Flushing
# =====================================================================

def flush_dist_video(dist_frames, video_task_gid, gid_to_name,
                     dist_video_dir, label):
    """Write accumulated dist frames to an MP4 video file and clear the buffer.

    Args:
        dist_frames:    list of RGB image arrays (frames).
        video_task_gid: global task ID for the tracked task.
        gid_to_name:    dict mapping global task IDs to task name strings.
        dist_video_dir: Path to the output directory.
        label:          string label for the video filename (e.g. step number).
    """
    if not dist_frames:
        return
    tag = gid_to_name.get(video_task_gid, 'unknown').replace('-', '_')
    vpath = dist_video_dir / f'q_dist_{label}_{tag}.mp4'
    imageio.mimsave(str(vpath), dist_frames, fps=5)
    print(f"  [Video] {vpath.name}  ({len(dist_frames)} frames, task={tag})")
    dist_frames.clear()


# =====================================================================
# 8. Offline Evaluation
# =====================================================================

def run_offline_eval(agent, cfg, train_task_names, step, logger,
                     video_recorder):
    """Run standard offline evaluation: create envs, evaluate each task, log, close.

    Args:
        agent:            the agent to evaluate (will be set to eval mode temporarily).
        cfg:              Hydra DictConfig.
        train_task_names: list of task name strings.
        step:             current training step (used as eval step and env seed).
        logger:           Logger instance.
        video_recorder:   VideoRecorder instance.

    Returns:
        mean_success: float -- mean success rate across all tasks.
    """
    print(f"\n===== Evaluation @ step {step} =====")
    agent.train(False)

    eval_envs = make(
        task_names=train_task_names,
        cameras=cfg.env.cameras,
        img_size=cfg.env.img_size,
        action_repeat=cfg.env.action_repeat,
        seed=cfg.seed + step,
        max_episode_steps=cfg.env.max_episode_steps,
    )

    total_success = 0
    for i, env in enumerate(eval_envs):
        task_name = train_task_names[i]
        avg_reward, success_rate = evaluate(
            env=env, agent=agent,
            num_episodes=cfg.train.rl.num_eval_episodes,
            video_recorder=video_recorder,
            step=step, task_name=task_name,
            num_exec=cfg.agent.action_chunking.num_exec,
        )
        logger.log(f'eval/{task_name}_success', success_rate, step)
        total_success += success_rate
        print(f"  {task_name:<30} | Reward {avg_reward:6.2f} | Success {success_rate:.2%}")

    for env in eval_envs:
        env.close()
    del eval_envs

    mean_success = total_success / max(1, len(train_task_names))
    logger.log('eval/mean_success_rate', mean_success, step)
    logger.dump(step, Ty='eval')
    print(f"  >>> Mean Success: {mean_success:.2%}\n")
    agent.train(True)

    return mean_success


# =====================================================================
# 9. Vis Sample Selection (for offline Q-dist videos)
# =====================================================================

def select_vis_sample(step, freq, task_ids, current_gid, agent, pixels, device):
    """Select a batch sample for Q-distribution visualization.

    Attempts to find a sample matching the currently tracked task.  If the
    tracked task is absent from the batch, falls back to the first sample
    and updates the tracked task ID.

    Args:
        step:        current training step.
        freq:        capture frequency (capture when ``step % freq == 0``).
        task_ids:    1-D tensor of global task IDs in the batch.
        current_gid: currently tracked global task ID (or ``None``).
        agent:       the agent (used for random-Q computation).
        pixels:      batch pixel tensor (on device).
        device:      torch device.

    Returns:
        should_capture:  bool -- whether to capture at this step.
        dist_sample_idx: int -- batch index of the selected sample.
        random_probs_np: numpy array or ``None`` -- Q(s, a_random) distribution.
        video_task_gid:  int -- (possibly updated) tracked task global ID.
    """
    should_capture = (step % freq == 0)
    dist_sample_idx = 0
    random_probs_np = None
    video_task_gid = current_gid

    if should_capture:
        if video_task_gid is None:
            video_task_gid = task_ids[0].item()
        match = (task_ids == video_task_gid).nonzero(as_tuple=True)[0]
        if len(match) > 0:
            dist_sample_idx = match[0].item()
        else:
            dist_sample_idx = 0
            video_task_gid = task_ids[0].item()
        random_probs_np = compute_random_q_probs(
            agent, pixels, task_ids, dist_sample_idx, device)

    return should_capture, dist_sample_idx, random_probs_np, video_task_gid


# =====================================================================
# 10. Dist Frame Collection (for offline Q-dist videos)
# =====================================================================

def collect_dist_frame(metrics, pixels, dist_sample_idx, support_np,
                       video_task_gid, gid_to_name, mc_returns_norm,
                       random_probs_np, completion, step):
    """Collect a single Q-distribution visualization frame from update metrics.

    Pops ``'_current_probs'`` and ``'_target_probs'`` from *metrics* and renders
    a frame.  Returns ``None`` if the keys are absent.

    Args:
        metrics:          dict returned by the agent's update method (modified in-place).
        pixels:           batch pixel tensor (on device).
        dist_sample_idx:  batch index of the sample to visualize.
        support_np:       1-D numpy array of Q-value bin centers.
        video_task_gid:   global task ID of the tracked task.
        gid_to_name:      dict mapping global task IDs to names.
        mc_returns_norm:  1-D tensor of normalized MC returns for the batch.
        random_probs_np:  numpy array or ``None`` -- Q(s, a_random) distribution.
        completion:       1-D tensor of completion rates for the batch.
        step:             current training step.

    Returns:
        frame: RGB numpy array ``(H, W, 3)`` or ``None``.
    """
    if '_current_probs' not in metrics:
        return None
    obs_np = (pixels[dist_sample_idx].cpu().permute(1, 2, 0).numpy() * 255
              ).clip(0, 255).astype(np.uint8)
    frame = render_q_dist_frame(
        support_np,
        metrics.pop('_current_probs'),
        metrics.pop('_target_probs'),
        step,
        obs_img=obs_np,
        task_name=gid_to_name.get(video_task_gid, '?'),
        mc_return=mc_returns_norm[dist_sample_idx].item(),
        random_probs=random_probs_np,
        completion_rate=completion[dist_sample_idx].item(),
    )
    return frame


# =====================================================================
# 11. Offline Checkpoint Saving
# =====================================================================

def save_offline_checkpoint(agent, work_dir, step, prefix):
    """Save an agent snapshot with a step-indexed filename.

    Saves two copies: ``{prefix}_{step}.pt`` and ``snapshot_latest.pt``.

    Args:
        agent:    the agent whose snapshot to save.
        work_dir: Path to the working directory.
        step:     current training step (included in filename).
        prefix:   filename prefix, e.g. ``'snapshot_td3bc'``.
    """
    snapshot = agent.save_snapshot()
    torch.save(snapshot, work_dir / f'{prefix}_{step}.pt')
    torch.save(snapshot, work_dir / 'snapshot_latest.pt')
    print(f"  [Saved] {prefix}_{step}.pt")


# =====================================================================
# 12. Default Offline Log Format
# =====================================================================

def _default_offline_format_log(m, step, num_steps, fps):
    return (
        f"[Step {step:>7d}/{num_steps}]  "
        f"FPS {fps:5.0f}  "
        f"critic {m.get('critic_loss', 0):.4f}  "
        f"actor {m.get('actor_loss', 0):.4f}  "
        f"bc {m.get('bc_loss', 0):.4f}  "
        f"lmbda {m.get('lmbda', 0):.3f}  "
        f"alpha {m.get('alpha', 0):.5f}  "
        f"Q_data {m.get('q_data', 0):+.2f}  "
        f"Q_pi {m.get('q_pi', 0):+.2f}")


# =====================================================================
# 13. Shared Offline Training Loop
# =====================================================================

def run_offline_training(
    *,
    agent, cfg,
    train_task_names, g2l_tensor, gid_to_name,
    normalizer, p_iter, n_iter, B_p, B_n, batch_size,
    checkpoint_prefix,
    update_fn,
    norm_alpha_fn=None,
    format_log_fn=None,
    done_message="Offline Training finished.",
):
    """Shared offline training loop for all four offline scripts.

    Args:
        update_fn:      ``callable(sac, bc_batch, rewards_norm, mc_returns_norm,
                        should_capture, dist_sample_idx) -> metrics``.
        norm_alpha_fn:  ``callable(agent) -> float`` returning the alpha value
                        for reward normalization. Defaults to ``agent.alpha``.
                        TD3BC passes ``lambda a: 0.0``.
        format_log_fn:  ``callable(metrics, step, num_steps, fps) -> str``.
                        Defaults to a generic format.
    """
    if norm_alpha_fn is None:
        norm_alpha_fn = lambda a: a.alpha
    if format_log_fn is None:
        format_log_fn = _default_offline_format_log

    work_dir = Path.cwd()
    device = torch.device(cfg.device)

    logger = Logger(work_dir, use_tb=cfg.train.use_tb)
    video_recorder = VideoRecorder(work_dir)

    num_steps  = cfg.train.rl.num_offline_steps
    log_every  = Every(LOG_FREQ)
    eval_every = Every(EVAL_FREQ)
    save_every = Every(SAVE_FREQ)
    timer = Timer()

    dist_frames    = []
    video_task_gid = None
    dist_video_dir = work_dir / 'dist_videos'
    dist_video_dir.mkdir(parents=True, exist_ok=True)
    support_np = agent.support.cpu().numpy()

    agent.train(True)

    for step in range(1, num_steps + 1):

        sac, bc_batch = mix_pn_batch(
            next(p_iter), next(n_iter), B_p, B_n, batch_size, device)

        local_ids       = g2l_tensor[sac['task_ids']]
        alpha           = norm_alpha_fn(agent)
        rewards_norm    = normalizer.normalize(sac['rewards'], local_ids, alpha)
        mc_returns_norm = normalizer.normalize(sac['mc_returns'], local_ids, alpha)

        should_capture, dist_sample_idx, random_probs_np, video_task_gid = \
            select_vis_sample(step, DIST_FRAME_FREQ, sac['task_ids'],
                              video_task_gid, agent, sac['pixels'], device)

        metrics = update_fn(sac, bc_batch, rewards_norm, mc_returns_norm,
                            should_capture, dist_sample_idx)

        if should_capture:
            frame = collect_dist_frame(
                metrics, sac['pixels'], dist_sample_idx, support_np,
                video_task_gid, gid_to_name, mc_returns_norm,
                random_probs_np, sac['completion'], step)
            if frame is not None:
                dist_frames.append(frame)

        if step % DIST_VIDEO_FREQ == 0 and dist_frames:
            flush_dist_video(dist_frames, video_task_gid, gid_to_name,
                             dist_video_dir, str(step))
            video_task_gid = None

        if log_every(step):
            elapsed = timer.total_time()
            fps = step / (elapsed + 1e-5)
            print(format_log_fn(metrics, step, num_steps, fps))
            for k, v in metrics.items():
                logger.log(f'train/{k}', v, step)
            logger.log('train/fps', fps, step)
            logger.dump(step, Ty='train')

        if eval_every(step):
            run_offline_eval(agent, cfg, train_task_names, step,
                             logger, video_recorder)

        if save_every(step):
            save_offline_checkpoint(agent, work_dir, step, checkpoint_prefix)

    flush_dist_video(dist_frames, video_task_gid, gid_to_name,
                     dist_video_dir, 'final')

    logger.close()
    print(done_message)
