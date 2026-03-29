"""
Shared utilities for online fine-tuning scripts (ON_BRC_BC, ON_CalQL, ON_SO2).

Provides:
  - Q-distribution visualization (dual-agent)
  - Online evaluation, checkpoint saving, batch conversion
  - Agent setup, buffer/normalizer setup, data loader helpers
  - Common training loop (run_online_training)
"""

import os
import pickle as pkl
import yaml

import hydra
import torch
import torch.nn.functional as F
import numpy as np
import imageio
from pathlib import Path
from omegaconf import read_write

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from core.wrapper_new import make
from core.data import make_loader
from core.replay_buffer import ParallelRawReplayBuffer
from tools.Eval import evaluate
from utils.logger import Logger
from utils.common import VideoRecorder
from utils.repo_paths import REPO_ROOT, CONF_DIR
from core.normalizer import RewardNormalizer
from utils.offline_train_utils import warmup_normalizer


# =====================================================================
# 1. Dual-Agent Q-Distribution Visualization
# =====================================================================

def render_dual_q_frame(support, probs_train, probs_ref,
                        env_step, obs_img, task_name):
    """Render a dual-critic Q-distribution visualization frame.

    Layout (2x2 grid):
        Left column : observation image (spanning both rows)
        Top-right   : Critic A (training agent) Q-distributions
        Bottom-right: Critic B (frozen reference) Q-distributions

    Each critic panel shows three action distributions:
        a_random   (light red)  -- uniformly sampled action
        a_snapshot (blue)       -- reference agent's deterministic action
        a_current  (green)      -- training agent's deterministic action
    """
    fig = plt.figure(figsize=(12, 7), dpi=80)
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 2.2], hspace=0.35, wspace=0.25)

    ax_img = fig.add_subplot(gs[:, 0])
    ax_img.imshow(obs_img)
    ax_img.set_axis_off()
    ax_img.set_title(f'{task_name} | step {env_step}', fontsize=11)

    ax_tr = fig.add_subplot(gs[0, 1])
    ax_rf = fig.add_subplot(gs[1, 1])

    n_bins = len(support)
    bw = (support[-1] - support[0]) / n_bins * 0.85 / 3

    def _panel(ax, pd, title):
        ax.bar(support - bw, pd['rand'], width=bw,
               alpha=0.6, color='#FF6B6B', label='a_random')
        ax.bar(support, pd['ref'], width=bw,
               alpha=0.7, color='#4C72B0', label='a_snapshot')
        ax.bar(support + bw, pd['curr'], width=bw,
               alpha=0.7, color='#2CA02C', label='a_current')
        mx = max(pd['rand'].max(), pd['ref'].max(), pd['curr'].max())
        ax.set_ylim(0, mx * 1.15 + 1e-6)
        ax.set_xlim(support[0] - 0.5, support[-1] + 0.5)
        ax.set_xlabel('Q-value', fontsize=9)
        ax.set_ylabel('Prob', fontsize=9)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.legend(loc='upper right', fontsize=7)

    _panel(ax_tr, probs_train, 'Critic A (Training)')
    _panel(ax_rf, probs_ref,   'Critic B (Frozen Ref)')

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3).copy()
    plt.close(fig)
    return img


def capture_q_dists(obs, task_id, agent, agent_ref, device):
    """Compute 6 Q-distributions (3 actions x 2 critics) for dual-agent visualization."""
    with torch.no_grad():
        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(device) / 255.0
        tid_t = torch.tensor([task_id], device=device).long()

        feat_a = agent.encoder(obs_t)
        temb_a = agent.task_embedding(tid_t)
        feat_b = agent_ref.encoder(obs_t)
        temb_b = agent_ref.task_embedding(tid_t)

        flat_dim = agent.flat_act_dim
        a_rand = torch.empty(1, flat_dim, device=device).uniform_(-1, 1)

        chunk_ref = agent_ref.act(obs, task_id, eval_mode=True)
        a_ref = torch.from_numpy(chunk_ref.flatten()).float().unsqueeze(0).to(device)

        chunk_curr = agent.act(obs, task_id, eval_mode=True)
        a_curr = torch.from_numpy(chunk_curr.flatten()).float().unsqueeze(0).to(device)

        pt, pr = {}, {}
        for a, name in [(a_rand, 'rand'), (a_ref, 'ref'), (a_curr, 'curr')]:
            lo_a = agent.critic(feat_a, a, temb_a)
            pt[name] = F.softmax(lo_a, dim=-1).mean(0).squeeze(0).cpu().numpy()
            lo_b = agent_ref.critic(feat_b, a, temb_b)
            pr[name] = F.softmax(lo_b, dim=-1).mean(0).squeeze(0).cpu().numpy()

    return pt, pr


# =====================================================================
# 2. Online Evaluation
# =====================================================================

def run_online_eval(agent, cfg, train_task_names, step, global_step,
                    logger, video_recorder, num_exec):
    """Run evaluation across all tasks during online training."""
    num_tasks = len(train_task_names)
    agent.train(False)

    eval_envs = make(
        task_names=train_task_names,
        cameras=cfg.env.cameras, img_size=cfg.env.img_size,
        action_repeat=cfg.env.action_repeat,
        seed=cfg.seed + 10000 + step,
        max_episode_steps=cfg.env.max_episode_steps)

    total_sr = 0
    for i, env in enumerate(eval_envs):
        tn = train_task_names[i]
        _, s_rate = evaluate(
            env=env, agent=agent,
            num_episodes=cfg.train.rl.num_eval_episodes,
            video_recorder=video_recorder,
            step=global_step, task_name=tn, num_exec=num_exec)
        logger.log(f'eval/{tn}_success', s_rate, global_step)
        total_sr += s_rate

    for env in eval_envs:
        env.close()
    del eval_envs

    mean_sr = total_sr / max(1, num_tasks)
    logger.log('eval/mean_success_rate', mean_sr, global_step)
    logger.dump(global_step, Ty='eval')
    print(f"  [Eval@S{global_step}] MeanSR={mean_sr:.1%}")
    agent.train(True)
    return mean_sr


# =====================================================================
# 3. Checkpoint Saving
# =====================================================================

def save_online_checkpoint(agent, work_dir, step, prefix):
    """Save ``{prefix}_{step}.pt`` and ``snapshot_latest.pt``."""
    snap = agent.save_snapshot()
    torch.save(snap, work_dir / f'{prefix}_{step}.pt')
    torch.save(snap, work_dir / 'snapshot_latest.pt')


# =====================================================================
# 4. Batch-to-Device Conversion
# =====================================================================

def load_batch_to_device(batch, device):
    """Move a DataLoader batch dict to device and flatten actions.

    Returns ``(imgs, nexts, acts, rews, dones, gids)``.
    """
    imgs  = batch['pixels'].to(device, non_blocking=True)
    nexts = batch['next_pixels'].to(device, non_blocking=True)
    acts  = batch['actions'].to(device, non_blocking=True)
    acts  = acts.view(acts.size(0), -1)
    rews  = batch['rewards'].to(device, non_blocking=True)
    dones = batch['dones'].to(device, non_blocking=True)
    gids  = batch['task_id'].to(device, non_blocking=True).long().squeeze(-1)
    return imgs, nexts, acts, rews, dones, gids


# =====================================================================
# 5. Reference Agent Freezing
# =====================================================================

def freeze_agent(agent_ref):
    """Set eval mode and disable all gradients."""
    agent_ref.train(False)
    for m in [agent_ref.actor, agent_ref.critic, agent_ref.critic_target,
              agent_ref.encoder, agent_ref.task_embedding]:
        for p in m.parameters():
            p.requires_grad_(False)


# =====================================================================
# 6. Linear Decay
# =====================================================================

def get_decay_weight(step, decay_steps):
    """``max(0, 1 - step / decay_steps)``.  Returns 1.0 if *decay_steps* <= 0."""
    if decay_steps <= 0:
        return 1.0
    return max(0.0, 1.0 - step / decay_steps)


# =====================================================================
# 7. Dual Agent Setup
# =====================================================================

def setup_dual_agents(cfg, device, AgentCls, *, AgentRefCls=None,
                      agent_kwargs=None, ref_kwargs=None,
                      use_min_q=False, snapshot_extra_fn=None):
    """Create training agent A (encoder unfrozen) + frozen reference agent B.

    Loads pretrained snapshot into both, then completely freezes B.
    *snapshot_extra_fn(agent, payload)* handles script-specific extras
    (e.g. CalQL's ``log_alpha_prime`` restoration).
    """
    if AgentRefCls is None:
        AgentRefCls = AgentCls
    if agent_kwargs is None:
        agent_kwargs = {}
    if ref_kwargs is None:
        ref_kwargs = agent_kwargs

    with read_write(cfg):
        cfg.train.rl.encoder_freeze = False
    agent = AgentCls(device, cfg, **agent_kwargs)
    agent.use_min_q = use_min_q
    agent.target_entropy = -agent.flat_act_dim / 2

    with read_write(cfg):
        cfg.train.rl.encoder_freeze = True
    agent_ref = AgentRefCls(device, cfg, **ref_kwargs)
    with read_write(cfg):
        cfg.train.rl.encoder_freeze = False

    snapshot_path_cfg = cfg.train.rl.rl_snapshot
    if not snapshot_path_cfg:
        raise ValueError(
            "train.rl.rl_snapshot is not set. "
            "Online fine-tuning scripts require an explicit checkpoint path, "
            "e.g. train.rl.rl_snapshot=/path/to/snapshot.pt."
        )

    snapshot_path = Path(snapshot_path_cfg).expanduser()
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Snapshot not found: {snapshot_path}")

    payload = torch.load(snapshot_path, map_location=device)
    for ag in [agent, agent_ref]:
        ag.actor.load_state_dict(payload['actor'])
        ag.critic.load_state_dict(payload['critic'])
        ag.critic_target.load_state_dict(payload['critic_target'])
        ag.encoder.load_state_dict(payload['encoder'])
        ag.task_embedding.load_state_dict(payload['task_embedding'])
    with torch.no_grad():
        agent.log_alpha.copy_(payload['log_alpha'].to(device))
    if snapshot_extra_fn is not None:
        snapshot_extra_fn(agent, payload)
    print(f"Snapshot loaded: {snapshot_path}")

    freeze_agent(agent_ref)
    return agent, agent_ref


# =====================================================================
# 8. Online Infrastructure
# =====================================================================

def setup_online_infra(cfg, device, num_tasks, global_to_local,
                       train_task_names, agent, *, p_suite=None, n_suite=None,
                       capacity_per_task=50_000, warmup_per_task=5000):
    """Create replay buffer + normalizer.

    When *p_suite* and *n_suite* are provided the normalizer is warmed up
    from the offline datasets.  When both are ``None`` a cold normalizer
    (no offline warmup) is returned — suitable for pure online training
    without any offline data mixing.

    Returns ``(buffer, normalizer, start_training, data_root)``.
    """
    global_task_ids = [None] * num_tasks
    for gid, lid in global_to_local.items():
        global_task_ids[lid] = gid

    buffer = ParallelRawReplayBuffer(
        num_tasks=num_tasks,
        capacity_per_task=capacity_per_task,
        img_height=cfg.env.img_size, img_width=cfg.env.img_size,
        act_dim=4, device=device,
        global_task_ids=global_task_ids)

    start_training = warmup_per_task * num_tasks

    data_root = REPO_ROOT / cfg.data.root_dir.lstrip('./')

    if p_suite is not None and n_suite is not None:
        normalizer = warmup_normalizer(
            data_root, cfg, train_task_names, global_to_local, agent,
            p_suite, n_suite)
    else:
        normalizer = RewardNormalizer(
            num_tasks=num_tasks,
            action_dim=agent.flat_act_dim,
            discount=cfg.train.rl.discount,
            v_max=cfg.train.rl.critic.v_max,
            target_entropy=agent.target_entropy,
        )
        print("Normalizer: cold start (no offline warmup)")

    return buffer, normalizer, start_training, data_root


# =====================================================================
# 9. Offline Data Loaders
# =====================================================================

def make_offline_loaders(data_root, cfg, train_task_names, batch_size,
                         p_suite, n_suite):
    """Create infinite iterators for P (expert) and N (sub-optimal) data.

    Returns ``(p_iter, n_iter)``.
    """
    def _make(suite):
        loader = make_loader(
            path=str(data_root), suite=suite,
            scenes=train_task_names,
            num_demos_per_task=cfg.data.num_demos_per_task,
            chunk_size=cfg.agent.action_chunking.num_queries,
            batch_size=batch_size,
            num_workers=cfg.train.num_workers,
            discount=cfg.train.rl.discount)
        return iter(loader)

    return _make(p_suite), _make(n_suite)


def make_single_loader(data_root, cfg, train_task_names, batch_size, suite):
    """Create a single infinite iterator for one dataset suite.

    Returns an iterator.
    """
    loader = make_loader(
        path=str(data_root), suite=suite,
        scenes=train_task_names,
        num_demos_per_task=cfg.data.num_demos_per_task,
        chunk_size=cfg.agent.action_chunking.num_queries,
        batch_size=batch_size,
        num_workers=cfg.train.num_workers,
        discount=cfg.train.rl.discount)
    return iter(loader)


# =====================================================================
# 10. Online-Offline Mixing
# =====================================================================

def mix_online_offline(on_data, bp_data, bn_data, off_p, off_n,
                       normalizer, g2l, alpha):
    """Mix online buffer samples with offline P+N data.

    Args:
        on_data:  ``(imgs, acts, rews_normalized, next_imgs, dones, gids)``
        bp_data:  ``(imgs, nexts, acts, rews, dones, gids)`` from :func:`load_batch_to_device`
        bn_data:  same format as *bp_data*
        off_p, off_n: number of P / N samples to take

    Returns:
        ``(pix_m, act_m, rew_m, nxt_m, don_m, gid_m)``
    """
    on_imgs, on_acts, on_rews_n, on_next, on_dones, on_gids = on_data
    bp_imgs, bp_next, bp_acts, bp_rews, bp_dones, bp_gids = bp_data
    bn_imgs, bn_next, bn_acts, bn_rews, bn_dones, bn_gids = bn_data

    of_imgs  = torch.cat([bp_imgs[:off_p],  bn_imgs[:off_n]])
    of_next  = torch.cat([bp_next[:off_p],  bn_next[:off_n]])
    of_acts  = torch.cat([bp_acts[:off_p],  bn_acts[:off_n]])
    of_rews  = torch.cat([bp_rews[:off_p],  bn_rews[:off_n]])
    of_dones = torch.cat([bp_dones[:off_p], bn_dones[:off_n]])
    of_gids  = torch.cat([bp_gids[:off_p],  bn_gids[:off_n]])
    of_rews_n = normalizer.normalize(of_rews, g2l[of_gids], alpha)

    return (torch.cat([on_imgs,   of_imgs]),
            torch.cat([on_acts,   of_acts]),
            torch.cat([on_rews_n, of_rews_n]),
            torch.cat([on_next,   of_next]),
            torch.cat([on_dones,  of_dones]),
            torch.cat([on_gids,   of_gids]))


# =====================================================================
# 11. Buffer Pre-fill  (from offline pkl data)
# =====================================================================

def prefill_buffer(buffer, data_root, train_task_names, full_task_map,
                   suite_p, suite_n, num_episodes):
    """Pre-fill replay buffer with complete episodes from P+N pkl datasets."""
    total_steps = 0
    for suite_name in [suite_p, suite_n]:
        for task_name in train_task_names:
            pkl_path = data_root / suite_name / f"{task_name}.pkl"
            if not pkl_path.exists():
                print(f"  [WARN] pkl not found, skip: {pkl_path}")
                continue
            with open(pkl_path, "rb") as f:
                data = pkl.load(f)

            task_id = full_task_map[task_name]
            obs_list = data["observations"]
            act_list = data["actions"]
            rew_list = data["rewards"]

            n_eps = min(num_episodes, len(obs_list))
            for ep_idx in range(n_eps):
                pixels  = obs_list[ep_idx]["pixels"]
                actions = act_list[ep_idx]
                rewards = rew_list[ep_idx]
                T = len(actions)
                for t in range(T):
                    obs_chw      = np.ascontiguousarray(pixels[t].transpose(2, 0, 1))
                    next_obs_chw = np.ascontiguousarray(pixels[t + 1].transpose(2, 0, 1))
                    terminated = (float(rewards[t]) >= 19.9)
                    buffer.add(obs_chw, actions[t], float(rewards[t]),
                               next_obs_chw, terminated, task_id)
                    total_steps += 1

    print(f"  Buffer pre-filled: {total_steps} steps  (buf.size={buffer.size})")


def load_full_task_map(cfg):
    """Load the full task palette mapping from ``conf/task_palette.yaml``."""
    with open(CONF_DIR / "task_palette.yaml", 'r') as f:
        return yaml.safe_load(f)


# =====================================================================
# 12. Main Training Loop
# =====================================================================

def _default_format_log(m, global_step, buf_size):
    """Default log line (used by BRC_BC and SO2)."""
    return (
        f"  [S={global_step:>7d}] "
        f"cri={m.get('critic_loss',0):.3f} "
        f"act={m.get('actor_loss',0):.3f} "
        f"bc_w={m.get('bc_weight',0):.3f} "
        f"Qd={m.get('q_data',0):+.1f} "
        f"Qp={m.get('q_pi',0):+.1f} "
        f"a={m.get('alpha',0):.1e} "
        f"buf={buf_size}")


def run_online_training(
    *,
    agent, agent_ref, cfg,
    train_task_names, g2l,
    buffer, normalizer,
    start_training, update_every, min_buf,
    checkpoint_prefix,
    update_fn,
    format_log_fn=None,
    log_interval=1000,
    eval_interval=5_000,
    save_interval=5_000,
    dist_capture_interval=5,
    dist_video_roll_freq=1,
    done_message="Online fine-tuning finished.",
):
    """Shared online training roll loop.

    Args:
        update_fn:      ``callable(global_step) -> metrics_dict``.
                        Called when gradient-update conditions are met.
                        Should sample data and call ``agent.update()`` internally.
        format_log_fn:  ``callable(metrics, global_step, buf_size) -> str``.
                        Defaults to :func:`_default_format_log`.
    """
    if format_log_fn is None:
        format_log_fn = _default_format_log

    work_dir = Path.cwd()
    device = torch.device(cfg.device)
    num_tasks = len(train_task_names)

    logger = Logger(work_dir, use_tb=cfg.train.use_tb)
    video_recorder = VideoRecorder(work_dir)

    num_exec  = cfg.agent.action_chunking.num_exec
    num_rolls = cfg.train.rl.num_rolls

    dist_video_dir = work_dir / 'dist_videos'
    dist_video_dir.mkdir(parents=True, exist_ok=True)
    support_np = agent.support.cpu().numpy()

    log_every  = log_interval  * num_tasks
    eval_every = eval_interval * num_tasks
    save_every = save_interval * num_tasks

    global_step = 0
    steps_since_update = 0
    last_metrics = None
    last_log_step = 0

    agent.train(True)

    for roll_idx in range(1, num_rolls + 1):

        envs = make(
            task_names=train_task_names,
            cameras=cfg.env.cameras, img_size=cfg.env.img_size,
            action_repeat=cfg.env.action_repeat,
            seed=cfg.seed + roll_idx,
            max_episode_steps=cfg.env.max_episode_steps)

        should_track = (roll_idx % dist_video_roll_freq == 0)
        if should_track:
            track_idxs = {roll_idx % num_tasks, (roll_idx + 15) % num_tasks}
            tracked_frames = {ti: [] for ti in track_idxs}
        else:
            track_idxs = set()

        roll_successes = []
        roll_transitions = 0
        env_indices = np.random.permutation(len(envs))

        for env_i, idx in enumerate(env_indices):
            env = envs[idx]
            task_name = train_task_names[int(idx)]
            local_tid = int(idx)
            is_tracked = (int(idx) in track_idxs)

            time_step = env.reset()
            obs     = time_step.observation['pixels']
            task_id = time_step.observation['task_id'][0]

            done = False
            ep_rewards = []
            success = False
            action_queue = []
            ep_step = 0

            while not done:
                if is_tracked and ep_step % dist_capture_interval == 0:
                    obs_img = obs.transpose(1, 2, 0)
                    pt, pr = capture_q_dists(
                        obs, task_id, agent, agent_ref, device)
                    frame = render_dual_q_frame(
                        support_np, pt, pr, ep_step, obs_img, task_name)
                    tracked_frames[int(idx)].append(frame)

                if len(action_queue) == 0:
                    with torch.no_grad():
                        chunk = agent.act(obs, task_id, eval_mode=False)
                        action_queue.extend(chunk[:min(num_exec, len(chunk))])
                action = action_queue.pop(0)

                time_step = env.step(action)

                next_obs   = time_step.observation['pixels']
                reward     = time_step.reward
                done       = time_step.last()
                terminated = time_step.observation.get('terminated', False)
                success    = success or terminated
                ep_rewards.append(reward)

                buffer.add(obs, action, reward, next_obs, terminated, task_id)
                obs = next_obs
                global_step += 1
                ep_step += 1
                roll_transitions += 1
                steps_since_update += 1

                if (steps_since_update >= update_every
                        and global_step >= start_training
                        and buffer.size >= min_buf):
                    steps_since_update = 0
                    last_metrics = update_fn(global_step)

            normalizer.update(local_tid, ep_rewards)
            roll_successes.append(float(success))

            if is_tracked and tracked_frames.get(int(idx)):
                tag = task_name.replace('-', '_')
                vpath = dist_video_dir / f'roll{roll_idx}_{tag}.mp4'
                imageio.mimsave(str(vpath), tracked_frames[int(idx)], fps=5)
                tracked_frames[int(idx)].clear()

        for env in envs:
            env.close()
        del envs

        sr = np.mean(roll_successes) if roll_successes else 0.0
        print(f"[Roll {roll_idx:>6d}] SR={sr:.1%} buf={buffer.size} step={global_step}")

        if last_metrics and global_step - last_log_step >= log_every:
            last_log_step = global_step
            print(format_log_fn(last_metrics, global_step, buffer.size))
            for k, v in last_metrics.items():
                logger.log(f'train/{k}', v, global_step)
            logger.log('train/roll_success_rate', sr, global_step)
            logger.dump(global_step, Ty='train')

        if global_step >= start_training and global_step % eval_every < roll_transitions + num_tasks:
            run_online_eval(
                agent, cfg, train_task_names, roll_idx, global_step,
                logger, video_recorder, num_exec)

        if global_step >= start_training and global_step % save_every < roll_transitions + num_tasks:
            save_online_checkpoint(agent, work_dir, global_step, checkpoint_prefix)

    logger.close()
    print(done_message)
