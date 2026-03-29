"""
Serial_Bro.py — State-based serial training with BroNet actor/critic components.

This script is the Bro-style state-input counterpart to the pixel pipelines:
  - observation: MetaWorld 39-dim global_state
  - actor:       BRCTanhPolicy
  - critic:      C51 DistributionalCritic
  - training:    roll-based online collection with optional offline mixing
"""

import os
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"

import sys
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
CONFIG_DIR = str(REPO_ROOT / "conf")

import copy
import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from omegaconf import DictConfig, read_write

from experiments.state.shared_config import *
from core.wrapper_new import make
from core.replay_buffer import FeatureReplayBuffer, ParallelFeatureReplayBuffer
from core.normalizer import RewardNormalizer
from core.networks import DistributionalCritic
from core.sac_c51_mixin import SACC51Mixin
from core.brc_components import (
    BRCTaskEmbedding, BRCActorBackbone, BRCTanhPolicy, BRCRewardNormalizer,
    StateEncoder, _orthogonal_init,
    TASK_EMB_DIM, ACTOR_WIDTH, ACTOR_DEPTH, LOG_STD_MIN, LOG_STD_MAX,
)
from utils.logger import Logger
from utils.common import set_seed_everywhere, VideoRecorder, Timer
from utils.offline_train_utils import setup_task_palette, load_state_offline_buffer


# =====================================================================
# 5. FaithfulBRCAgent
# =====================================================================

class FaithfulBRCAgent(SACC51Mixin):
    """Standalone BRC agent — faithful to the original JAX implementation.

    Does NOT inherit from RLAgent (too many architectural differences).
    Reuses DistributionalCritic (BroNet-based C51 ensemble) from core/networks.py.

    Key BRC-specific aspects:
      - BRCTanhPolicy actor (BroNet width=256, depth=1)
      - BRCTaskEmbedding (32-dim, L2-normalized)
      - Critic: DistributionalCritic(width=4096, depth=2)
      - C51 range: [-10, 10]
      - Critic loss: sum over ensemble (not mean)
      - Temperature optimizer: Adam(b1=0.5)
      - No action chunking, no encoder
    """

    def __init__(self, device, num_tasks):
        self.device = device
        self.num_tasks = num_tasks

        # C51 support
        self.num_bins = NUM_BINS
        self.v_min = V_MIN
        self.v_max = V_MAX
        self.support = torch.linspace(V_MIN, V_MAX, NUM_BINS).to(device)
        self.gamma = DISCOUNT
        self.tau = TAU

        # Target entropy: -act_dim / 2 (BRC default)
        self.target_entropy = -ACT_DIM / 2.0

        # --- Modules ---
        self.task_embedding = BRCTaskEmbedding(50, TASK_EMB_DIM).to(device)

        actor_input_dim = STATE_DIM + TASK_EMB_DIM  # 39 + 32 = 71
        self.actor = BRCTanhPolicy(
            input_dim=actor_input_dim,
            action_dim=ACT_DIM,
            hidden_dim=ACTOR_WIDTH,
            depth=ACTOR_DEPTH,
        ).to(device)

        self.critic = DistributionalCritic(
            repr_dim=STATE_DIM,
            action_dim=ACT_DIM,
            task_emb_dim=TASK_EMB_DIM,
            hidden_dim=CRITIC_WIDTH,
            depth=CRITIC_DEPTH,
            num_bins=NUM_BINS,
            ensemble_size=ENSEMBLE_SIZE,
        ).to(device)
        self.critic_target = copy.deepcopy(self.critic)

        self.log_alpha = torch.tensor(
            np.log(INIT_TEMPERATURE), device=device, dtype=torch.float32,
            requires_grad=True)

        # --- Optimizers (BRC-faithful) ---
        self.actor_opt = torch.optim.AdamW(
            self.actor.parameters(), lr=ACTOR_LR, weight_decay=WEIGHT_DECAY)
        # Critic opt includes task_embedding params (BRC embeds task_emb inside
        # critic; we keep it separate but train via critic_opt for equivalence)
        self.critic_opt = torch.optim.AdamW(
            list(self.critic.parameters()) +
            list(self.task_embedding.parameters()),
            lr=CRITIC_LR, weight_decay=WEIGHT_DECAY)
        self.alpha_opt = torch.optim.Adam(
            [self.log_alpha], lr=TEMP_LR, betas=(TEMP_ADAM_B1, 0.999))

        self.training = True

    # ------------------------------------------------------------------
    # Mode switching
    # ------------------------------------------------------------------

    def train(self, mode=True):
        self.training = mode
        self.actor.train(mode)
        self.critic.train(mode)
        self.task_embedding.train(mode)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def act(self, state, task_id, eval_mode=False):
        """Select a single action (no chunking).

        Args:
            state:    numpy array (39,) — raw MetaWorld state.
            task_id:  int — global task ID.
            eval_mode: if True, use deterministic mean; else sample.

        Returns:
            numpy array (4,) clipped to [-1, 1].
        """
        obs = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        tid = torch.tensor([task_id], device=self.device, dtype=torch.long)
        temb = self.task_embedding(tid)
        actor_input = torch.cat([obs, temb], dim=-1)

        if eval_mode:
            dist = self.actor(actor_input, temperature=1.0)
            action = dist.mean
        else:
            dist = self.actor(actor_input, temperature=1.0)
            action = dist.sample()

        return action.squeeze(0).cpu().numpy().clip(-1.0, 1.0)

    # ------------------------------------------------------------------
    # Update: C51 critic (BRC-faithful)
    # ------------------------------------------------------------------

    def _update_critic(self, obs, action, reward, next_obs, task_emb,
                       terminated=None, return_dists=False, dist_idx=0):
        """C51 distributional critic update.

        Bootstrap logic:
          - terminated (success): target_support = reward (no bootstrap)
          - truncated / mid-episode: target_support = reward + gamma * next_support
        """
        metrics = {}

        # BRC actor interface: cat([obs, temb]) -> policy
        def actor_fn(temb, obs_feat):
            actor_in = torch.cat([obs_feat, temb], dim=-1)
            return self.actor(actor_in, temperature=1.0)

        target_dist, q_mean = self._c51_project_target(
            next_obs, task_emb, reward, actor_fn)
        metrics['q_mean'] = q_mean

        # Critic forward
        current_logits = self.critic(obs, action, task_emb)  # (E, B, bins)
        current_log_probs = F.log_softmax(current_logits, dim=-1)

        # BRC loss reduction: -(target[None] * log_probs).sum(-1).mean(-1).sum(-1)
        # = sum over ensemble of (mean over batch of CE)
        critic_loss = -(target_dist.unsqueeze(0) * current_log_probs) \
            .sum(-1).mean(-1).sum(-1)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        metrics['critic_loss'] = critic_loss.item()

        # Capture distributions for visualization
        if return_dists:
            with torch.no_grad():
                cur_probs = F.softmax(current_logits.detach(), dim=-1).mean(0)
                metrics['_current_probs'] = cur_probs[dist_idx].cpu().numpy()
                metrics['_target_probs'] = target_dist[dist_idx].cpu().numpy()

        return metrics

    # ------------------------------------------------------------------
    # Update: Actor (pure SAC, BRC-faithful)
    # ------------------------------------------------------------------

    def _update_actor(self, obs, task_emb):
        """Pure SAC actor update, faithful to BRC.

        actor_loss = (alpha * log_prob - Q_value).mean()
        Q_value = E[Q] over distributional support, mean over ensemble.
        """
        metrics = {}

        # Task embedding is detached for actor update (BRC: actor_opt does not
        # include critic/task_emb params; equivalent to stop-gradient)
        task_emb_d = task_emb.detach()
        obs_d = obs.detach()
        actor_input = torch.cat([obs_d, task_emb_d], dim=-1)

        dist = self.actor(actor_input, temperature=1.0)
        new_action = dist.rsample()
        log_prob = dist.log_prob(new_action).sum(-1, keepdim=True)

        q_value = self._get_expected_q(obs_d, new_action, task_emb_d)
        actor_loss = (self.alpha.detach() * log_prob - q_value).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        metrics['actor_loss'] = actor_loss.item()
        metrics['entropy'] = -log_prob.mean().item()
        metrics['q_pi'] = q_value.mean().item()
        return metrics, -log_prob.mean().detach()

    # ------------------------------------------------------------------
    # Full update step
    # ------------------------------------------------------------------

    def update(self, batch, return_dists=False, dist_idx=0):
        """BRC-faithful update order (aligned with Parallel_Bro):
        1. Compute task_emb (pre-critic)
        2. Update critic
        3. Soft update target critic
        4. Compute FRESH task_emb (post-critic, for actor)
        5. Update actor (uses fresh task_emb)
        6. Update alpha
        """
        obs, action, reward, next_obs, task_ids, terminated = batch

        task_emb = self.task_embedding(task_ids)

        # 1. Critic update
        critic_m = self._update_critic(
            obs, action, reward, next_obs, task_emb,
            terminated=terminated,
            return_dists=return_dists, dist_idx=dist_idx)

        # 2. Soft update target
        self._soft_update()

        # 3. Fresh task_emb from UPDATED task_embedding
        task_emb_fresh = self.task_embedding(task_ids)

        # 4. Actor update (with fresh task_emb)
        actor_m, entropy = self._update_actor(obs, task_emb_fresh)

        # 5. Alpha update
        alpha_m = self._update_alpha(entropy)

        metrics = {**critic_m, **actor_m, **alpha_m}
        metrics['reward'] = reward.mean().item()
        return metrics

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save_snapshot(self):
        return {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'task_embedding': self.task_embedding.state_dict(),
            'log_alpha': self.log_alpha.detach().cpu(),
        }

    def load_snapshot(self, payload):
        self.actor.load_state_dict(payload['actor'])
        self.critic.load_state_dict(payload['critic'])
        self.critic_target.load_state_dict(payload['critic_target'])
        self.task_embedding.load_state_dict(payload['task_embedding'])
        self.log_alpha = payload['log_alpha'].to(self.device).requires_grad_(True)
        self.alpha_opt = torch.optim.Adam(
            [self.log_alpha], lr=TEMP_LR, betas=(TEMP_ADAM_B1, 0.999))


# =====================================================================
# Evaluation
# =====================================================================

def run_brc_eval(agent, cfg, train_task_names, step, global_step,
                 logger, video_recorder):
    """Evaluate BRC agent across all tasks using global_state."""
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
        success_count = 0

        video_recorder.init(enabled=True)

        for _ in range(cfg.train.rl.num_eval_episodes):
            time_step = env.reset()
            obs = time_step.observation['global_state']
            task_id = time_step.observation['task_id'][0]

            done = False
            is_success = False

            while not done:
                action = agent.act(obs, task_id, eval_mode=True)
                time_step = env.step(action)
                obs = time_step.observation['global_state']
                done = time_step.last()
                if time_step.observation['goal_achieved'] > 0.5:
                    is_success = True
                video_recorder.record(env)

            if is_success:
                success_count += 1

        video_recorder.save(f'{tn}_step{global_step}.mp4')
        s_rate = success_count / max(1, cfg.train.rl.num_eval_episodes)
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
# 9. Main Training Loop
# =====================================================================

@hydra.main(config_path=CONFIG_DIR, config_name="config")
def train(cfg: DictConfig):
    work_dir = Path.cwd()
    device = torch.device(cfg.device)
    set_seed_everywhere(cfg.seed)

    # --- Task palette ---
    train_task_names, global_to_local, _, g2l = \
        setup_task_palette(cfg, device)
    num_tasks = len(train_task_names)

    # =================================================================
    # Agent (learn from scratch)
    # =================================================================
    agent = FaithfulBRCAgent(device, num_tasks)

    # =================================================================
    # Buffer (online) — per-task parallel structure matching BRC
    # =================================================================
    global_task_ids = [None] * num_tasks
    for gid, lid in global_to_local.items():
        global_task_ids[lid] = gid

    buffer = ParallelFeatureReplayBuffer(
        num_tasks=num_tasks,
        capacity_per_task=CAPACITY_PER_TASK,
        repr_dim=STATE_DIM,
        act_dim=ACT_DIM,
        device=device,
        global_task_ids=global_task_ids)

    START_TRAINING = WARMUP_PER_TASK * num_tasks

    # =================================================================
    # Offline buffer (P + N mixed)
    # =================================================================
    off_bs = int(BATCH_SIZE * OFF_RATIO)
    on_bs = BATCH_SIZE - off_bs

    data_root = REPO_ROOT / cfg.data.root_dir.lstrip('./')

    # =================================================================
    # Normalizer (BRC-faithful bootstrap) — warm up with P + N data
    # =================================================================
    normalizer = BRCRewardNormalizer(
        num_tasks=num_tasks,
        action_dim=ACT_DIM,
        discount=DISCOUNT,
        v_max=V_MAX)

    if off_bs > 0:
        offline_buffer, all_reward_seqs = load_state_offline_buffer(
            data_root, P_SUITE, N_SUITE, train_task_names, device)
        print("Pre-updating Reward Normalizer with P + N datasets...")
        for gid, seqs in all_reward_seqs.items():
            lid = g2l[gid]
            for r_seq in seqs:
                normalizer.update(lid, r_seq)
        print(f"  returns_max = {normalizer.returns_max}")
        print("Reward Normalizer ready.\n")
    else:
        offline_buffer = None
        print("OFF_RATIO=0, skipping offline buffer loading.\n")

    # =================================================================
    # Logger & tools
    # =================================================================
    logger = Logger(work_dir, use_tb=cfg.train.use_tb)
    video_recorder = VideoRecorder(work_dir)

    num_rolls = cfg.train.rl.num_rolls
    min_buf = on_bs + 1
    update_every = max(1, num_tasks // UTD)  # gradient update every N env steps

    log_every  = LOG_INTERVAL  * num_tasks
    eval_every = EVAL_INTERVAL * num_tasks
    save_every = SAVE_INTERVAL * num_tasks

    global_step = 0
    steps_since_update = 0
    last_metrics = None
    last_log_step = 0

    # =================================================================
    # Print config
    # =================================================================
    print(f"=== Serial_Bro (BRC reproduction) | {work_dir} ===")
    print(f"  state_dim={STATE_DIM} batch={BATCH_SIZE}(on={on_bs}+off={off_bs}) UTD={UTD} "
          f"discount={DISCOUNT} V=[{V_MIN},{V_MAX}]")
    print(f"  actor={ACTOR_WIDTH}x{ACTOR_DEPTH} critic={CRITIC_WIDTH}x{CRITIC_DEPTH} "
          f"task_emb={TASK_EMB_DIM} ensemble={ENSEMBLE_SIZE}")
    print(f"  lr_actor={ACTOR_LR} lr_critic={CRITIC_LR} lr_temp={TEMP_LR} "
          f"temp_b1={TEMP_ADAM_B1} init_temp={INIT_TEMPERATURE}")
    print(f"  target_entropy={agent.target_entropy:.2f} "
          f"start_training={START_TRAINING}")
    print(f"  P suite={P_SUITE}  N suite={N_SUITE}  "
          f"OFF_RATIO={OFF_RATIO}  offline_buf={offline_buffer.size if offline_buffer is not None else 0}")

    agent.train(True)

    # =================================================================
    # Roll loop
    # =================================================================
    for roll_idx in range(1, num_rolls + 1):

        # --- Create envs ---
        envs = make(
            task_names=train_task_names,
            cameras=cfg.env.cameras, img_size=cfg.env.img_size,
            action_repeat=cfg.env.action_repeat,
            seed=cfg.seed + roll_idx,
            max_episode_steps=cfg.env.max_episode_steps)

        roll_successes = []
        roll_transitions = 0
        env_indices = np.random.permutation(len(envs))

        # --- Data collection (no gradient updates) ---
        for env_i, idx in enumerate(env_indices):
            env = envs[idx]
            task_name = train_task_names[int(idx)]
            local_tid = int(idx)

            time_step = env.reset()
            obs = time_step.observation['global_state']
            task_id = time_step.observation['task_id'][0]

            done = False
            ep_reward = 0.0
            ep_rewards = []
            success = False
            ep_step = 0

            while not done:
                if global_step < START_TRAINING:
                    action = np.random.uniform(-1, 1, size=(ACT_DIM,)).astype(np.float32)
                else:
                    action = agent.act(obs, task_id, eval_mode=False)

                time_step = env.step(action)
                next_obs = time_step.observation['global_state']
                reward = time_step.reward
                done = time_step.last()
                terminated = time_step.observation.get('terminated', False)
                success = success or terminated
                ep_reward += reward
                ep_rewards.append(reward)

                buffer.add(obs, action, reward, next_obs, terminated, task_id)
                obs = next_obs
                global_step += 1
                ep_step += 1
                roll_transitions += 1
                steps_since_update += 1

                # --- Step-level gradient update ---
                if (steps_since_update >= update_every
                        and global_step >= START_TRAINING
                        and buffer.size >= min_buf):
                    steps_since_update = 0

                    on_batch = buffer.sample_chunk(on_bs, 1, DISCOUNT)
                    on_obs, on_acts, on_rews, on_next, on_dones, on_gids = on_batch
                    on_rews_n = normalizer.normalize(
                        on_rews, g2l[on_gids], agent.alpha)

                    if off_bs > 0 and offline_buffer is not None and offline_buffer.size > 0:
                        ex_batch = offline_buffer.sample_chunk(off_bs, 1, DISCOUNT)
                        ex_obs, ex_acts, ex_rews, ex_next, ex_dones, ex_gids = ex_batch
                        ex_rews_n = normalizer.normalize(
                            ex_rews, g2l[ex_gids], agent.alpha)

                        b_obs   = torch.cat([on_obs, ex_obs])
                        b_acts  = torch.cat([on_acts, ex_acts])
                        b_rews  = torch.cat([on_rews_n, ex_rews_n])
                        b_next  = torch.cat([on_next, ex_next])
                        b_gids  = torch.cat([on_gids, ex_gids])
                        b_dones = torch.cat([on_dones, ex_dones])
                    else:
                        b_obs   = on_obs
                        b_acts  = on_acts
                        b_rews  = on_rews_n
                        b_next  = on_next
                        b_gids  = on_gids
                        b_dones = on_dones

                    last_metrics = agent.update(
                        batch=(b_obs, b_acts, b_rews, b_next, b_gids, b_dones))

            normalizer.update(local_tid, ep_rewards)
            roll_successes.append(float(success))

        # --- Close envs ---
        for env in envs:
            env.close()
        del envs

        # --- Post-roll logging ---
        sr = np.mean(roll_successes) if roll_successes else 0.0
        print(f"[Roll {roll_idx:>6d}] SR={sr:.1%} buf={buffer.size} step={global_step}")

        # --- Step-based logging / eval / save ---
        if last_metrics and global_step - last_log_step >= log_every:
            last_log_step = global_step
            m = last_metrics
            print(
                f"  [S={global_step:>7d}] "
                f"cri={m.get('critic_loss',0):.3f} "
                f"act={m.get('actor_loss',0):.3f} "
                f"Qpi={m.get('q_pi',0):+.1f} "
                f"a={m.get('alpha',0):.1e} "
                f"ent={m.get('entropy',0):.2f} "
                f"buf={buffer.size}")
            for k, v in m.items():
                if not k.startswith('_'):
                    logger.log(f'train/{k}', v, global_step)
            logger.log('train/roll_success_rate', sr, global_step)
            logger.dump(global_step, Ty='train')

        if global_step >= START_TRAINING and global_step % eval_every < roll_transitions + num_tasks:
            run_brc_eval(agent, cfg, train_task_names, roll_idx,
                         global_step, logger, video_recorder)

        if global_step >= START_TRAINING and global_step % save_every < roll_transitions + num_tasks:
            snap = agent.save_snapshot()
            torch.save(snap, work_dir / f'snapshot_brc_{global_step}.pt')
            torch.save(snap, work_dir / 'snapshot_latest.pt')

    logger.close()
    print("Serial_Bro training finished.")


if __name__ == '__main__':
    train()
