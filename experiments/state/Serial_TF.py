"""
Serial_TF.py — State-based serial training with GPT actor + StateEncoder.

This script learns from 39-dim MetaWorld states instead of pixels:
  - encoder: StateEncoder
  - actor:   GPT-based Actor
  - critic:  C51 DistributionalCritic
  - training: roll-based online collection, optional offline batch mixing,
              and offline reward-sequence warmup for the normalizer
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
from core.agent import RLAgent, Actor
from core.replay_buffer import FeatureReplayBuffer, ParallelFeatureReplayBuffer
from core.brc_components import BRCRewardNormalizer, StateEncoder
from core.networks import TaskEmbedding, DistributionalCritic
from utils.logger import Logger
from utils.common import set_seed_everywhere, VideoRecorder, Timer
from utils.offline_train_utils import setup_task_palette, load_state_offline_buffer
from utils.online_train_utils import save_online_checkpoint


# =====================================================================
# StateRLAgent (subclasses RLAgent, replaces encoder & act)
# =====================================================================

class StateRLAgent(RLAgent):
    """RLAgent variant that uses low-dim state input via MLP encoder.

    Overrides:
        _init_modules: creates StateEncoder instead of VisionEncoder.
        act:           removes /255 pixel normalisation.
    """

    def __init__(self, device, cfg, state_dim=STATE_DIM):
        self._state_dim = state_dim
        super().__init__(device, cfg)

    def _init_modules(self, cfg):
        """Create all network modules (StateEncoder replaces VisionEncoder)."""
        self.task_embedding = TaskEmbedding(
            num_tasks=50, embed_dim=cfg.agent.repr_dim
        ).to(self.device)

        # MLP encoder instead of ResNet vision encoder
        self.encoder = StateEncoder(
            state_dim=self._state_dim,
            output_size=cfg.agent.repr_dim,
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
        self.target_entropy = -self.flat_act_dim / 2 if te is None else te

    @torch.no_grad()
    def act(self, obs, task_id, eval_mode=True):
        """Action selection from raw state vector (no /255 normalisation)."""
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float().to(self.device)
        obs = obs.unsqueeze(0)          # (1, state_dim)
        task_ids = torch.tensor([task_id], device=self.device).long()

        feat = self.encoder(obs)
        task_emb = self.task_embedding(task_ids)
        dist = self.actor(task_emb, feat)

        flat_action = dist.mean if eval_mode else dist.sample()
        return flat_action.view(self.act_chunk_size, self.raw_act_dim).cpu().numpy()

    def update(self, batch, bc_alpha=2.5, return_dists=False,
               dist_sample_idx=0, use_bc_loss=False, bc_batch=None):
        """Update order aligned with Parallel_TF:
        critic → soft_update → actor (fresh encoder+task_emb) → alpha.
        """
        obs, action, reward, next_obs, done, task_ids = batch
        metrics = {}
        metrics['reward'] = reward.mean().item()

        # Encode (with grad for critic)
        obs_feat = self.encoder(obs)
        task_emb = self.task_embedding(task_ids)
        with torch.no_grad():
            next_obs_feat = self.encoder(next_obs)

        # 1. Critic
        critic_m = self._update_critic(
            obs_feat, action, reward, next_obs_feat, done, task_emb,
            return_dists=return_dists, dist_sample_idx=dist_sample_idx)
        metrics.update(critic_m)

        # 2. Soft update target
        self._soft_update(self.critic, self.critic_target, self.tau)

        # 3. Actor (fresh encoder + task_emb, detached)
        with torch.no_grad():
            obs_feat_fresh = self.encoder(obs)
            task_emb_fresh = self.task_embedding(task_ids)

        dist = self.actor(task_emb_fresh, obs_feat_fresh)
        new_action = dist.rsample()
        new_log_prob = dist.log_prob(new_action).sum(-1, keepdim=True)

        q_value = self._get_expected_q(obs_feat_fresh, new_action, task_emb_fresh)
        actor_loss = (self.alpha.detach() * new_log_prob - q_value).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        metrics['actor_loss'] = actor_loss.item()
        metrics['sac_loss'] = actor_loss.item()
        metrics['bc_loss'] = 0.0
        metrics['lmbda'] = 0.0
        metrics['q_pi'] = q_value.mean().item()

        # 4. Alpha
        alpha_loss = (self.alpha * (-new_log_prob - self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        metrics['alpha_loss'] = alpha_loss.item()
        metrics['alpha'] = self.alpha.item()

        return metrics


# =====================================================================
# Evaluation
# =====================================================================

def run_state_eval(agent, cfg, train_task_names, step, global_step,
                   logger, video_recorder, num_exec):
    """Evaluate state-based agent across all tasks.

    Identical to run_online_eval except uses obs['global_state'] for action
    selection instead of obs['pixels'].
    """
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
        ep_rewards = []
        success_count = 0

        video_recorder.init(enabled=True)

        for _ in range(cfg.train.rl.num_eval_episodes):
            time_step = env.reset()
            obs = time_step.observation['global_state']
            task_id = time_step.observation['task_id'][0]

            done = False
            ep_reward = 0.0
            is_success = False
            action_queue = []

            while not done:
                if len(action_queue) == 0:
                    with torch.no_grad():
                        chunk = agent.act(obs, task_id, eval_mode=True)
                        action_queue.extend(chunk[:min(num_exec, len(chunk))])

                action = action_queue.pop(0)
                time_step = env.step(action)

                obs = time_step.observation['global_state']
                reward = time_step.reward
                done = time_step.last()
                if time_step.observation['goal_achieved'] > 0.5:
                    is_success = True
                ep_reward += reward
                video_recorder.record(env)

            ep_rewards.append(ep_reward)
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
# Main
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
    # Agent (learn from scratch, no snapshot)
    # =================================================================
    with read_write(cfg):
        cfg.train.rl.encoder_freeze = False

    agent = StateRLAgent(device, cfg, state_dim=STATE_DIM)
    agent.use_min_q = False

    # =================================================================
    # Online buffer — per-task parallel structure matching BRC
    # =================================================================
    global_task_ids = [None] * num_tasks
    for gid, lid in global_to_local.items():
        global_task_ids[lid] = gid

    buffer = ParallelFeatureReplayBuffer(
        num_tasks=num_tasks,
        capacity_per_task=CAPACITY_PER_TASK,
        repr_dim=STATE_DIM,
        act_dim=4,
        device=device,
        global_task_ids=global_task_ids)

    START_TRAINING = WARMUP_PER_TASK * num_tasks

    # =================================================================
    # Offline buffer (P + N mixed)
    # =================================================================
    off_bs = int(BATCH_SIZE * OFF_RATIO)
    on_bs  = BATCH_SIZE - off_bs

    data_root = REPO_ROOT / cfg.data.root_dir.lstrip('./')
    offline_buffer, all_reward_seqs = load_state_offline_buffer(
        data_root, P_SUITE, N_SUITE, train_task_names, device)

    # =================================================================
    # Normalizer — warm up with P + N data
    # =================================================================
    normalizer = BRCRewardNormalizer(
        num_tasks=len(train_task_names),
        action_dim=agent.flat_act_dim,
        discount=cfg.train.rl.discount,
        v_max=cfg.train.rl.critic.v_max,
    )

    print("Pre-updating Reward Normalizer with P + N datasets...")
    for gid, seqs in all_reward_seqs.items():
        lid = g2l[gid]
        for r_seq in seqs:
            # BRC original: always bootstrap (no success distinction)
            normalizer.update(lid, r_seq)
    print(f"  returns_max = {normalizer.returns_max}")
    print("Reward Normalizer ready.\n")

    # =================================================================
    # Logger & tools
    # =================================================================
    logger = Logger(work_dir, use_tb=cfg.train.use_tb)
    video_recorder = VideoRecorder(work_dir)
    timer = Timer()

    chunk_size = cfg.agent.action_chunking.num_queries
    num_exec   = cfg.agent.action_chunking.num_exec
    gamma      = cfg.train.rl.discount
    num_rolls  = cfg.train.rl.num_rolls
    min_buf    = on_bs + 1
    update_every = max(1, num_tasks // UTD)

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
    print(f"=== Serial_TF (from scratch + offline mix) | {work_dir} ===")
    print(f"  state_dim={STATE_DIM} batch={BATCH_SIZE}(on={on_bs}+off={off_bs}) UTD={UTD} "
          f"chunk={chunk_size} gamma={gamma} V=[{agent.v_min},{agent.v_max}] "
          f"use_min_q=False")
    print(f"  P suite={P_SUITE}  N suite={N_SUITE}  "
          f"OFF_RATIO={OFF_RATIO}  offline_buf={offline_buffer.size}")
    print(f"  target_entropy={agent.target_entropy:.2f} "
          f"encoder={'unfrozen' if not agent.freeze_encoder else 'frozen'}")

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
            obs     = time_step.observation['global_state']
            task_id = time_step.observation['task_id'][0]

            done = False
            ep_reward = 0.0
            ep_rewards = []
            success = False
            action_queue = []
            ep_step = 0

            while not done:
                if global_step < START_TRAINING:
                    action = np.random.uniform(-1, 1, size=(4,)).astype(np.float32)
                else:
                    if len(action_queue) == 0:
                        with torch.no_grad():
                            chunk = agent.act(obs, task_id, eval_mode=False)
                            action_queue.extend(chunk[:min(num_exec, len(chunk))])
                    action = action_queue.pop(0)

                time_step = env.step(action)

                next_obs = time_step.observation['global_state']
                reward   = time_step.reward
                done     = time_step.last()
                terminated = time_step.observation.get('terminated', False)
                success  = success or terminated
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

                    on_batch = buffer.sample_chunk(on_bs, chunk_size, gamma)
                    on_s, on_a, on_r, on_ns, on_d, on_g = on_batch
                    on_r_n = normalizer.normalize(
                        on_r, g2l[on_g], agent.alpha)

                    if off_bs > 0 and offline_buffer is not None and offline_buffer.size > 0:
                        ex_batch = offline_buffer.sample_chunk(off_bs, chunk_size, gamma)
                        ex_s, ex_a, ex_r, ex_ns, ex_d, ex_g = ex_batch
                        ex_r_n = normalizer.normalize(
                            ex_r, g2l[ex_g], agent.alpha)

                        b_states = torch.cat([on_s, ex_s])
                        b_acts   = torch.cat([on_a, ex_a])
                        b_rews_n = torch.cat([on_r_n, ex_r_n])
                        b_next   = torch.cat([on_ns, ex_ns])
                        b_dones  = torch.cat([on_d, ex_d])
                        b_gids   = torch.cat([on_g, ex_g])
                    else:
                        b_states = on_s
                        b_acts   = on_a
                        b_rews_n = on_r_n
                        b_next   = on_ns
                        b_dones  = on_d
                        b_gids   = on_g

                    last_metrics = agent.update(
                        batch=(b_states, b_acts, b_rews_n,
                               b_next, b_dones, b_gids),
                        use_bc_loss=False)

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
                f"Qd={m.get('q_data',0):+.1f} "
                f"Qp={m.get('q_pi',0):+.1f} "
                f"a={m.get('alpha',0):.1e} "
                f"buf={buffer.size}")
            for k, v in m.items():
                logger.log(f'train/{k}', v, global_step)
            logger.log('train/roll_success_rate', sr, global_step)
            logger.dump(global_step, Ty='train')

        if global_step >= START_TRAINING and global_step % eval_every < roll_transitions + num_tasks:
            run_state_eval(
                agent, cfg, train_task_names, roll_idx, global_step,
                logger, video_recorder, num_exec)

        if global_step >= START_TRAINING and global_step % save_every < roll_transitions + num_tasks:
            save_online_checkpoint(agent, work_dir, global_step, 'snapshot_serial_tf')

    logger.close()
    print("Serial_TF training finished.")


if __name__ == '__main__':
    train()
