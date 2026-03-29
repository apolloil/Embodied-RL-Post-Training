"""
Parallel_TF.py — Parallel lockstep training with GPT Actor + StateEncoder.

Combines the Parallel_Bro training framework (parallel env, BRCReplayBuffer,
mask=always 1, normalizer always bootstrap) with the GPT-based Actor and
StateEncoder architecture from Serial_TF.

Key design:
  - Training loop: identical to Parallel_BRO (parallel lockstep, unified 200 steps)
  - Actor: GPT backbone + StochasticActorHead (from core.agent.Actor)
  - Encoder: StateEncoder MLP (39 → 512)
  - TaskEmbedding: 512-dim L2-normalized (from core.networks.TaskEmbedding)
  - Critic: DistributionalCritic with repr_dim=512 (encoded features)
  - Mask: always 1.0 (BRC MetaWorld convention)
  - Actor task_emb: uses FRESH task_emb from updated critic
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

from core.wrapper_parallel import ParallelMetaWorldEnv
from core.replay_buffer import BRCReplayBuffer, FeatureReplayBuffer
from utils.logger import Logger
from utils.common import set_seed_everywhere
from utils.offline_train_utils import setup_task_palette, load_state_offline_buffer

from experiments.state.shared_config import *
from core.agent import Actor
from core.networks import TaskEmbedding, DistributionalCritic
from core.brc_components import BRCRewardNormalizer, StateEncoder
from core.sac_c51_mixin import SACC51Mixin

# TF-specific: encoder output / GPT input / TaskEmbedding dim
REPR_DIM = 512


# =====================================================================
# ParallelTFAgent — GPT Actor + StateEncoder in Parallel_Bro framework
# =====================================================================

class ParallelTFAgent(SACC51Mixin):
    """Parallel lockstep agent with GPT Actor + StateEncoder.

    Architecture:
      - StateEncoder: MLP (39 → 512)
      - TaskEmbedding: 512-dim, L2-normalized
      - Actor: GPT backbone + StochasticActorHead
      - Critic: DistributionalCritic (BroNet, width=4096, depth=2)

    Training:
      - Mask always 1.0 (BRC convention)
      - Actor uses fresh task_emb from updated critic
      - Encoder params in critic optimizer (unfrozen)
    """

    def __init__(self, device, num_tasks, cfg):
        self.device = device
        self.num_tasks = num_tasks

        self.num_bins = NUM_BINS
        self.v_min = V_MIN
        self.v_max = V_MAX
        self.support = torch.linspace(V_MIN, V_MAX, NUM_BINS).to(device)
        self.gamma = DISCOUNT
        self.tau = TAU

        self.target_entropy = -ACT_DIM / 2.0

        # --- Modules ---
        self.encoder = StateEncoder(
            state_dim=STATE_DIM,
            output_size=REPR_DIM,
        ).to(device)

        self.task_embedding = TaskEmbedding(
            num_tasks=50, embed_dim=REPR_DIM
        ).to(device)

        self.actor = Actor(
            repr_dim=REPR_DIM,
            act_dim=ACT_DIM,
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
            gpt_dropout=cfg.agent.gpt.dropout,
        ).to(device)

        self.flat_act_dim = ACT_DIM * cfg.agent.action_chunking.num_queries

        self.critic = DistributionalCritic(
            repr_dim=REPR_DIM,
            action_dim=self.flat_act_dim,
            task_emb_dim=REPR_DIM,
            hidden_dim=CRITIC_WIDTH,
            depth=CRITIC_DEPTH,
            num_bins=NUM_BINS,
            ensemble_size=ENSEMBLE_SIZE,
        ).to(device)
        self.critic_target = copy.deepcopy(self.critic)

        self.log_alpha = torch.tensor(
            np.log(INIT_TEMPERATURE), device=device, dtype=torch.float32,
            requires_grad=True)

        # --- Optimizers ---
        self.actor_opt = torch.optim.AdamW(
            self.actor.parameters(), lr=ACTOR_LR, weight_decay=WEIGHT_DECAY)

        # Critic optimizer includes encoder + task_embedding (unfrozen)
        self.critic_opt = torch.optim.AdamW(
            list(self.critic.parameters()) +
            list(self.encoder.parameters()) +
            list(self.task_embedding.parameters()),
            lr=CRITIC_LR, weight_decay=WEIGHT_DECAY)

        self.alpha_opt = torch.optim.Adam(
            [self.log_alpha], lr=TEMP_LR, betas=(TEMP_ADAM_B1, 0.999))

        self.training = True

    def train(self, mode=True):
        self.training = mode
        self.encoder.train(mode)
        self.actor.train(mode)
        self.critic.train(mode)
        self.task_embedding.train(mode)

    # ------------------------------------------------------------------
    # Critic update (mask always 1.0)
    # ------------------------------------------------------------------

    def _update_critic(self, obs, action, reward, next_obs, task_ids):
        """C51 critic update. Encodes obs through StateEncoder."""
        metrics = {}

        # Encode observations (encoder is in critic_opt, gradients flow)
        obs_feat = self.encoder(obs)
        task_emb = self.task_embedding(task_ids)

        # GPT actor interface: actor(task_emb, obs_feat)
        def actor_fn(temb, obs_f):
            return self.actor(temb, obs_f)

        with torch.no_grad():
            next_obs_feat = self.encoder(next_obs)

        target_dist, q_mean = self._c51_project_target(
            next_obs_feat, task_emb, reward, actor_fn)
        metrics['q_mean'] = q_mean

        current_logits = self.critic(obs_feat, action, task_emb)
        current_log_probs = F.log_softmax(current_logits, dim=-1)

        critic_loss = -(target_dist.unsqueeze(0) * current_log_probs) \
            .sum(-1).mean(-1).sum(-1)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        metrics['critic_loss'] = critic_loss.item()
        return metrics

    # ------------------------------------------------------------------
    # Actor update (uses FRESH task_emb + encoded obs)
    # ------------------------------------------------------------------

    def _update_actor(self, obs, task_ids):
        """Actor update using fresh task_emb from updated critic."""
        metrics = {}

        with torch.no_grad():
            obs_feat = self.encoder(obs)
            task_emb = self.task_embedding(task_ids)

        dist = self.actor(task_emb, obs_feat)
        new_action = dist.rsample()
        log_prob = dist.log_prob(new_action).sum(-1, keepdim=True)

        q_value = self._get_expected_q(obs_feat, new_action, task_emb)
        actor_loss = (self.alpha.detach() * log_prob - q_value).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        metrics['actor_loss'] = actor_loss.item()
        metrics['entropy'] = -log_prob.mean().item()
        metrics['q_pi'] = q_value.mean().item()
        return metrics, -log_prob.mean().detach()

    # ------------------------------------------------------------------
    # Full update
    # ------------------------------------------------------------------

    def update(self, batch):
        """Update order: critic → soft_update → actor (fresh emb) → alpha."""
        obs, action, reward, next_obs, task_ids, terminated = batch

        # 1. Critic update (encoder + task_emb gradients flow here)
        critic_m = self._update_critic(obs, action, reward, next_obs, task_ids)

        # 2. Soft update target
        self._soft_update()

        # 3. Actor update (with fresh encoder + task_emb, detached)
        actor_m, entropy = self._update_actor(obs, task_ids)

        # 4. Alpha update
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
            'encoder': self.encoder.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'task_embedding': self.task_embedding.state_dict(),
            'log_alpha': self.log_alpha.detach().cpu(),
        }

    def load_snapshot(self, payload):
        self.actor.load_state_dict(payload['actor'])
        self.encoder.load_state_dict(payload['encoder'])
        self.critic.load_state_dict(payload['critic'])
        self.critic_target.load_state_dict(payload['critic_target'])
        self.task_embedding.load_state_dict(payload['task_embedding'])
        self.log_alpha = payload['log_alpha'].to(self.device).requires_grad_(True)
        self.alpha_opt = torch.optim.Adam(
            [self.log_alpha], lr=TEMP_LR, betas=(TEMP_ADAM_B1, 0.999))

    @torch.no_grad()
    def get_eval_actions(self, obs_np, task_ids_np, device):
        """Vectorized deterministic action selection for parallel evaluation."""
        obs_t = torch.from_numpy(obs_np).float().to(device)
        tid_t = torch.from_numpy(task_ids_np).long().to(device)
        obs_feat = self.encoder(obs_t)
        task_emb = self.task_embedding(tid_t)
        dist = self.actor(task_emb, obs_feat)
        return dist.mean.cpu().numpy().clip(-1.0, 1.0)


# =====================================================================
# Main Training Loop
# =====================================================================

@hydra.main(config_path=CONFIG_DIR, config_name="config")
def train(cfg: DictConfig):
    work_dir = Path.cwd()
    device = torch.device(cfg.device)
    set_seed_everywhere(cfg.seed)

    with read_write(cfg):
        cfg.train.rl.discount = DISCOUNT
        cfg.env.action_repeat = 1
        cfg.agent.action_chunking.num_queries = 1
        cfg.agent.action_chunking.num_exec = 1

    # --- Task palette ---
    train_task_names, global_to_local, _, g2l = \
        setup_task_palette(cfg, device)
    num_tasks = len(train_task_names)

    task_palette_path = REPO_ROOT / "conf" / "task_palette.yaml"
    import yaml
    with open(task_palette_path, 'r') as f:
        task_palette = yaml.safe_load(f)

    # --- Agent ---
    agent = ParallelTFAgent(device, num_tasks, cfg)
    agent.train(True)

    # --- Batch sizes ---
    off_bs = int(BATCH_SIZE * OFF_RATIO)
    on_bs  = BATCH_SIZE - off_bs

    # --- Offline data (optional) ---
    offline_buffer = None
    all_reward_seqs = {}
    data_root = REPO_ROOT / cfg.data.root_dir.lstrip('./')
    if off_bs > 0:
        offline_buffer, all_reward_seqs = load_state_offline_buffer(
            data_root, P_SUITE, N_SUITE, train_task_names, device,
            task_palette=task_palette)


    # --- Online replay buffer ---
    buffer = BRCReplayBuffer(
        num_tasks=num_tasks,
        capacity=CAPACITY_PER_TASK,
        state_dim=STATE_DIM,
        act_dim=ACT_DIM)

    # --- Normalizer ---
    normalizer = BRCRewardNormalizer(
        num_tasks=num_tasks,
        action_dim=ACT_DIM,
        discount=DISCOUNT,
        v_max=V_MAX)

    if all_reward_seqs:
        print("Pre-updating Reward Normalizer with P + N datasets...")
        for gid, seqs in all_reward_seqs.items():
            lid = g2l[gid]
            for r_seq in seqs:
                normalizer.update(lid, r_seq)
        print(f"  returns_max = {normalizer.returns_max}")

    normalizer.init_parallel(target_entropy=agent.target_entropy)
    print("Reward Normalizer ready.\n")

    # --- Parallel environment ---
    unified_max_steps = {name: MAX_EP_STEPS for name in train_task_names}
    env = ParallelMetaWorldEnv(
        task_names=train_task_names,
        seed=cfg.seed,
        max_episode_steps=unified_max_steps,
        action_repeat=cfg.env.action_repeat)

    # --- Logger ---
    logger = Logger(work_dir, use_tb=cfg.train.use_tb)

    # --- Print config ---
    print(f"=== Parallel_TF | {work_dir} ===")
    print(f"  state_dim={STATE_DIM} repr_dim={REPR_DIM} "
          f"batch={BATCH_SIZE}(on={on_bs}+off={off_bs}) "
          f"UTD={UTD} discount={DISCOUNT} V=[{V_MIN},{V_MAX}]")
    print(f"  weight_decay={WEIGHT_DECAY} tau={TAU} init_temp={INIT_TEMPERATURE}")
    print(f"  max_ep_steps={MAX_EP_STEPS} (unified) action_repeat={cfg.env.action_repeat}")
    print(f"  mask=always_1 | actor_task_emb=fresh (post-critic)")
    print(f"  actor=GPT encoder=StateEncoder({STATE_DIM}→{REPR_DIM}) "
          f"critic={CRITIC_WIDTH}x{CRITIC_DEPTH} "
          f"task_emb={REPR_DIM} ensemble={ENSEMBLE_SIZE}")
    print(f"  target_entropy={agent.target_entropy:.2f} "
          f"start_training={WARMUP_PER_TASK} max_steps={MAX_STEPS}")
    if off_bs > 0:
        print(f"  OFF_RATIO={OFF_RATIO} P={P_SUITE} N={N_SUITE} "
              f"offline_buf={offline_buffer.size}")
    else:
        print(f"  Pure online (OFF_RATIO=0)")

    # --- Global task IDs ---
    global_task_ids_np = env.global_task_ids

    # =================================================================
    # Training loop
    # =================================================================
    observations = env.reset()
    last_metrics = None

    for step in range(1, MAX_STEPS + 1):

        # --- Action selection ---
        if step < WARMUP_PER_TASK:
            actions = np.random.uniform(-1, 1,
                size=(num_tasks, ACT_DIM)).astype(np.float32)
        else:
            with torch.no_grad():
                obs_t = torch.from_numpy(observations).float().to(device)
                tid_t = torch.from_numpy(global_task_ids_np).long().to(device)
                obs_feat = agent.encoder(obs_t)
                task_emb = agent.task_embedding(tid_t)
                dist = agent.actor(task_emb, obs_feat)
                actions = dist.sample().cpu().numpy().clip(-1.0, 1.0)

        # --- Environment step ---
        next_obs, rewards, terminated, truncated, _ = env.step(actions)

        # --- Normalizer update ---
        normalizer.update_parallel(rewards, truncated)

        # --- Insert into replay buffer ---
        masks = env.generate_masks(terminated, truncated)
        buffer.insert(observations, actions, rewards, masks, next_obs)

        # --- Auto-reset done envs ---
        observations = env.reset_where_done(next_obs, terminated, truncated)

        # --- Gradient updates ---
        if step >= WARMUP_PER_TASK and buffer.size >= BATCH_SIZE:
            batches = buffer.sample(on_bs, UTD)

            for u in range(UTD):
                obs_u   = batches['observations'][u]
                act_u   = batches['actions'][u]
                rew_u   = batches['rewards'][u]
                mask_u  = batches['masks'][u]
                nobs_u  = batches['next_observations'][u]
                tid_u   = batches['task_ids'][u]

                if offline_buffer is not None and off_bs > 0:
                    ex = offline_buffer.sample_chunk(off_bs, 1, DISCOUNT)
                    ex_obs, ex_act, ex_rew, ex_nobs, ex_done, ex_gid = ex
                    ex_obs_np  = ex_obs.cpu().numpy()
                    ex_act_np  = ex_act.cpu().numpy()
                    ex_rew_np  = ex_rew.cpu().numpy().reshape(-1)
                    ex_nobs_np = ex_nobs.cpu().numpy()
                    ex_done_np = ex_done.cpu().numpy().reshape(-1)
                    ex_mask_np = np.ones_like(ex_done_np)
                    ex_gid_np  = ex_gid.cpu().numpy()
                    ex_lid_np = g2l[ex_gid.long()].cpu().numpy()

                    alpha_val = agent.alpha.item()
                    on_rew_n = normalizer.normalize_parallel(
                        rew_u, tid_u, alpha_val)
                    ex_rew_n = normalizer.normalize_parallel(
                        ex_rew_np, ex_lid_np, alpha_val)

                    on_gid_np = global_task_ids_np[tid_u]

                    b_obs  = np.concatenate([obs_u, ex_obs_np])
                    b_act  = np.concatenate([act_u, ex_act_np])
                    b_rew  = np.concatenate([on_rew_n, ex_rew_n])
                    b_nobs = np.concatenate([nobs_u, ex_nobs_np])
                    b_mask = np.concatenate([mask_u, ex_mask_np])
                    b_gid  = np.concatenate([on_gid_np, ex_gid_np])
                else:
                    alpha_val = agent.alpha.item()
                    b_rew = normalizer.normalize_parallel(
                        rew_u, tid_u, alpha_val)
                    b_obs  = obs_u
                    b_act  = act_u
                    b_nobs = nobs_u
                    b_mask = mask_u
                    b_gid  = global_task_ids_np[tid_u]

                b_obs_t  = torch.from_numpy(b_obs).float().to(device)
                b_act_t  = torch.from_numpy(b_act).float().to(device)
                b_rew_t  = torch.from_numpy(b_rew).float().to(device).unsqueeze(-1)
                b_nobs_t = torch.from_numpy(b_nobs).float().to(device)
                b_gid_t  = torch.from_numpy(b_gid).long().to(device)
                b_term_t = (1.0 - torch.from_numpy(b_mask).float()).to(device).unsqueeze(-1)

                last_metrics = agent.update(
                    batch=(b_obs_t, b_act_t, b_rew_t, b_nobs_t,
                           b_gid_t, b_term_t))

        # --- Logging ---
        if step % LOG_INTERVAL == 0 and last_metrics:
            m = last_metrics
            print(
                f"[S={step:>7d}] "
                f"cri={m.get('critic_loss',0):.3f} "
                f"act={m.get('actor_loss',0):.3f} "
                f"Qpi={m.get('q_pi',0):+.1f} "
                f"a={m.get('alpha',0):.1e} "
                f"ent={m.get('entropy',0):.2f} "
                f"buf={buffer.size}")
            for k, v in m.items():
                if not k.startswith('_'):
                    logger.log(f'train/{k}', v, step)
            logger.dump(step, Ty='train')

        # --- Evaluation ---
        if step % EVAL_INTERVAL == 0 and step >= WARMUP_PER_TASK:
            agent.train(False)
            eval_results = env.evaluate(agent, EVAL_EPISODES, device)
            agent.train(True)

            mean_sr = np.mean(eval_results['goal'])
            mean_ret = np.mean(eval_results['return'])
            logger.log('eval/mean_success_rate', mean_sr, step)
            logger.log('eval/mean_return', mean_ret, step)
            for i, name in enumerate(train_task_names):
                logger.log(f'eval_task/{name}_sr', eval_results['goal'][i], step)
                logger.log(f'eval_task/{name}_return', eval_results['return'][i], step)
            logger.dump(step, Ty='eval')
            print(f"  [Eval@{step}] MeanSR={mean_sr:.1%} MeanRet={mean_ret:.1f}")

        # --- Save ---
        if step % SAVE_INTERVAL == 0:
            snap = agent.save_snapshot()
            torch.save(snap, work_dir / f'snapshot_parallel_tf_{step}.pt')
            torch.save(snap, work_dir / 'snapshot_latest.pt')
            print(f"  Saved snapshot at step {step}")

    logger.close()
    print("Parallel_TF training finished.")


if __name__ == '__main__':
    train()
