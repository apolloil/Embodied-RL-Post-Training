"""
OFF_TD3BC.py — Faithful TD3+BC Offline RL with C51 Distributional Critic.

Key differences from SAC+BC (OFF_BRC_BC):
  1. No entropy regularisation in target distribution
  2. Deterministic policy (dist.mean, not rsample)
  3. Target policy smoothing (clipped Gaussian noise)
  4. Delayed policy update (one Actor update per policy_freq Critic updates)
  5. No temperature alpha learning
"""

import os
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"

import sys
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
CONFIG_DIR = str(REPO_ROOT / "conf")

import copy
import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from core.agent import RLAgent
from utils.common import set_seed_everywhere
from utils.offline_train_utils import (
    setup_task_palette, setup_offline_infra, run_offline_training,
)

# =====================================================================
# Hyperparameters (TD3BC-specific only)
# =====================================================================
BC_ALPHA     = 2.5
POLICY_NOISE = 0.2
NOISE_CLIP   = 0.2
POLICY_FREQ  = 2


# =====================================================================
# TD3BCAgent  (subclasses RLAgent, overrides update logic)
# =====================================================================

class TD3BCAgent(RLAgent):

    def __init__(self, device, cfg,
                 bc_alpha=BC_ALPHA,
                 policy_noise=POLICY_NOISE,
                 noise_clip=NOISE_CLIP,
                 policy_freq=POLICY_FREQ):
        super().__init__(device, cfg)

        self.bc_alpha     = bc_alpha
        self.policy_noise = policy_noise
        self.noise_clip   = noise_clip
        self.policy_freq  = policy_freq
        self._critic_step_count = 0

        self.actor_target = copy.deepcopy(self.actor)
        self.actor_target.requires_grad_(False)

    def train(self, training=True):
        super().train(training)
        self.actor_target.eval()

    # ------------------------------------------------------------------
    # TD3BC Critic Update
    # ------------------------------------------------------------------

    def _update_critic_td3bc(self, obs, action, reward, next_obs, done,
                             task_emb, return_dists=False, dist_sample_idx=0):
        metrics = {}

        with torch.no_grad():
            next_dist   = self.actor_target(task_emb, next_obs)
            next_action = next_dist.mean

            noise = (torch.randn_like(next_action) * self.policy_noise
                     ).clamp(-self.noise_clip, self.noise_clip)
            next_action = (next_action + noise).clamp(-1.0, 1.0)

            metrics['q_next'] = self._get_expected_q(
                next_obs, next_action, task_emb).mean().item()

            next_logits    = self.critic_target(next_obs, next_action, task_emb)
            next_probs_all = F.softmax(next_logits, dim=-1)

            if self.use_min_q:
                sup         = self.support.view(1, 1, -1)
                expected_q  = (next_probs_all * sup).sum(-1)
                min_member  = expected_q.argmin(dim=0)
                B           = next_logits.size(1)
                next_probs  = next_probs_all[min_member, torch.arange(B, device=next_logits.device)]
            else:
                next_probs  = next_probs_all.mean(dim=0)

            target_support = reward + self.gamma * self.support.unsqueeze(0)
            target_support = target_support.clamp(self.v_min, self.v_max)

            delta_z = float(self.v_max - self.v_min) / (self.num_bins - 1)
            b = torch.clamp((target_support - self.v_min) / delta_z,
                            0, self.num_bins - 1)
            l = b.floor().long()
            u = torch.clamp(l + 1, max=self.num_bins - 1)
            d = b - l.float()

            target_dist = torch.zeros_like(next_probs)
            target_dist.scatter_add_(1, l, next_probs * (1.0 - d))
            target_dist.scatter_add_(1, u, next_probs * d)

        current_logits = self.critic(obs, action, task_emb)

        with torch.no_grad():
            metrics['q_data'] = self._get_expected_q(
                obs, action, task_emb).mean().item()

        if return_dists:
            with torch.no_grad():
                cur_probs = F.softmax(current_logits, dim=-1).mean(dim=0)
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
    # TD3BC Full Update (delayed policy)
    # ------------------------------------------------------------------

    def update(self, batch, bc_alpha=None, return_dists=False,
               dist_sample_idx=0, use_bc_loss=True, bc_batch=None,
               bc_weight=1.0, **kwargs):
        metrics = {}
        pixels, action, reward, next_pixels, done, task_ids = batch
        metrics['reward'] = reward.mean().item()

        if self.freeze_encoder:
            with torch.no_grad():
                img_feat      = self.encoder(pixels)
                next_img_feat = self.encoder(next_pixels)
                task_emb      = self.task_embedding(task_ids)
        else:
            img_feat = self.encoder(pixels)
            task_emb = self.task_embedding(task_ids)
            with torch.no_grad():
                next_img_feat = self.encoder(next_pixels)

        self._critic_step_count += 1
        critic_metrics = self._update_critic_td3bc(
            img_feat, action, reward, next_img_feat, done, task_emb,
            return_dists=return_dists, dist_sample_idx=dist_sample_idx)
        metrics.update(critic_metrics)

        if self._critic_step_count % self.policy_freq == 0:
            img_feat_d = img_feat.detach()
            task_emb_d = task_emb.detach()

            dist      = self.actor(task_emb_d, img_feat_d)
            pi_action = dist.mean

            q_value = self._get_expected_q(img_feat_d, pi_action, task_emb_d)
            q_loss  = -q_value.mean()

            if use_bc_loss:
                if bc_batch is not None:
                    bc_pix, bc_act, _, _, _, bc_tid = bc_batch
                    with torch.no_grad():
                        bc_feat = self.encoder(bc_pix).detach()
                        bc_temb = self.task_embedding(bc_tid).detach()
                    bc_dist = self.actor(bc_temb, bc_feat)
                    bc_loss = F.mse_loss(bc_dist.mean, bc_act)
                else:
                    bc_loss = F.mse_loss(pi_action, action)
                alpha_val = self.bc_alpha if bc_alpha is None else bc_alpha
                lmbda = alpha_val / (q_value.abs().mean().detach() + 1e-8)
                actor_loss = bc_weight * bc_loss + lmbda * q_loss
                metrics['bc_loss']   = bc_loss.item()
                metrics['lmbda']     = lmbda.item()
                metrics['bc_weight'] = bc_weight
            else:
                actor_loss = q_loss
                metrics['bc_loss']   = 0.0
                metrics['lmbda']     = 0.0
                metrics['bc_weight'] = 0.0

            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            metrics['actor_loss'] = actor_loss.item()
            metrics['q_pi']       = q_value.mean().item()

            self._soft_update(self.critic, self.critic_target, self.tau)
            self._soft_update(self.actor,  self.actor_target,  self.tau)
        else:
            metrics['actor_loss']  = 0.0
            metrics['bc_loss']     = 0.0
            metrics['lmbda']       = 0.0
            metrics['bc_weight']   = bc_weight
            metrics['q_pi']        = 0.0

        metrics['alpha']      = 0.0
        metrics['alpha_loss'] = 0.0
        metrics['sac_loss']   = 0.0

        return metrics

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save_snapshot(self):
        d = super().save_snapshot()
        d['actor_target']        = self.actor_target.state_dict()
        d['_critic_step_count']  = self._critic_step_count
        return d

    def load_snapshot(self, payload):
        super().load_snapshot(payload)
        if 'actor_target' in payload:
            self.actor_target.load_state_dict(payload['actor_target'])
        else:
            self.actor_target.load_state_dict(self.actor.state_dict())
        if '_critic_step_count' in payload:
            self._critic_step_count = payload['_critic_step_count']


# =====================================================================
# Main
# =====================================================================

@hydra.main(config_path=CONFIG_DIR, config_name="config")
def train(cfg: DictConfig):
    device = torch.device(cfg.device)
    set_seed_everywhere(cfg.seed)

    train_task_names, global_to_local, gid_to_name, g2l_tensor = \
        setup_task_palette(cfg, device)

    agent = TD3BCAgent(
        device, cfg,
        bc_alpha=BC_ALPHA, policy_noise=POLICY_NOISE,
        noise_clip=NOISE_CLIP, policy_freq=POLICY_FREQ)

    if cfg.train.rl.use_rl_snapshot:
        if not cfg.train.rl.rl_snapshot:
            raise ValueError(
                "train.rl.use_rl_snapshot=true but train.rl.rl_snapshot is not set. "
                "Pass train.rl.rl_snapshot=/path/to/snapshot.pt."
            )
        snapshot_path = os.path.expanduser(str(cfg.train.rl.rl_snapshot))
        if not os.path.exists(snapshot_path):
            raise FileNotFoundError(f"RL snapshot not found: {snapshot_path}")
        print(f"Loading RL snapshot: {snapshot_path}")
        payload = torch.load(snapshot_path, map_location=device)
        agent.load_snapshot(payload)
        print("RL snapshot loaded.")

    normalizer, p_iter, n_iter, B_p, B_n, batch_size = \
        setup_offline_infra(cfg, device, train_task_names, global_to_local, agent)

    print(f"=== TD3+BC Offline | BC_ALPHA={BC_ALPHA} noise={POLICY_NOISE}(clip={NOISE_CLIP}) "
          f"policy_freq={POLICY_FREQ} batch={batch_size}(P={B_p},N={B_n}) ===\n")

    def update_fn(sac, bc_batch, rewards_norm, mc_returns_norm, capture, idx):
        return agent.update(
            batch=(sac['pixels'], sac['flat_actions'], rewards_norm,
                   sac['next_pixels'], sac['dones'], sac['task_ids']),
            bc_alpha=BC_ALPHA,
            return_dists=capture, dist_sample_idx=idx,
            bc_batch=bc_batch)

    run_offline_training(
        agent=agent, cfg=cfg,
        train_task_names=train_task_names, g2l_tensor=g2l_tensor,
        gid_to_name=gid_to_name,
        normalizer=normalizer, p_iter=p_iter, n_iter=n_iter,
        B_p=B_p, B_n=B_n, batch_size=batch_size,
        checkpoint_prefix='snapshot_td3bc',
        update_fn=update_fn,
        norm_alpha_fn=lambda a: 0.0,
        done_message="Offline TD3+BC Training finished.")


if __name__ == '__main__':
    train()
