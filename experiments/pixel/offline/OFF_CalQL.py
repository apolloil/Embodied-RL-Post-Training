"""
OFF_CalQL.py — Cal-QL (Calibrated Q-Learning) Offline RL Training.

CQL regularisation + Cal-QL MC Return calibration on top of C51 Cross-Entropy
Critic Loss.  Actor Loss is pure SAC (no BC term).

Core idea (Nakamoto et al., 2023):
  CQL:    logsumexp_a Q(s,a) - Q(s,a_data)  -- penalise OOD, boost data Q
  Cal-QL: max(Q(s,a_pi), mc_return) replaces Q(s,a_pi) to avoid over-conservatism
  Lagrange: alpha_prime auto-tunes CQL coefficient so cql_diff ~ target_action_gap
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

import hydra
import torch
import torch.nn.functional as F
import numpy as np
from omegaconf import DictConfig

from core.agent import RLAgent
from utils.common import set_seed_everywhere
from utils.offline_train_utils import (
    setup_task_palette, setup_offline_infra, run_offline_training,
)

# =====================================================================
# Hyperparameters (CalQL-specific only)
# =====================================================================
BC_ALPHA                = 0
CQL_N_ACTIONS           = 10
CQL_TEMP                = 1.0
CQL_CONSERVATIVE_WEIGHT = float(os.environ.get("CQL_WEIGHT", 0.1))
CQL_TARGET_ACTION_GAP   = float(os.environ.get("CQL_GAP", 4.693))
CQL_ALPHA_LR            = 3e-4


# =====================================================================
# CalQLAgent  (subclasses RLAgent, overrides critic update logic)
# =====================================================================

class CalQLAgent(RLAgent):

    def __init__(self, device, cfg,
                 cql_n_actions=CQL_N_ACTIONS,
                 cql_temp=CQL_TEMP,
                 cql_conservative_weight=CQL_CONSERVATIVE_WEIGHT,
                 cql_target_action_gap=CQL_TARGET_ACTION_GAP,
                 cql_alpha_lr=CQL_ALPHA_LR):
        super().__init__(device, cfg)

        self.cql_n_actions           = cql_n_actions
        self.cql_temp                = cql_temp
        self.cql_conservative_weight = cql_conservative_weight
        self.cql_target_action_gap   = cql_target_action_gap

        self.log_alpha_prime = torch.tensor(
            np.log(1.0), dtype=torch.float32, device=device)
        self.log_alpha_prime.requires_grad = True
        self.alpha_prime_opt = torch.optim.Adam(
            [self.log_alpha_prime], lr=cql_alpha_lr)

    @property
    def alpha_prime(self):
        return torch.clamp(self.log_alpha_prime.exp(), 0.0, 1e6)

    # ------------------------------------------------------------------
    # CQL/Cal-QL Critic Update
    # ------------------------------------------------------------------

    def _update_critic(self, obs, action, reward, next_obs, done, task_emb,
                       return_dists=False, dist_sample_idx=0, mc_returns=None):
        metrics = {}
        device = obs.device
        B = obs.shape[0]
        N = self.cql_n_actions

        with torch.no_grad():
            next_dist_  = self.actor(task_emb, next_obs)
            next_action   = next_dist_.sample()
            next_log_prob = next_dist_.log_prob(next_action).sum(-1, keepdim=True)

            metrics['q_next'] = self._get_expected_q(
                next_obs, next_action, task_emb).mean().item()

            next_logits = self.critic_target(next_obs, next_action, task_emb)
            next_probs_all = F.softmax(next_logits, dim=-1)
            if self.use_min_q:
                sup_t = self.support.view(1, 1, -1)
                eq = (next_probs_all * sup_t).sum(-1)
                min_idx = eq.argmin(dim=0)
                B_t = next_logits.size(1)
                next_probs = next_probs_all[min_idx, torch.arange(B_t, device=next_logits.device)]
            else:
                next_probs = next_probs_all.mean(dim=0)

            alpha_val      = self.alpha.detach()
            next_support   = self.support.unsqueeze(0) - alpha_val * next_log_prob
            target_support = reward + self.gamma * next_support
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
        c51_loss = -(target_dist.unsqueeze(0) * current_log_probs).sum(-1).mean()

        # CQL / Cal-QL regularisation
        with torch.no_grad():
            pi_curr = self.actor(task_emb, obs)
            cql_curr_actions = torch.stack(
                [pi_curr.sample() for _ in range(N)], dim=1)
            cql_curr_log_pis = torch.stack(
                [pi_curr.log_prob(cql_curr_actions[:, i]).sum(-1)
                 for i in range(N)], dim=1)

            pi_next = self.actor(task_emb, next_obs)
            cql_next_actions = torch.stack(
                [pi_next.sample() for _ in range(N)], dim=1)
            cql_next_log_pis = torch.stack(
                [pi_next.log_prob(cql_next_actions[:, i]).sum(-1)
                 for i in range(N)], dim=1)

            cql_rand_actions = torch.empty(
                B, N, self.flat_act_dim, device=device).uniform_(-1, 1)

        obs_rep = obs.unsqueeze(1).expand(-1, N, -1).reshape(B * N, -1)
        te_rep  = task_emb.unsqueeze(1).expand(-1, N, -1).reshape(B * N, -1)
        sup     = self.support.view(1, 1, -1)

        def _q_batch(acts):
            logits = self.critic(obs_rep, acts.reshape(B * N, -1), te_rep)
            probs  = F.softmax(logits, dim=-1)
            return (probs * sup).sum(-1).mean(0).reshape(B, N)

        q_rand = _q_batch(cql_rand_actions)
        q_curr = _q_batch(cql_curr_actions)
        q_next = _q_batch(cql_next_actions)

        if mc_returns is not None:
            mc_exp = mc_returns.expand(-1, N)
            q_curr = torch.max(q_curr, mc_exp)
            q_next = torch.max(q_next, mc_exp)

        random_density = np.log(0.5 ** self.flat_act_dim)
        cat_q = torch.cat([
            (q_rand - random_density)     / self.cql_temp,
            (q_curr - cql_curr_log_pis)   / self.cql_temp,
            (q_next - cql_next_log_pis)   / self.cql_temp,
        ], dim=1)

        cql_ood = torch.logsumexp(cat_q, dim=1) * self.cql_temp

        q_data_probs = F.softmax(current_logits, dim=-1)
        q_data = (q_data_probs * sup).sum(-1).mean(0)
        cql_diff = (cql_ood - q_data).mean()

        alpha_prime_detached = self.alpha_prime.detach()
        cql_penalty = (alpha_prime_detached * self.cql_conservative_weight
                       * (cql_diff - self.cql_target_action_gap))

        critic_loss = c51_loss + cql_penalty

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        alpha_prime_loss = -(self.alpha_prime * self.cql_conservative_weight
                             * (cql_diff.detach() - self.cql_target_action_gap))
        self.alpha_prime_opt.zero_grad()
        alpha_prime_loss.backward()
        self.alpha_prime_opt.step()

        metrics['critic_loss']      = critic_loss.item()
        metrics['c51_loss']         = c51_loss.item()
        metrics['cql_diff']         = cql_diff.item()
        metrics['cql_penalty']      = cql_penalty.item()
        metrics['alpha_prime']      = self.alpha_prime.item()
        metrics['alpha_prime_loss'] = alpha_prime_loss.item()
        return metrics

    # ------------------------------------------------------------------
    # Offline Update  (pure SAC actor, no BC)
    # ------------------------------------------------------------------

    def update(self, batch, bc_alpha=2.5, return_dists=False,
               dist_sample_idx=0, use_bc_loss=False, mc_returns=None,
               bc_batch=None):
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

        critic_metrics = self._update_critic(
            img_feat, action, reward, next_img_feat, done, task_emb,
            return_dists=return_dists, dist_sample_idx=dist_sample_idx,
            mc_returns=mc_returns)
        metrics.update(critic_metrics)

        self._soft_update(self.critic, self.critic_target, self.tau)

        img_feat_d = img_feat.detach()
        task_emb_fresh = self.task_embedding(task_ids)

        dist       = self.actor(task_emb_fresh, img_feat_d)
        new_action = dist.rsample()
        new_log_prob = dist.log_prob(new_action).sum(-1, keepdim=True)

        q_value  = self._get_expected_q(img_feat_d, new_action, task_emb_fresh)
        sac_loss = (self.alpha.detach() * new_log_prob - q_value).mean()

        actor_loss = sac_loss
        metrics['bc_loss'] = 0.0
        metrics['lmbda']   = 0.0

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        metrics['actor_loss'] = actor_loss.item()
        metrics['sac_loss']   = sac_loss.item()
        metrics['q_pi']       = q_value.mean().item()

        alpha_loss = (self.alpha * (-new_log_prob - self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        metrics['alpha_loss'] = alpha_loss.item()
        metrics['alpha']      = self.alpha.item()

        return metrics

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save_snapshot(self):
        d = super().save_snapshot()
        d['log_alpha_prime'] = self.log_alpha_prime
        d['alpha_prime_opt'] = self.alpha_prime_opt.state_dict()
        return d

    def load_snapshot(self, payload):
        super().load_snapshot(payload)
        if 'log_alpha_prime' in payload:
            self.log_alpha_prime = payload['log_alpha_prime'].to(self.device)
            self.log_alpha_prime.requires_grad = True
            self.alpha_prime_opt = torch.optim.Adam(
                [self.log_alpha_prime], lr=self.alpha_prime_opt.defaults['lr'])
        if 'alpha_prime_opt' in payload:
            self.alpha_prime_opt.load_state_dict(payload['alpha_prime_opt'])


# =====================================================================
# Main
# =====================================================================

def _calql_format_log(m, step, num_steps, fps):
    return (
        f"[Step {step:>7d}/{num_steps}]  "
        f"FPS {fps:5.0f}  "
        f"critic {m.get('critic_loss', 0):.4f}  "
        f"c51 {m.get('c51_loss', 0):.4f}  "
        f"cql_diff {m.get('cql_diff', 0):+.3f}  "
        f"alpha' {m.get('alpha_prime', 0):.4f}  "
        f"actor {m.get('actor_loss', 0):.4f}  "
        f"alpha {m.get('alpha', 0):.5f}  "
        f"Q_data {m.get('q_data', 0):+.2f}  "
        f"Q_pi {m.get('q_pi', 0):+.2f}")


@hydra.main(config_path=CONFIG_DIR, config_name="config")
def train(cfg: DictConfig):
    device = torch.device(cfg.device)
    set_seed_everywhere(cfg.seed)

    train_task_names, global_to_local, gid_to_name, g2l_tensor = \
        setup_task_palette(cfg, device)

    agent = CalQLAgent(
        device, cfg,
        cql_n_actions=CQL_N_ACTIONS, cql_temp=CQL_TEMP,
        cql_conservative_weight=CQL_CONSERVATIVE_WEIGHT,
        cql_target_action_gap=CQL_TARGET_ACTION_GAP,
        cql_alpha_lr=CQL_ALPHA_LR)

    normalizer, p_iter, n_iter, B_p, B_n, batch_size = \
        setup_offline_infra(cfg, device, train_task_names, global_to_local, agent)

    print(f"=== Cal-QL Offline | batch={batch_size}(P={B_p},N={B_n}) "
          f"CQL: N={CQL_N_ACTIONS} temp={CQL_TEMP} weight={CQL_CONSERVATIVE_WEIGHT} "
          f"gap={CQL_TARGET_ACTION_GAP} ===\n")

    def update_fn(sac, bc_batch, rewards_norm, mc_returns_norm, capture, idx):
        return agent.update(
            batch=(sac['pixels'], sac['flat_actions'], rewards_norm,
                   sac['next_pixels'], sac['dones'], sac['task_ids']),
            bc_alpha=BC_ALPHA,
            return_dists=capture, dist_sample_idx=idx,
            use_bc_loss=False,
            mc_returns=mc_returns_norm,
            bc_batch=bc_batch)

    run_offline_training(
        agent=agent, cfg=cfg,
        train_task_names=train_task_names, g2l_tensor=g2l_tensor,
        gid_to_name=gid_to_name,
        normalizer=normalizer, p_iter=p_iter, n_iter=n_iter,
        B_p=B_p, B_n=B_n, batch_size=batch_size,
        checkpoint_prefix='snapshot_calql',
        update_fn=update_fn,
        format_log_fn=_calql_format_log,
        done_message="Cal-QL Offline Training finished.")


if __name__ == '__main__':
    train()
