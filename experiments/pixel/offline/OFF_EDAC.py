"""
EDAC — Uncertainty-Based Offline RL with Diversified Q-Ensemble.

Core design (An et al., NeurIPS 2021):
    Critic Loss = C51 Cross-Entropy + eta * Diversity Loss
    Diversity   = cosine similarity of action gradients across ensemble members
    Actor  Loss = pure SAC loss (alpha*log_pi - Q(s, pi(s))), no BC term

Ensemble size is overridden to EDAC_NUM_QS (default 10).
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
from omegaconf import DictConfig, read_write

from core.agent import RLAgent
from utils.common import set_seed_everywhere
from utils.offline_train_utils import (
    setup_task_palette, setup_offline_infra, run_offline_training,
)

# =====================================================================
# Hyperparameters (EDAC-specific only)
# =====================================================================
EDAC_NUM_QS = 10
EDAC_ETA    = 1.0


# =====================================================================
# EDACAgent  (subclasses RLAgent, overrides critic update logic)
# =====================================================================

class EDACAgent(RLAgent):

    def __init__(self, device, cfg, num_qs=EDAC_NUM_QS, eta=EDAC_ETA):
        with read_write(cfg):
            original_ensemble_size = cfg.train.rl.critic.ensemble_size
            cfg.train.rl.critic.ensemble_size = num_qs

        super().__init__(device, cfg)

        with read_write(cfg):
            cfg.train.rl.critic.ensemble_size = original_ensemble_size

        self.eta = eta
        self.num_qs = num_qs

    # ------------------------------------------------------------------
    # EDAC Critic Update:  C51 Cross-Entropy + eta * Gradient Diversity
    # ------------------------------------------------------------------

    def _update_critic(self, obs, action, reward, next_obs, done, task_emb,
                       return_dists=False, dist_sample_idx=0):
        metrics = {}
        device = obs.device
        E = self.num_qs

        with torch.no_grad():
            next_dist_ = self.actor(task_emb, next_obs)
            next_action = next_dist_.sample()

            if self.target_noise_sigma > 0:
                noise = (torch.randn_like(next_action) * self.target_noise_sigma
                         ).clamp(-self.target_noise_clip, self.target_noise_clip)
                next_action = (next_action + noise).clamp(-1.0, 1.0)

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

            alpha_val = self.alpha.detach()
            next_support = self.support.unsqueeze(0) - alpha_val * next_log_prob
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

        action_rg = action.detach().requires_grad_(True)
        current_logits = self.critic(obs, action_rg, task_emb)

        with torch.no_grad():
            metrics['q_data'] = self._get_expected_q(
                obs, action, task_emb).mean().item()

        if return_dists:
            with torch.no_grad():
                cur_probs = F.softmax(current_logits, dim=-1).mean(dim=0)
                idx = dist_sample_idx
                metrics['_current_probs'] = cur_probs[idx].cpu().numpy()
                metrics['_target_probs'] = target_dist[idx].cpu().numpy()

        current_log_probs = F.log_softmax(current_logits, dim=-1)
        c51_loss = -(target_dist.unsqueeze(0) * current_log_probs).sum(-1).mean()

        if self.eta > 0:
            probs = F.softmax(current_logits, dim=-1)
            sup = self.support.view(1, 1, -1)
            expected_q = (probs * sup).sum(-1)

            grads = []
            for i in range(E):
                g = torch.autograd.grad(
                    expected_q[i].sum(), action_rg,
                    retain_graph=True, create_graph=True)[0]
                grads.append(g)
            grads = torch.stack(grads, dim=0)

            grads = grads / (grads.norm(p=2, dim=-1, keepdim=True) + 1e-10)

            grads_t = grads.permute(1, 0, 2)
            cos_sim = torch.einsum('bik,bjk->bij', grads_t, grads_t)
            mask = torch.eye(E, device=device).unsqueeze(0)
            diversity_loss = ((1 - mask) * cos_sim).sum(dim=(1, 2)).mean() / (E - 1)

            metrics['diversity_loss'] = diversity_loss.item()
        else:
            diversity_loss = 0.0
            metrics['diversity_loss'] = 0.0

        critic_loss = c51_loss + self.eta * diversity_loss

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        metrics['critic_loss'] = critic_loss.item() if isinstance(critic_loss, torch.Tensor) else critic_loss
        metrics['c51_loss'] = c51_loss.item()
        return metrics

    # ------------------------------------------------------------------
    # EDAC Offline Update  (pure SAC Actor, no BC)
    # ------------------------------------------------------------------

    def update(self, batch, return_dists=False, dist_sample_idx=0, **kwargs):
        metrics = {}
        pixels, action, reward, next_pixels, done, task_ids = batch
        metrics['reward'] = reward.mean().item()

        if self.freeze_encoder:
            with torch.no_grad():
                img_feat = self.encoder(pixels)
                next_img_feat = self.encoder(next_pixels)
                task_emb = self.task_embedding(task_ids)
        else:
            img_feat = self.encoder(pixels)
            task_emb = self.task_embedding(task_ids)
            with torch.no_grad():
                next_img_feat = self.encoder(next_pixels)

        critic_metrics = self._update_critic(
            img_feat, action, reward, next_img_feat, done, task_emb,
            return_dists=return_dists, dist_sample_idx=dist_sample_idx)
        metrics.update(critic_metrics)

        self._soft_update(self.critic, self.critic_target, self.tau)

        img_feat_d = img_feat.detach()
        task_emb_fresh = self.task_embedding(task_ids)

        dist = self.actor(task_emb_fresh, img_feat_d)
        new_action = dist.rsample()
        new_log_prob = dist.log_prob(new_action).sum(-1, keepdim=True)

        q_value = self._get_expected_q(img_feat_d, new_action, task_emb_fresh)
        actor_loss = (self.alpha.detach() * new_log_prob - q_value).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        metrics['actor_loss'] = actor_loss.item()
        metrics['sac_loss'] = actor_loss.item()
        metrics['q_pi'] = q_value.mean().item()
        metrics['bc_loss'] = 0.0
        metrics['lmbda'] = 0.0

        alpha_loss = (self.alpha * (-new_log_prob - self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        metrics['alpha_loss'] = alpha_loss.item()
        metrics['alpha'] = self.alpha.item()

        return metrics


# =====================================================================
# Main
# =====================================================================

def _edac_format_log(m, step, num_steps, fps):
    return (
        f"[Step {step:>7d}/{num_steps}]  "
        f"FPS {fps:5.0f}  "
        f"critic {m.get('critic_loss', 0):.4f}  "
        f"c51 {m.get('c51_loss', 0):.4f}  "
        f"div {m.get('diversity_loss', 0):.4f}  "
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

    agent = EDACAgent(device, cfg, num_qs=EDAC_NUM_QS, eta=EDAC_ETA)

    normalizer, p_iter, n_iter, B_p, B_n, batch_size = \
        setup_offline_infra(cfg, device, train_task_names, global_to_local, agent)

    print(f"=== EDAC Offline | Ensemble={EDAC_NUM_QS} eta={EDAC_ETA} "
          f"use_min_q={agent.use_min_q} batch={batch_size}(P={B_p},N={B_n}) ===\n")

    def update_fn(sac, bc_batch, rewards_norm, mc_returns_norm, capture, idx):
        return agent.update(
            batch=(sac['pixels'], sac['flat_actions'], rewards_norm,
                   sac['next_pixels'], sac['dones'], sac['task_ids']),
            return_dists=capture, dist_sample_idx=idx)

    run_offline_training(
        agent=agent, cfg=cfg,
        train_task_names=train_task_names, g2l_tensor=g2l_tensor,
        gid_to_name=gid_to_name,
        normalizer=normalizer, p_iter=p_iter, n_iter=n_iter,
        B_p=B_p, B_n=B_n, batch_size=batch_size,
        checkpoint_prefix='snapshot_edac',
        update_fn=update_fn,
        format_log_fn=_edac_format_log,
        done_message="EDAC Offline Training finished.")


if __name__ == '__main__':
    train()
