"""
ON_CalQL.py — Online Fine-tuning with CQL Regularization (decaying).

Core design:
    Critic update = C51 Loss (online+offline mixed batch)
                  + cql_weight(step) * CQL Penalty (offline batch only, Cal-QL MC calibration)
    Actor  update = Pure SAC loss (no BC term)
    cql_weight    = max(0, 1 - global_step / CQL_DECAY_STEPS)  linear decay

Dual-agent Q-dist visualization (Agent A: CalQLOnlineAgent, Agent B: frozen CalQLAgent).
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
from pathlib import Path
from omegaconf import DictConfig

from experiments.pixel.offline.OFF_CalQL import CalQLAgent
from utils.common import set_seed_everywhere
from utils.offline_train_utils import setup_task_palette
from utils.online_train_utils import (
    get_decay_weight, setup_dual_agents, setup_online_infra,
    make_offline_loaders, mix_online_offline, load_batch_to_device,
    run_online_training,
)


# =====================================================================
# Hyperparameters
# =====================================================================
OFF_RATIO               = 0.5
SAC_P_RATIO             = 0.25
P_SUITE                 = "Expert_Goal_Observable"
N_SUITE                 = "Medium_Goal_Observable"

CQL_N_ACTIONS           = 10
CQL_TEMP                = 1.0
CQL_CONSERVATIVE_WEIGHT = 0.1
CQL_TARGET_ACTION_GAP   = 4.693
CQL_ALPHA_LR            = 3e-4
CQL_DECAY_STEPS         = 500_000

CQL_KWARGS = dict(
    cql_n_actions=CQL_N_ACTIONS,
    cql_temp=CQL_TEMP,
    cql_conservative_weight=CQL_CONSERVATIVE_WEIGHT,
    cql_target_action_gap=CQL_TARGET_ACTION_GAP,
    cql_alpha_lr=CQL_ALPHA_LR,
)


# =====================================================================
# CalQLOnlineAgent  (extends CalQLAgent, adds update_online method)
# =====================================================================

class CalQLOnlineAgent(CalQLAgent):
    """Online Cal-QL Agent with decaying CQL regularization."""

    def get_cql_weight(self, step):
        return get_decay_weight(step, CQL_DECAY_STEPS)

    def _compute_cql_penalty(self, cql_obs, cql_action, cql_next_obs,
                             cql_task_emb, mc_returns=None):
        """CQL regularization penalty + Cal-QL calibration on offline features."""
        metrics = {}
        device = cql_obs.device
        B = cql_obs.shape[0]
        N = self.cql_n_actions

        with torch.no_grad():
            pi_curr = self.actor(cql_task_emb, cql_obs)
            cql_curr_actions = torch.stack(
                [pi_curr.sample() for _ in range(N)], dim=1)
            cql_curr_log_pis = torch.stack(
                [pi_curr.log_prob(cql_curr_actions[:, i]).sum(-1)
                 for i in range(N)], dim=1)

            pi_next = self.actor(cql_task_emb, cql_next_obs)
            cql_next_actions = torch.stack(
                [pi_next.sample() for _ in range(N)], dim=1)
            cql_next_log_pis = torch.stack(
                [pi_next.log_prob(cql_next_actions[:, i]).sum(-1)
                 for i in range(N)], dim=1)

            cql_rand_actions = torch.empty(
                B, N, self.flat_act_dim, device=device).uniform_(-1, 1)

        obs_rep = cql_obs.unsqueeze(1).expand(-1, N, -1).reshape(B * N, -1)
        te_rep  = cql_task_emb.unsqueeze(1).expand(-1, N, -1).reshape(B * N, -1)
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

        cql_data_logits = self.critic(cql_obs, cql_action, cql_task_emb)
        cql_data_probs = F.softmax(cql_data_logits, dim=-1)
        q_data = (cql_data_probs * sup).sum(-1).mean(0)
        cql_diff = (cql_ood - q_data).mean()

        alpha_prime_detached = self.alpha_prime.detach()
        cql_penalty = (alpha_prime_detached * self.cql_conservative_weight
                       * (cql_diff - self.cql_target_action_gap))

        metrics['cql_diff']    = cql_diff.item()
        metrics['cql_penalty'] = cql_penalty.item()
        return cql_penalty, cql_diff, metrics

    def update_online(self, batch, cql_batch=None, step=0):
        """Online CalQL update with dual data streams."""
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

        # ===== 1. C51 Target & Loss =====
        with torch.no_grad():
            next_dist_ = self.actor(task_emb, next_img_feat)
            next_action = next_dist_.sample()
            next_log_prob = next_dist_.log_prob(next_action).sum(-1, keepdim=True)

            metrics['q_next'] = self._get_expected_q(
                next_img_feat, next_action, task_emb).mean().item()

            next_logits = self.critic_target(next_img_feat, next_action, task_emb)
            next_probs_all = F.softmax(next_logits, dim=-1)            # (E, B, bins)
            if self.use_min_q:
                sup_t = self.support.view(1, 1, -1)
                eq = (next_probs_all * sup_t).sum(-1)                  # (E, B)
                min_idx = eq.argmin(dim=0)                             # (B,)
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

        current_logits = self.critic(img_feat, action, task_emb)

        with torch.no_grad():
            metrics['q_data'] = self._get_expected_q(
                img_feat, action, task_emb).mean().item()

        current_log_probs = F.log_softmax(current_logits, dim=-1)
        c51_loss = -(target_dist.unsqueeze(0) * current_log_probs).sum(-1).mean()
        metrics['c51_loss'] = c51_loss.item()

        # ===== 2. CQL Regularization (decaying) =====
        cql_weight = self.get_cql_weight(step)
        metrics['cql_weight'] = cql_weight

        cql_penalty = torch.tensor(0.0, device=pixels.device)
        cql_diff_val = 0.0
        if cql_batch is not None and cql_weight > 0:
            cql_pix, cql_act, cql_rew, cql_next_pix, cql_done, cql_tid, cql_mc = cql_batch

            if self.freeze_encoder:
                with torch.no_grad():
                    cql_feat      = self.encoder(cql_pix)
                    cql_next_feat = self.encoder(cql_next_pix)
                    cql_temb      = self.task_embedding(cql_tid)
            else:
                cql_feat = self.encoder(cql_pix)
                cql_temb = self.task_embedding(cql_tid)
                with torch.no_grad():
                    cql_next_feat = self.encoder(cql_next_pix)

            cql_penalty, cql_diff_t, cql_metrics = self._compute_cql_penalty(
                cql_feat, cql_act, cql_next_feat, cql_temb, mc_returns=cql_mc)
            metrics.update(cql_metrics)
            cql_diff_val = cql_diff_t.item()

        metrics['cql_diff'] = metrics.get('cql_diff', 0.0)
        metrics['cql_penalty'] = metrics.get('cql_penalty', 0.0)

        # ===== 3. Total Critic Loss =====
        critic_loss = c51_loss + cql_weight * cql_penalty
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()
        metrics['critic_loss'] = critic_loss.item()

        if cql_batch is not None and cql_weight > 0:
            alpha_prime_loss = -(self.alpha_prime * self.cql_conservative_weight
                                 * (cql_diff_val - self.cql_target_action_gap))
            self.alpha_prime_opt.zero_grad()
            alpha_prime_loss.backward()
            self.alpha_prime_opt.step()
            metrics['alpha_prime']      = self.alpha_prime.item()
            metrics['alpha_prime_loss'] = alpha_prime_loss.item()
        else:
            metrics['alpha_prime']      = self.alpha_prime.item()
            metrics['alpha_prime_loss'] = 0.0

        # ===== 4. Soft Update =====
        self._soft_update(self.critic, self.critic_target, self.tau)

        # ===== 5. Actor (pure SAC) =====
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
        metrics['sac_loss']   = actor_loss.item()
        metrics['q_pi']       = q_value.mean().item()
        metrics['bc_loss']    = 0.0
        metrics['lmbda']      = 0.0

        # ===== 6. Alpha =====
        alpha_loss = (self.alpha * (-new_log_prob - self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        metrics['alpha_loss'] = alpha_loss.item()
        metrics['alpha']      = self.alpha.item()

        return metrics


# =====================================================================
# Main
# =====================================================================

def _calql_format_log(m, global_step, buf_size):
    return (
        f"  [S={global_step:>7d}] "
        f"cri={m.get('critic_loss',0):.3f} "
        f"c51={m.get('c51_loss',0):.3f} "
        f"cql={m.get('cql_diff',0):+.2f} "
        f"cql_w={m.get('cql_weight',0):.3f} "
        f"a'={m.get('alpha_prime',0):.3f} "
        f"act={m.get('actor_loss',0):.3f} "
        f"Qd={m.get('q_data',0):+.1f} "
        f"Qp={m.get('q_pi',0):+.1f} "
        f"a={m.get('alpha',0):.1e} "
        f"buf={buf_size}")


@hydra.main(config_path=CONFIG_DIR, config_name="config")
def train(cfg: DictConfig):
    device = torch.device(cfg.device)
    set_seed_everywhere(cfg.seed)

    train_task_names, global_to_local, _, g2l = setup_task_palette(cfg, device)
    num_tasks = len(train_task_names)

    def _snapshot_extra(agent, payload):
        if 'log_alpha_prime' in payload:
            agent.log_alpha_prime = payload['log_alpha_prime'].to(device)
            agent.log_alpha_prime.requires_grad = True
            agent.alpha_prime_opt = torch.optim.Adam(
                [agent.log_alpha_prime], lr=CQL_ALPHA_LR)
            if 'alpha_prime_opt' in payload:
                agent.alpha_prime_opt.load_state_dict(payload['alpha_prime_opt'])

    agent, agent_ref = setup_dual_agents(
        cfg, device, CalQLOnlineAgent,
        AgentRefCls=CalQLAgent,
        agent_kwargs=CQL_KWARGS, ref_kwargs=CQL_KWARGS,
        snapshot_extra_fn=_snapshot_extra)

    buffer, normalizer, start_training, data_root = setup_online_infra(
        cfg, device, num_tasks, global_to_local, train_task_names, agent,
        p_suite=P_SUITE, n_suite=N_SUITE)

    batch_size = cfg.train.rl.batch_size
    off_bs = int(batch_size * OFF_RATIO)
    on_bs  = batch_size - off_bs
    off_p  = int(off_bs * SAC_P_RATIO)
    off_n  = off_bs - off_p

    p_iter, n_iter = make_offline_loaders(
        data_root, cfg, train_task_names, off_bs, P_SUITE, N_SUITE)

    chunk_size   = cfg.agent.action_chunking.num_queries
    gamma        = cfg.train.rl.discount
    UTD          = cfg.train.rl.UTD
    update_every = max(1, num_tasks // UTD)
    min_buf      = on_bs + chunk_size

    print(f"=== CalQL Online Fine-tuning | {Path.cwd()} ===")
    print(f"  batch={batch_size}(off={off_bs}[P={off_p}+N={off_n}]+on={on_bs}) UTD={UTD} "
          f"chunk={chunk_size} gamma={gamma} V=[{agent.v_min},{agent.v_max}] "
          f"encoder={'unfrozen' if not agent.freeze_encoder else 'frozen'}")
    print(f"  CQL: N={CQL_N_ACTIONS} temp={CQL_TEMP} weight={CQL_CONSERVATIVE_WEIGHT} "
          f"gap={CQL_TARGET_ACTION_GAP} decay_steps={CQL_DECAY_STEPS}")
    print(f"  Actor: pure SAC (no BC)")

    # ----- Gradient update callback -----
    def update_fn(global_step):
        on = buffer.sample_chunk(on_bs, chunk_size, gamma)
        on_imgs, on_acts, on_rews, on_next, on_dones, on_gids = on
        on_imgs = on_imgs.float().div_(255.0)
        on_next = on_next.float().div_(255.0)
        on_rews_n = normalizer.normalize(on_rews, g2l[on_gids], agent.alpha)

        cql_batch_arg = None
        if off_bs > 0:
            bp_raw = next(p_iter)
            bp = load_batch_to_device(bp_raw, device)
            bp_mc = bp_raw['mc_returns'].to(device, non_blocking=True)
            bn_raw = next(n_iter)
            bn = load_batch_to_device(bn_raw, device)
            bn_mc = bn_raw['mc_returns'].to(device, non_blocking=True)

            pix_m, act_m, rew_m, nxt_m, don_m, gid_m = mix_online_offline(
                (on_imgs, on_acts, on_rews_n, on_next, on_dones, on_gids),
                bp, bn, off_p, off_n, normalizer, g2l, agent.alpha)

            # CQL batch: full P+N with normalized mc_returns
            bp_imgs, bp_next, bp_acts, bp_rews, bp_dones, bp_gids = bp
            bn_imgs, bn_next, bn_acts, bn_rews, bn_dones, bn_gids = bn
            cql_imgs  = torch.cat([bp_imgs, bn_imgs])
            cql_acts  = torch.cat([bp_acts, bn_acts])
            cql_rews  = torch.cat([bp_rews, bn_rews])
            cql_next  = torch.cat([bp_next, bn_next])
            cql_dones = torch.cat([bp_dones, bn_dones])
            cql_gids  = torch.cat([bp_gids, bn_gids])
            cql_mc    = torch.cat([bp_mc, bn_mc])
            cql_mc_n  = normalizer.normalize(cql_mc, g2l[cql_gids], agent.alpha)
            cql_batch_arg = (cql_imgs, cql_acts, cql_rews,
                             cql_next, cql_dones, cql_gids, cql_mc_n)
        else:
            pix_m, act_m, rew_m = on_imgs, on_acts, on_rews_n
            nxt_m, don_m, gid_m = on_next, on_dones, on_gids

        return agent.update_online(
            batch=(pix_m, act_m, rew_m, nxt_m, don_m, gid_m),
            cql_batch=cql_batch_arg, step=global_step)

    run_online_training(
        agent=agent, agent_ref=agent_ref, cfg=cfg,
        train_task_names=train_task_names, g2l=g2l,
        buffer=buffer, normalizer=normalizer,
        start_training=start_training, update_every=update_every, min_buf=min_buf,
        checkpoint_prefix='snapshot_calql_post',
        update_fn=update_fn,
        format_log_fn=_calql_format_log,
        done_message="CalQL Online fine-tuning finished.")


if __name__ == '__main__':
    train()
