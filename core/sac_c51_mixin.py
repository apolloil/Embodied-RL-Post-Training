"""
core/sac_c51_mixin.py — Shared SAC + C51 methods for standalone BRC agents.

Provides a mixin class with methods that are identical across
FaithfulBRCAgent, UltimateBRCAgent, and ParallelTFAgent:
  - alpha property
  - _get_expected_q: distributional → scalar expected Q
  - _update_alpha: SAC temperature update
  - _soft_update: Polyak averaging for target critic
  - _c51_project_target: C51 categorical projection (no_grad block)

Subclasses must set: self.support, self.critic, self.critic_target,
self.log_alpha, self.alpha_opt, self.target_entropy, self.tau,
self.v_min, self.v_max, self.num_bins, self.gamma.
"""

import torch
import torch.nn.functional as F


class SACC51Mixin:
    """Mixin providing shared SAC + C51 distributional RL methods.

    Expects the following attributes on self:
      - support: Tensor (num_bins,)
      - critic, critic_target: DistributionalCritic
      - log_alpha: Tensor (scalar, requires_grad)
      - alpha_opt: optimizer for log_alpha
      - target_entropy: float
      - tau: float (Polyak averaging coefficient)
      - v_min, v_max: float (C51 support range)
      - num_bins: int
      - gamma: float (discount factor)
    """

    @property
    def alpha(self):
        return self.log_alpha.exp()

    # ------------------------------------------------------------------
    # Q-value: distributional → scalar expected Q, mean over ensemble
    # ------------------------------------------------------------------

    def _get_expected_q(self, obs, action, task_emb):
        logits = self.critic(obs, action, task_emb)           # (E, B, bins)
        probs = F.softmax(logits, dim=-1)                     # (E, B, bins)
        support = self.support.view(1, 1, -1)                 # (1, 1, bins)
        q_per_member = (probs * support).sum(-1)              # (E, B)
        return q_per_member.mean(0).unsqueeze(-1)             # (B, 1)

    # ------------------------------------------------------------------
    # C51 categorical projection (shared no_grad block)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _c51_project_target(self, next_obs_or_feat, task_emb,
                            reward, actor_fn):
        """Compute C51 target distribution.

        Args:
            next_obs_or_feat: encoded features or raw obs for next state.
            task_emb: task embedding tensor.
            reward: (B, 1) reward tensor.
            actor_fn: callable(task_emb, obs) -> distribution with .sample()
                      and .log_prob(). For BRC actors, wrap as lambda.

        Returns:
            target_dist: (B, num_bins) target categorical distribution.
            q_mean: float, mean target Q for logging.
        """
        next_dist = actor_fn(task_emb, next_obs_or_feat)
        next_action = next_dist.sample()
        next_log_prob = next_dist.log_prob(next_action).sum(-1, keepdim=True)

        next_logits = self.critic_target(next_obs_or_feat, next_action, task_emb)
        next_probs = F.softmax(next_logits, dim=-1).mean(dim=0)

        alpha_val = self.alpha.detach()
        next_support = self.support.unsqueeze(0) - alpha_val * next_log_prob
        target_support = reward + self.gamma * next_support  # mask=1 always
        target_support = target_support.clamp(self.v_min, self.v_max)

        delta_z = (self.v_max - self.v_min) / (self.num_bins - 1)
        b = (target_support - self.v_min) / delta_z
        b = b.clamp(0, self.num_bins - 1)
        l = b.floor().long()
        u = torch.clamp(l + 1, max=self.num_bins - 1)
        d = b - l.float()

        target_dist = torch.zeros_like(next_probs)
        target_dist.scatter_add_(1, l, next_probs * (1.0 - d))
        target_dist.scatter_add_(1, u, next_probs * d)

        q_mean = (self.support.unsqueeze(0) * target_dist).sum(-1).mean().item()
        return target_dist, q_mean

    # ------------------------------------------------------------------
    # Temperature update
    # ------------------------------------------------------------------

    def _update_alpha(self, entropy):
        alpha_loss = (self.alpha * (entropy - self.target_entropy)).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()
        return {'alpha': self.alpha.item(), 'alpha_loss': alpha_loss.item()}

    # ------------------------------------------------------------------
    # Soft update (Polyak averaging)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _soft_update(self):
        for p, tp in zip(self.critic.parameters(),
                         self.critic_target.parameters()):
            tp.data.lerp_(p.data, self.tau)
