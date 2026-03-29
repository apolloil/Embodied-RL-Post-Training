import numpy as np
import torch

class RewardNormalizer:
    """
    Reward Normalizer aligned with BRC (BiggerRegularizedCategorical) logic.
    Customized for Meta-World / Embodied-RL-Post-Training:
    1. Updates based on full trajectories per task.
    2. Assumes 'truncate=True' (Fixed Horizon), forcing bootstrap value estimation.
    3. Uses specific denominator scaling formula.
    """
    def __init__(self, num_tasks, action_dim, discount, v_max, target_entropy=None):
        """
        Args:
            num_tasks: Total number of tasks (e.g. 50).
            action_dim: Flattened action dimension (chunk_size * raw_act_dim).
            discount: Gamma.
            v_max: C51 V_max support.
            target_entropy: SAC target entropy (negative). Defaults to -action_dim.
        """
        self.num_tasks = num_tasks
        self.action_dim = action_dim
        self.discount = discount
        self.v_max = v_max
        self.target_entropy = float(target_entropy) if target_entropy is not None else -float(action_dim)
        
        # Effective horizon = 1 / (1 - gamma)
        self.effective_horizon = 1.0 / (1.0 - discount)
        
        # Initialize stats (Min/Max Returns per task)
        # BRC logic: start min at inf, max at -inf
        self.returns_min = np.zeros(num_tasks, dtype=np.float32) + np.inf
        self.returns_max = np.zeros(num_tasks, dtype=np.float32) - np.inf

    def _calculate_return_stats(self, rewards):
        """
        Calculates min and max returns for a single trajectory.
        Args:
            rewards: reward sequence.
        Bootstrap logic:
            - Always bootstrap = mean(rewards) * horizon (BRC original)
        """
        rewards = np.array(rewards, dtype=np.float32)
        T = len(rewards)
        if T == 0:
            return np.inf, -np.inf

        values = np.zeros_like(rewards)
        
        bootstrap = rewards.mean() * self.effective_horizon
        
        # Backward calculation of returns
        # V_t = r_t + gamma * V_{t+1}
        running_return = bootstrap
        
        for t in reversed(range(T)):
            running_return = rewards[t] + self.discount * running_return
            values[t] = running_return
            
        return values.min(), values.max()

    def update(self, task_id, reward_list):
        """
        Update statistics for a specific task using a finished trajectory.
        Args:
            task_id: int, ID of the task.
            reward_list: list or np.array of raw rewards.
        """
        # Check if reward_list is empty (works for both list and numpy array)
        if len(reward_list) == 0:
            return

        v_min, v_max = self._calculate_return_stats(reward_list)
        
        # Update global stats for this task
        if v_min != np.inf:
            self.returns_min[task_id] = min(self.returns_min[task_id], v_min)
        if v_max != -np.inf:
            self.returns_max[task_id] = max(self.returns_max[task_id], v_max)

    def normalize(self, batch_rewards, batch_task_ids, temperature):
        """
        Normalize batch rewards.
        Formula: denominator = (max_abs_ret + alpha * |target_entropy|/2 * horizon) / v_max
        
        Args:
            batch_rewards: Tensor (B, 1)
            batch_task_ids: Tensor (B,)
            temperature: Tensor or float (Current Alpha)
        """
        device = batch_rewards.device
        
        # 1. Gather stats for batch tasks (CPU -> GPU)
        ids_cpu = batch_task_ids.cpu().numpy().astype(int)
        
        min_rets = self.returns_min[ids_cpu]
        max_rets = self.returns_max[ids_cpu]
        
        # Calculate Max Absolute Return: max(|min|, |max|)
        max_abs_ret = np.maximum(np.abs(min_rets), np.abs(max_rets))
        
        # Convert to Tensor (B, 1)
        max_abs_ret = torch.from_numpy(max_abs_ret).float().to(device).unsqueeze(1)
        
        # 2. Handle Temperature
        if torch.is_tensor(temperature):
            alpha = temperature.detach()
        else:
            alpha = float(temperature)
            
        # 3. Calculate Denominator (Scaling Factor)
        # Formula: (max_abs + alpha * |target_entropy|/2 * horizon) / v_max
        entropy_term = alpha * (-self.target_entropy / 2.0) * self.effective_horizon
        
        numerator = max_abs_ret + entropy_term
        scaling_factor = numerator / self.v_max
        
        # Safety clamp to avoid division by zero
        scaling_factor = torch.clamp(scaling_factor, min=1e-3)
        
        return batch_rewards / scaling_factor

    # ------------------------------------------------------------------
    # BRC-original style: per-step update + batch normalize (for Parallel_Bro)
    # ------------------------------------------------------------------

    def init_parallel(self, target_entropy):
        """Initialize state for BRC-style per-step updates.

        Must be called before using update_parallel / normalize_parallel.
        """
        self.target_entropy = target_entropy
        # Per-task reward accumulators (variable-length episodes)
        self._ep_rewards = [[] for _ in range(self.num_tasks)]

    def update_parallel(self, rewards, done):
        """BRC-original per-step normalizer update.

        Called once per parallel step with arrays of shape (num_tasks,).
        Accumulates rewards; when an episode ends (done=True),
        computes discounted returns and updates min/max stats.
        Always bootstraps (BRC original: MetaWorld terminal is always False).
        """
        for i in range(self.num_tasks):
            self._ep_rewards[i].append(float(rewards[i]))
            if done[i]:
                ep_r = np.array(self._ep_rewards[i], dtype=np.float32)
                # BRC original: always bootstrap
                bootstrap = ep_r.mean() * self.effective_horizon
                values = np.zeros_like(ep_r)
                for t in reversed(range(len(ep_r))):
                    values[t] = ep_r[t] + self.discount * bootstrap
                    bootstrap = values[t]
                self.returns_min[i] = min(self.returns_min[i], values.min())
                self.returns_max[i] = max(self.returns_max[i], values.max())
                self._ep_rewards[i] = []

    def normalize_parallel(self, rewards_np, task_ids_np, temperature):
        """BRC-original batch normalization (numpy in, numpy out).

        Args:
            rewards_np:  (batch_size,) or (num_batches, batch_size) float32
            task_ids_np: same shape, int — local task indices
            temperature: scalar float (current alpha)

        Returns:
            normalized rewards, same shape as input.
        """
        # denominator per task: max(returns_max, |returns_min|)
        denom = np.where(
            self.returns_max > np.abs(self.returns_min),
            self.returns_max,
            np.abs(self.returns_min))
        # BRC formula: (max_abs - temp * horizon * target_entropy / 2) / v_max
        denom = (denom - temperature * self.effective_horizon
                 * self.target_entropy / 2.0) / self.v_max
        # Safety clamp
        denom = np.clip(denom, a_min=1e-3, a_max=None)
        return rewards_np / denom[task_ids_np]
