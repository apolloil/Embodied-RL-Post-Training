import yaml
import random
import numpy as np
import pickle as pkl
from pathlib import Path
from collections import defaultdict

import torch
import torchvision.transforms as transforms
from torch.utils.data import IterableDataset, DataLoader


class BCDataset(IterableDataset):
    """
    Versatile offline dataset for both BC and RL training.

    BC mode  (discount=None, default):
        _sample returns {pixels, actions, task_id}, action chunks are zero-padded.
    RL mode  (discount=float):
        _sample returns {pixels, next_pixels, actions, rewards, dones, task_id},
        action chunks are exact (no padding), rewards are discounted over the chunk.
    """

    def __init__(
        self,
        path,
        suite,
        scenes,
        num_demos_per_task,
        chunk_size,
        discount=None,          # Enables RL mode when set
    ):
        # ---- Task Palette ----
        script_dir = Path(__file__).parent.parent
        task_palette_path = script_dir / "conf" / "task_palette.yaml"
        with open(task_palette_path, 'r') as f:
            self.task_map = yaml.safe_load(f)

        self._chunk_size = chunk_size
        self._discount = discount
        self._rl_mode = discount is not None

        # Pre-compute discount vector (RL mode)
        if self._rl_mode:
            self._discount_vec = np.power(
                discount, np.arange(chunk_size, dtype=np.float64)
            )

        # ---- Load data ----
        self._paths = [Path(path) / f"{suite}/{scene}.pkl" for scene in scenes]
        self._episodes = {}
        self._num_samples = 0

        for _path_idx, path_obj in enumerate(self._paths):
            print(f"Loading {str(path_obj)}")
            with open(str(path_obj), "rb") as f:
                data = pkl.load(f)

            observations = data["observations"]
            actions = data["actions"]
            actions = [np.clip(a, -0.99999997, 0.99999997) for a in actions]
            rewards = data.get("rewards", None)

            if self._rl_mode and rewards is None:
                raise ValueError(
                    f"RL mode requires 'rewards' key in {path_obj}"
                )

            task_name = path_obj.stem
            if task_name not in self.task_map:
                raise ValueError(f"Task '{task_name}' not found in task_palette.yaml")
            task_id = self.task_map[task_name]

            self._episodes[_path_idx] = []

            for i in range(min(num_demos_per_task, len(observations))):
                episode = dict(
                    observation=observations[i],   # dict containing "pixels"
                    action=actions[i],             # (T, action_dim)
                    task_id=task_id,
                )
                if rewards is not None:
                    episode["reward"] = rewards[i]  # (T,)

                # RL mode: pre-compute MC return with bootstrap
                # MetaWorld episodes are terminated by time-limit truncation
                # (not true terminal), so the end should always bootstrap.
                if self._rl_mode and rewards is not None:
                    traj_r = np.asarray(rewards[i], dtype=np.float32)
                    effective_horizon = 1.0 / (1.0 - discount)

                    # Infer success (first step where reward >= 20.0)
                    goal_steps = np.where(traj_r >= 20.0 - 0.1)[0]
                    episode["goal_achieved_step"] = (
                        int(goal_steps[0]) if len(goal_steps) > 0 else len(traj_r)
                    )
                    episode["episode_success"] = len(goal_steps) > 0

                    if traj_r[-1] > 20 - 0.1:
                        bootstrap = 0.0
                    else:
                        bootstrap = float(traj_r.mean()) * effective_horizon

                    mc = np.zeros_like(traj_r)
                    running = bootstrap
                    for t_idx in reversed(range(len(traj_r))):
                        running = traj_r[t_idx] + discount * running
                        mc[t_idx] = running
                    episode["mc_return"] = mc

                self._episodes[_path_idx].append(episode)
                self._num_samples += len(observations[i]["pixels"])

        # ---- Image preprocessing ----
        self.aug = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])

        self.envs_till_idx = len(self._episodes)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _sample_episode(self):
        idx = random.randint(0, self.envs_till_idx - 1)
        return random.choice(self._episodes[idx])

    def _sample(self):
        episode = self._sample_episode()
        if self._rl_mode:
            return self._sample_rl(episode)
        return self._sample_bc(episode)

    def _sample_bc(self, episode):
        """Sample a random timestep with a zero-padded action chunk (BC mode)."""
        observations = episode["observation"]
        actions = episode["action"]

        t = np.random.randint(0, len(observations["pixels"]))
        pixel = self.aug(observations["pixels"][t])

        action_dim = actions.shape[-1]
        action_chunk = np.zeros((self._chunk_size, action_dim), dtype=np.float32)
        end_idx = min(len(actions), t + self._chunk_size)
        action_chunk[:end_idx - t] = actions[t:end_idx]

        return {
            "pixels": pixel,                                        # (3, H, W)
            "actions": torch.from_numpy(action_chunk).float(),      # (chunk, act_dim)
            "task_id": torch.tensor(episode["task_id"], dtype=torch.long),
        }

    def _sample_rl(self, episode):
        """Sample an exact chunk with next_pixels, discounted reward, and done flag (RL mode)."""
        observations = episode["observation"]
        actions = episode["action"]
        rewards = episode["reward"]

        L = len(actions)
        k = self._chunk_size
        if L < k:
            return self._sample()  # Trajectory too short; resample

        t = np.random.randint(0, L - k + 1)

        pixel      = self.aug(observations["pixels"][t])
        next_pixel = self.aug(observations["pixels"][t + k])

        action_chunk = actions[t : t + k]
        chunk_reward = float(np.dot(rewards[t : t + k], self._discount_vec))
        goal_step = episode.get("goal_achieved_step", L)
        terminated = 1.0 if (t >= goal_step) else 0.0

        # MC return: G_t (return-to-go from step t, precomputed)
        mc_val = episode["mc_return"][t] if "mc_return" in episode else 0.0

        # Task completion rate (failed episodes return -1.0 sentinel)
        if episode.get("episode_success", True):
            goal_step = episode.get("goal_achieved_step", L)
            completion_rate = 1.0 if t >= goal_step else float(t) / max(goal_step, 1)
        else:
            completion_rate = -1.0

        return {
            "pixels":          pixel,                                          # (3, H, W)
            "next_pixels":     next_pixel,                                     # (3, H, W)
            "actions":         torch.from_numpy(action_chunk).float(),         # (chunk, act_dim)
            "rewards":         torch.tensor([chunk_reward], dtype=torch.float32),  # (1,)
            "mc_returns":      torch.tensor([mc_val], dtype=torch.float32),        # (1,)
            "dones":           torch.tensor([terminated], dtype=torch.float32),     # (1,)
            "task_id":         torch.tensor(episode["task_id"], dtype=torch.long),
            "completion_rate": torch.tensor([completion_rate], dtype=torch.float32),  # (1,)
        }

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def dump_rewards(self):
        """Export raw reward sequences for all tasks, used for Normalizer warm-up.

        Returns:
            dict: {global_task_id: [reward_array_1, reward_array_2, ...]}
        """
        rewards_map = defaultdict(list)
        for episodes in self._episodes.values():
            for ep in episodes:
                if "reward" in ep:
                    rewards_map[ep["task_id"]].append(ep["reward"])
        return dict(rewards_map)

    def __iter__(self):
        while True:
            yield self._sample()

    def __len__(self):
        return self._num_samples


# ======================================================================
# DataLoader Factory
# ======================================================================

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(
    path,
    suite,
    scenes,
    num_demos_per_task,
    chunk_size,
    batch_size,
    num_workers=4,
    discount=None,      # Passed through to BCDataset; defaults to BC mode
):
    dataset = BCDataset(
        path=path,
        suite=suite,
        scenes=scenes,
        num_demos_per_task=num_demos_per_task,
        chunk_size=chunk_size,
        discount=discount,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        drop_last=True,
    )

    return loader
