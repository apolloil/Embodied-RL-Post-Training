"""
Parallel MetaWorld environment wrapper (state-based, no rendering).

Mirrors the BRC original `jaxrl/envs.py` ParallelEnv API:
  - All tasks step in lockstep (one parallel step = one global step)
  - Auto-reset on done (terminated or truncated)
  - Returns numpy arrays throughout

NO render / cv2 / mujoco_renderer code — parallel rendering is unsafe.
"""

import numpy as np
import yaml
import random
import metaworld
from pathlib import Path


class ParallelMetaWorldEnv:
    """State-based parallel MetaWorld environment (no rendering)."""

    def __init__(self, task_names, seed, max_episode_steps, action_repeat=1):
        """
        Args:
            task_names: list of MetaWorld task name strings.
            seed: random seed.
            max_episode_steps: dict mapping task_name -> int.
            action_repeat: number of times to repeat each action (default 2).
        """
        script_dir = Path(__file__).parent.parent
        task_palette_path = script_dir / "conf" / "task_palette.yaml"
        with open(task_palette_path, 'r') as f:
            task_map = yaml.safe_load(f)

        self.task_names = list(task_names)
        self.num_tasks = len(self.task_names)
        self.envs = []
        self.ml1s = []
        self.global_task_ids = []
        self._max_steps = []
        self._ep_steps = np.zeros(self.num_tasks, dtype=np.int64)
        self._action_repeat = action_repeat

        for idx, name in enumerate(self.task_names):
            if name not in task_map:
                raise ValueError(f"Task '{name}' not found in task_palette.yaml")
            self.global_task_ids.append(task_map[name])

            ml1 = metaworld.ML1(name)
            env = ml1.train_classes[name]()  # no render_mode
            env._partially_observable = False  # Goal-observable (aligned with BRC JAX)
            env.seed(seed + idx)
            self.envs.append(env)
            self.ml1s.append(ml1)
            self._max_steps.append(max_episode_steps[name])

        self.global_task_ids = np.array(self.global_task_ids, dtype=np.int64)
        self._max_steps_arr = np.array(self._max_steps, dtype=np.int64)

        # Determine state/action dims (MetaWorld: obs=39, act=4 for all tasks)
        self.state_dim = self.envs[0].observation_space.shape[0]
        self.act_dim = self.envs[0].action_space.shape[0]

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_single(self, idx):
        """Reset a single env with a random task goal (BRC/wrapper_new style)."""
        env = self.envs[idx]
        ml1 = self.ml1s[idx]
        task = random.choice(ml1.train_tasks)
        env.set_task(task)
        env._partially_observable = False  # set_task resets this; force goal-observable
        obs, _ = env.reset()
        self._ep_steps[idx] = 0
        return obs.astype(np.float64)

    def reset(self):
        """Reset all envs. Returns (num_tasks, state_dim)."""
        states = np.stack([self._reset_single(i) for i in range(self.num_tasks)])
        return states

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, actions):
        """Step all envs in parallel (with action repeat).

        BRC convention: MetaWorld episodes are always treated as truncated
        (never terminated), so the agent always bootstraps.  Success info
        is stored in ``self.successes`` for evaluation purposes.

        Args:
            actions: (num_tasks, act_dim) numpy array.

        Returns:
            next_obs:   (num_tasks, state_dim)
            rewards:    (num_tasks,)  — accumulated over action_repeat steps
            terminated: (num_tasks,) bool — always False (BRC convention)
            truncated:  (num_tasks,) bool — True when episode ends (any reason)
            task_ids:   (num_tasks,) int64 — global task IDs
        """
        next_obs = np.empty((self.num_tasks, self.state_dim), dtype=np.float64)
        rewards = np.zeros(self.num_tasks, dtype=np.float32)
        truncated = np.zeros(self.num_tasks, dtype=bool)
        self.successes = np.zeros(self.num_tasks, dtype=bool)

        for i, (env, action) in enumerate(zip(self.envs, actions)):
            for _ in range(self._action_repeat):
                obs, reward, term, trunc, info = env.step(action)
                self._ep_steps[i] += 1
                rewards[i] += reward

                if bool(info.get("success", False)):
                    self.successes[i] = True
                    truncated[i] = True
                    break
                if self._ep_steps[i] >= self._max_steps_arr[i]:
                    truncated[i] = True
                    break

            # Max steps reached without success
            if self._ep_steps[i] >= self._max_steps_arr[i]:
                truncated[i] = True

            next_obs[i] = obs

        # BRC convention: terminated is always False
        terminated = np.zeros(self.num_tasks, dtype=bool)
        return next_obs, rewards, terminated, truncated, self.global_task_ids

    # ------------------------------------------------------------------
    # Auto-reset where done (BRC original pattern)
    # ------------------------------------------------------------------

    def reset_where_done(self, obs, terminated, truncated):
        """Reset envs that are done, replacing their obs in-place.

        BRC original: after inserting transition into buffer, reset done envs
        so that `obs` is ready for the next step.

        Returns:
            obs: updated observations (done envs get fresh reset obs)
        """
        done = terminated | truncated
        for i in range(self.num_tasks):
            if done[i]:
                obs[i] = self._reset_single(i)
        return obs

    # ------------------------------------------------------------------
    # Mask generation (BRC convention: mask = 1.0 - terminated)
    # ------------------------------------------------------------------

    @staticmethod
    def generate_masks(terminated, truncated):
        """BRC convention: mask is always 1.0 (always bootstrap).

        terminated is always False in our wrapper, so mask = 1.0 - 0 = 1.0.
        Kept for API compatibility.

        Returns: (num_tasks,) float32
        """
        return np.ones(len(truncated), dtype=np.float32)

    # ------------------------------------------------------------------
    # Evaluation (BRC original style, no rendering)
    # ------------------------------------------------------------------

    def evaluate(self, agent, num_episodes, device):
        """Run evaluation: each task runs num_episodes, all in parallel.

        Returns:
            dict with 'goal' (per-task success rate) and 'return' (per-task mean return).
        """
        goals = np.zeros(self.num_tasks, dtype=np.float32)
        returns = np.zeros(self.num_tasks, dtype=np.float32)
        n_rollouts = np.zeros(self.num_tasks, dtype=np.int32)

        obs = self.reset()

        ep_returns = np.zeros(self.num_tasks, dtype=np.float32)
        ep_successes = np.zeros(self.num_tasks, dtype=bool)

        while np.min(n_rollouts) < num_episodes:
            # Mask: which tasks still need more episodes
            active = n_rollouts < num_episodes

            # Get actions (eval mode = deterministic)
            actions = agent.get_eval_actions(obs, self.global_task_ids, device)

            next_obs, rewards, terminated, truncated, _ = self.step(actions)
            done = terminated | truncated

            ep_returns += rewards
            ep_successes |= self.successes

            for i in range(self.num_tasks):
                if done[i] and active[i]:
                    goals[i] += float(ep_successes[i])
                    returns[i] += ep_returns[i]
                    n_rollouts[i] += 1
                    ep_returns[i] = 0.0
                    ep_successes[i] = False

            obs = next_obs
            obs = self.reset_where_done(obs, terminated, truncated)

        return {
            'goal': goals / num_episodes,
            'return': returns / num_episodes,
        }
