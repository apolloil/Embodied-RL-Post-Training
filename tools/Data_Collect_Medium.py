"""
Data_Collect_Medium.py — Collect sub-optimal trajectory datasets.

Compared with Data_Collect_Expert.py, this script:
  1. injects Gaussian noise after a random step in the episode,
  2. retains both successful and failed trajectories,
  3. saves data under datasets/Medium_Goal_Observable/,
  4. calibrates epsilon so per-task success rates stay in a target range.
"""

import os
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"

import sys
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
CONFIG_DIR = str(REPO_ROOT / "conf")

import hydra
import torch
import numpy as np
import pickle as pkl
import random
from pathlib import Path
from omegaconf import DictConfig

from core.agent import BCAgent
from core.wrapper_new import make
from utils.common import set_seed_everywhere
from utils.repo_paths import REPO_ROOT

# =====================================================================
# User configuration
# =====================================================================
NUM_EPISODES    = 50            # Number of trajectories to collect per task
SAVE_SUITE      = "Medium_Goal_Observable"  # Save sub-directory name
CLIP_RATIO      = 0.4

# --- Dynamic calibration ---
INIT_EPSILON    = 1.0           # Starting epsilon for calibration
TARGET_LO       = 0.30          # Target success rate lower bound
TARGET_HI       = 0.60          # Target success rate upper bound
PROBE_EPISODES  = 20            # Number of episodes for calibration probing
MIN_DELTA       = 0.05          # Stop calibration when epsilon adjustment < this
MAX_DELTA       = 16            # Stop calibration when epsilon adjustment > this
MAX_CAL_ITERS   = 20            # Maximum calibration iterations


# =====================================================================
# Single episode execution
# =====================================================================

def run_one_episode(env, agent, t_max, epsilon):
    """Run a single trajectory and return the collected data.

    Args:
        env: Environment instance.
        agent: Expert agent used for action inference.
        t_max: Step index after which noise injection may begin.
        epsilon: Standard deviation of Gaussian noise added to actions.

    Returns:
        trajectory: Dict with keys 'observations', 'actions', 'rewards'.
        is_success: Whether the episode achieved the goal.
    """
    time_step = env.reset()

    episode_pixels = []
    episode_states = []
    episode_actions = []
    episode_rewards = []

    obs = time_step.observation['pixels']
    task_id = time_step.observation['task_id'][0]

    episode_pixels.append(obs.transpose(1, 2, 0).copy())
    episode_states.append(time_step.observation['global_state'].copy())

    noise_start = random.randint(1, t_max)
    step_count = 0
    done = False
    is_success = False

    while not done:
        step_count += 1

        with torch.no_grad():
            action_chunk = agent.act(obs, task_id, eval_mode=False)
            action = action_chunk[0]

        if step_count > noise_start:
            noise = np.random.normal(0.0, epsilon, size=action.shape)
            action = np.clip(action + noise, -1.0, 1.0).astype(np.float32)

        time_step = env.step(action)
        obs    = time_step.observation['pixels']
        reward = time_step.reward
        done   = time_step.last()

        terminated = time_step.observation.get('terminated', False)
        if terminated:
            is_success = True

        episode_pixels.append(obs.transpose(1, 2, 0).copy())
        episode_states.append(time_step.observation['global_state'].copy())
        episode_actions.append(action.copy())
        episode_rewards.append(reward)

    trajectory = {
        'observations': [{
            'pixels': np.array(episode_pixels, dtype=np.uint8),
            'global_state': np.array(episode_states, dtype=np.float32),
        }],
        'actions': np.array(episode_actions, dtype=np.float32),
        'rewards': np.array(episode_rewards, dtype=np.float32),
    }
    return trajectory, is_success


# =====================================================================
# Dynamic epsilon calibration
# =====================================================================

def calibrate_epsilon(env, agent, task_name, t_max):
    """Search for an epsilon that yields a success rate in [TARGET_LO, TARGET_HI].

    Uses a bisection / doubling strategy. Stops early when the adjustment
    magnitude falls below MIN_DELTA or exceeds MAX_DELTA.

    Args:
        env: Environment instance.
        agent: Expert agent.
        task_name: Name of the current task (for logging).
        t_max: Noise injection threshold step.

    Returns:
        Calibrated epsilon value.
    """
    eps = INIT_EPSILON
    lo = 0.0        # Epsilon lower bound (success rate surely > TARGET_HI)
    hi = None       # Epsilon upper bound (success rate surely < TARGET_LO); not yet established

    print(f"[{task_name}] Calibrating epsilon "
          f"(target success: {TARGET_LO:.0%}~{TARGET_HI:.0%}, "
          f"probe={PROBE_EPISODES} eps)")

    for it in range(MAX_CAL_ITERS):
        # --- Probe ---
        successes = sum(
            run_one_episode(env, agent, t_max, eps)[1]
            for _ in range(PROBE_EPISODES)
        )
        rate = successes / PROBE_EPISODES

        hi_s = f"{hi:.4f}" if hi is not None else "None"
        print(f"  [iter {it}] eps={eps:.4f}  success={rate:.0%}  "
              f"(lo={lo:.4f}, hi={hi_s})")

        # --- Decision ---
        if TARGET_LO <= rate <= TARGET_HI:
            print(f"  -> Hit target range, eps={eps:.4f}")
            return eps

        old_eps = eps

        if rate < TARGET_LO:
            # Noise too large -> decrease epsilon
            hi = eps
            eps = (lo + hi) / 2.0
        else:
            # Noise too small -> increase epsilon
            lo = eps
            if hi is not None:
                eps = (lo + hi) / 2.0
            else:
                eps = eps * 2.0     # Double

        if abs(eps - old_eps) < MIN_DELTA:
            print(f"  -> Delta {abs(eps - old_eps):.4f} < {MIN_DELTA}, "
                  f"stop at eps={eps:.4f}")
            return eps

        if abs(eps - old_eps) > MAX_DELTA:
            print(f"  -> Delta {abs(eps - old_eps):.4f} > {MAX_DELTA}, "
                  f"stop at eps={eps:.4f}")
            return eps

    print(f"  -> Max iters reached, using eps={eps:.4f}")
    return eps


# =====================================================================
# Formal collection
# =====================================================================

def collect_suboptimal(env, agent, num_episodes, task_name, t_max, epsilon):
    """Collect *num_episodes* sub-optimal trajectories (all retained).

    Args:
        env: Environment instance.
        agent: Expert agent.
        num_episodes: Number of episodes to collect.
        task_name: Task name (for logging).
        t_max: Noise injection threshold step.
        epsilon: Calibrated noise standard deviation.

    Returns:
        List of trajectory dicts.
    """
    trajectories = []
    success_count = 0

    print(f"[{task_name}] Collecting {num_episodes} episodes "
          f"(T_max={t_max}, eps={epsilon:.4f})")

    for ep_idx in range(1, num_episodes + 1):
        traj, ok = run_one_episode(env, agent, t_max, epsilon)
        trajectories.append(traj)
        if ok:
            success_count += 1

        tag = "OK" if ok else "fail"
        T = len(traj['actions'])
        print(f"  [{ep_idx}/{num_episodes}] steps={T}, {tag}  "
              f"(success: {success_count}/{ep_idx})")

    rate = success_count / max(num_episodes, 1)
    print(f"[{task_name}] Done: {success_count}/{num_episodes} "
          f"success ({rate:.0%})\n")
    return trajectories


# =====================================================================
# Main
# =====================================================================

@hydra.main(config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig):
    work_dir = Path.cwd()
    print(f"Sub-optimal data collection workspace: {work_dir}")

    set_seed_everywhere(cfg.seed)
    device = torch.device(cfg.device)

    if cfg.env.task_names == "MT30":
        collect_task_names = cfg.env.MT30
    elif cfg.env.task_names == "MT50":
        collect_task_names = cfg.env.MT50
    else:
        raise ValueError(f"Invalid task_names: {cfg.env.task_names}")

    envs = make(
        task_names=collect_task_names,
        cameras=cfg.env.cameras,
        img_size=cfg.env.img_size,
        action_repeat=cfg.env.action_repeat,
        seed=cfg.seed,
        max_episode_steps=cfg.env.max_episode_steps,
    )

    agent = BCAgent(device=device, cfg=cfg)
    snapshot_path = Path(cfg.eval.snapshot_path)
    if not snapshot_path.exists():
        print(f"[Warning] No snapshot at {snapshot_path}, using random init.")
    else:
        print(f"Loading expert snapshot: {snapshot_path}")
        payload = torch.load(str(snapshot_path), map_location=device)
        agent.load_snapshot(payload)

    save_dir = REPO_ROOT / "datasets" / SAVE_SUITE
    save_dir.mkdir(parents=True, exist_ok=True)

    action_repeat = cfg.env.action_repeat

    print(f"Tasks: {len(envs)}, episodes/task: {NUM_EPISODES}, "
          f"action_repeat: {action_repeat}")
    print(f"Calibration: init_eps={INIT_EPSILON}, "
          f"target=[{TARGET_LO:.0%}, {TARGET_HI:.0%}], "
          f"probe={PROBE_EPISODES}, min_delta={MIN_DELTA}")
    print("=" * 60)

    for i, env in enumerate(envs):
        task_name = collect_task_names[i]

        save_path = save_dir / f"{task_name}.pkl"
        if save_path.exists():
            print(f"[{task_name}] Already exists, skip: {save_path}\n")
            continue

        raw_max_steps = cfg.env.max_episode_steps[task_name]
        t_max = raw_max_steps // action_repeat
        t_max = int(t_max * CLIP_RATIO)
        if t_max < 2:
            t_max = 2

        # 1) Calibrate epsilon
        epsilon = calibrate_epsilon(env, agent, task_name, t_max)

        # 2) Formal collection
        trajectories = collect_suboptimal(
            env, agent, NUM_EPISODES, task_name, t_max, epsilon)

        # 3) Save
        all_observations = [traj['observations'][0] for traj in trajectories]
        all_actions      = [traj['actions'] for traj in trajectories]
        all_rewards      = [traj['rewards'] for traj in trajectories]

        data_to_save = {
            'observations': all_observations,
            'actions':      all_actions,
            'rewards':      all_rewards,
        }

        with open(save_path, 'wb') as f:
            pkl.dump(data_to_save, f)

        print(f"Saved {len(trajectories)} episodes (eps={epsilon:.4f}) "
              f"-> {save_path}\n")

    print("=" * 60)
    print(f"Done. All data saved to {save_dir}")


if __name__ == "__main__":
    main()
