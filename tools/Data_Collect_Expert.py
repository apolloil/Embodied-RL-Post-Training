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
import os
import yaml
from pathlib import Path
from omegaconf import DictConfig
from tqdm import tqdm

from core.agent import BCAgent
from core.wrapper_new import make
from utils.common import set_seed_everywhere
from utils.repo_paths import CONF_DIR, REPO_ROOT

def collect_data(env, agent, num_episodes, task_name, num_exec):
    """Collect trajectories, keeping only successful ones.

    Args:
        env: Environment instance.
        agent: Policy agent used for action inference.
        num_episodes: Number of successful episodes to collect per task.
        task_name: Name of the current task (for logging).
        num_exec: Number of steps to execute per action chunk.

    Returns:
        trajectories: List of trajectory dicts, each containing
            'observations', 'actions', and 'rewards'.
    """
    trajectories = []
    success_count = 0
    attempt_count = 0
    
    print(f"[{task_name}] Starting data collection, target: {num_episodes} successful episodes")
    
    while success_count < num_episodes:
        attempt_count += 1
        time_step = env.reset()
        
        # Initialize trajectory data
        episode_pixels = []  # Pixel frames for all time steps
        episode_states = []  # State vectors for all time steps
        episode_actions = []
        episode_rewards = []
        is_success = False
        
        obs = time_step.observation['pixels'] 
        task_id = time_step.observation['task_id'][0]
        
        # Save initial observation
        # Note: pixels are in CHW format; convert to HWC for dataset compatibility
        obs_hwc = obs.transpose(1, 2, 0)  # CHW -> HWC
        episode_pixels.append(obs_hwc.copy())
        episode_states.append(time_step.observation['global_state'].copy())
        
        # Action buffer queue
        action_queue = []
        done = False
        
        while not done:
            # Request a new inference only when the queue is empty
            if len(action_queue) == 0:
                with torch.no_grad():
                    # agent.act returns the full chunk: (num_queries, act_dim)
                    full_chunk = agent.act(obs, task_id, eval_mode=False)
                    
                    # Only take the first num_exec steps for execution
                    n_steps_to_exec = min(num_exec, len(full_chunk))
                    actions_to_buffer = full_chunk[:n_steps_to_exec]
                    
                    # Convert numpy array to list and enqueue
                    action_queue.extend(actions_to_buffer)
            
            # Dequeue and execute one action
            action = action_queue.pop(0)
            
            # Environment step
            time_step = env.step(action)
            
            # Update observation
            obs = time_step.observation['pixels']
            reward = time_step.reward
            done = time_step.last()
            
            terminated = time_step.observation.get('terminated', False)
            if terminated:
                is_success = True
            
            # Store data
            obs_hwc = obs.transpose(1, 2, 0)  # CHW -> HWC
            episode_pixels.append(obs_hwc.copy())
            episode_states.append(time_step.observation['global_state'].copy())
            episode_actions.append(action.copy())
            episode_rewards.append(reward)
        
        # Only keep successful trajectories
        if is_success:
            success_count += 1
            pixels_array = np.array(episode_pixels, dtype=np.uint8)    # (T+1, H, W, 3)
            states_array = np.array(episode_states, dtype=np.float32) # (T+1, 39)
            trajectory = {
                'observations': [{
                    'pixels': pixels_array,
                    'global_state': states_array,
                }],
                'actions': np.array(episode_actions, dtype=np.float32),  # (T, action_dim)
                'rewards': np.array(episode_rewards, dtype=np.float32)   # (T,)
            }
            trajectories.append(trajectory)
            print(f"[{task_name}] Attempt {attempt_count} -> Success! (collected: {success_count}/{num_episodes})")
        else:
            print(f"[{task_name}] Attempt {attempt_count} -> Failed (collected: {success_count}/{num_episodes})")
    
    print(f"[{task_name}] Done! Total attempts: {attempt_count}, successes: {success_count}")
    return trajectories


@hydra.main(config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig):
    work_dir = Path.cwd()
    print(f"Data Collection workspace: {work_dir}")

    # 1. Set random seed
    set_seed_everywhere(cfg.seed)
    
    # 2. Prepare compute device
    device = torch.device(cfg.device)

    # 3. Load Task Palette (static ID mapping)
    task_palette_path = CONF_DIR / "task_palette.yaml"
    if not task_palette_path.exists():
        raise FileNotFoundError(f"Task palette not found at {task_palette_path}. Please create it first.")

    # 4. Initialize environment list
    collect_task_names = cfg.env.MT30 
    envs = make(
        task_names=collect_task_names,
        cameras=cfg.env.cameras,
        img_size=cfg.env.img_size,
        action_repeat=cfg.env.action_repeat,
        seed=cfg.seed,
        max_episode_steps=cfg.env.max_episode_steps
    )
    
    # 5. Initialize Agent
    agent = BCAgent(device=device, cfg=cfg)

    # 6. Load model weights
    snapshot_path = Path(cfg.eval.snapshot_path)
    if not snapshot_path.exists():
        print(f"[Warning] No snapshot found at {snapshot_path}, collecting with random initialization.")
    else:
        print(f"Loading snapshot from {snapshot_path}")
        with open(snapshot_path, 'rb') as f:
            payload = torch.load(f, map_location=device)
        agent.load_snapshot(payload)

    # 7. Create save directory
    save_dir = REPO_ROOT / "datasets/Expert_Goal_Observable"
    save_dir.mkdir(exist_ok=True)
    print(f"Data will be saved to: {save_dir}")

    # 8. Start data collection loop
    print(f"Starting data collection on {len(envs)} tasks...")
    print("-" * 50)

    # Number of episodes to collect per task
    num_episodes_per_task = 100

    for i, env in enumerate(envs):
        task_name = collect_task_names[i]
        
        # Check whether data already exists for this task
        save_path = save_dir / f"{task_name}.pkl"
        if save_path.exists():
            print(f"\n[{task_name}] Data file already exists, skipping: {save_path}")
            continue
        
        print(f"\nCollecting data for task: {task_name}")
        
        # Run data collection
        trajectories = collect_data(
            env, 
            agent, 
            num_episodes_per_task, 
            task_name,
            cfg.agent.action_chunking.num_exec
        )
        
        # Organize data format (consistent with existing dataset format)
        # Format: observations (list[dict]), actions (list[np.ndarray]), rewards (list[np.ndarray])
        # observations[i] is a dict with a 'pixels' key; pixels has shape (T, H, W, 3)
        all_observations = [traj['observations'][0] for traj in trajectories]  # Each traj's observations is list[dict]; take the first
        all_actions = [traj['actions'] for traj in trajectories]  # Already numpy arrays
        all_rewards = [traj['rewards'] for traj in trajectories]  # Already numpy arrays
        
        # Build save payload
        data_to_save = {
            'observations': all_observations,
            'actions': all_actions,
            'rewards': all_rewards
        }
        
        # Save data
        with open(save_path, 'wb') as f:
            pkl.dump(data_to_save, f)
        
        print(f"Saved {len(trajectories)} episodes to {save_path}")
        print(f"  - Observations: {len(all_observations)} episodes")
        print(f"  - Actions: {len(all_actions)} episodes")
        print(f"  - Rewards: {len(all_rewards)} episodes")

    print("-" * 50)
    print(f"Data collection complete! All data saved to {save_dir}")

if __name__ == "__main__":
    main()
