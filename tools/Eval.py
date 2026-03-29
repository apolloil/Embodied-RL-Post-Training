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
import os
import yaml
from pathlib import Path
from omegaconf import DictConfig

from core.agent import BCAgent
from core.wrapper_new import make
from utils.logger import Logger
from utils.common import set_seed_everywhere, VideoRecorder
from utils.repo_paths import CONF_DIR

def evaluate(env, agent, num_episodes, video_recorder, step, task_name, num_exec):
    episode_rewards = []
    success_count = 0

    video_recorder.init(enabled=True)

    for i in range(num_episodes):
        time_step = env.reset()
        
        episode_reward = 0
        done = False
        is_success = False
        
        obs = time_step.observation['pixels'] 
        task_id = time_step.observation['task_id'][0] 
        
        # [Key logic] Action buffer queue
        action_queue = []
        
        while not done:
            # Only request a new inference when the queue is empty
            if len(action_queue) == 0:
                with torch.no_grad():
                    # agent.act returns the full chunk: (num_queries, act_dim)
                    full_chunk = agent.act(obs, task_id, eval_mode=True)
                    
                    # [Key logic] Only take the first num_exec steps for execution
                    # Note: prevent num_exec from exceeding the actual chunk length
                    n_steps_to_exec = min(num_exec, len(full_chunk))
                    actions_to_buffer = full_chunk[:n_steps_to_exec]
                    
                    # Convert numpy array to list and enqueue
                    action_queue.extend(actions_to_buffer)
            
            # Dequeue and execute one action
            action = action_queue.pop(0)
            
            # Environment step
            time_step = env.step(action)
            
            # Update observation (must be refreshed even when the queue is
            # not empty, so the next inference uses the latest state)
            obs = time_step.observation['pixels']
            reward = time_step.reward
            done = time_step.last()
            
            if time_step.observation['goal_achieved'] > 0.5:
                is_success = True
            
            episode_reward += reward
            video_recorder.record(env)
            
            # If the environment terminates early (done), remaining actions
            # in the queue are automatically discarded; the next episode
            # starts cleanly.

        episode_rewards.append(episode_reward)
        if is_success:
            success_count += 1

    video_recorder.save(f'{task_name}_step{step}.mp4')
    return np.mean(episode_rewards), success_count / num_episodes


@hydra.main(config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig):
    work_dir = Path.cwd()
    print(f"Evaluation workspace: {work_dir}")

    # 1. Set random seed
    set_seed_everywhere(cfg.seed)
    
    # 2. Prepare compute device
    device = torch.device(cfg.device)

    # 3. Load Task Palette (static ID mapping)
    task_palette_path = CONF_DIR / "task_palette.yaml"
    if not task_palette_path.exists():
        raise FileNotFoundError(f"Task palette not found at {task_palette_path}. Please create it first.")

    # 4. Initialize environment list
    eval_task_names = cfg.env.MT30 
    envs = make(
        task_names=eval_task_names,
        cameras=cfg.env.cameras,
        img_size=cfg.env.img_size,
        action_repeat=cfg.env.action_repeat,
        seed=cfg.seed,
        max_episode_steps=cfg.env.max_episode_steps
    )
    
    # 5. Initialize utilities
    logger = Logger(str(work_dir / "eval_logs"), use_tb=cfg.train.use_tb)
    video_recorder = VideoRecorder(str(work_dir / "eval_videos"))

    # 6. Initialize Agent
    agent = BCAgent(device=device, cfg=cfg)

    # 7. Load model weights
    snapshot_path_cfg = cfg.eval.snapshot_path
    if not snapshot_path_cfg:
        print("[Warning] eval.snapshot_path is not set, evaluating with random initialization.")
    else:
        snapshot_path = Path(snapshot_path_cfg).expanduser()
        if not snapshot_path.exists():
            print(f"[Warning] No snapshot found at {snapshot_path}, evaluating with random initialization.")
        else:
            print(f"Loading snapshot from {snapshot_path}")
            with open(snapshot_path, 'rb') as f:
                payload = torch.load(f, map_location=device)
            agent.load_snapshot(payload)

    # 8. Start multi-task evaluation loop
    global_step = 0 
    total_success_rate = 0
    avg_rewards = {}

    print(f"Starting evaluation on {len(envs)} tasks...")
    print("-" * 50)

    for i, env in enumerate(envs):
        task_name = eval_task_names[i]
        
        # Run evaluation
        avg_reward, success_rate = evaluate(
            env, 
            agent, 
            cfg.eval.num_eval_episodes, 
            video_recorder, 
            global_step,
            task_name,
            cfg.agent.action_chunking.num_exec
        )
        
        # Record results
        avg_rewards[task_name] = avg_reward
        total_success_rate += success_rate
        
        # Write to log
        logger.log(f'eval/{task_name}_reward', avg_reward, global_step)
        logger.log(f'eval/{task_name}_success', success_rate, global_step)
        
        print(f"Task: {task_name:<30} | Reward: {avg_reward:6.2f} | Success: {success_rate:.2%}")

    print("-" * 50)
    
    # Record global average metrics
    mean_success_rate = total_success_rate / len(envs)
    logger.log('eval/mean_success_rate', mean_success_rate, global_step)
    
    # Flush logs and close
    logger.dump(global_step, Ty='eval')
    logger.close()
    
    print(f"Evaluation Complete. Mean Success Rate: {mean_success_rate:.2%}")

if __name__ == "__main__":
    main()
