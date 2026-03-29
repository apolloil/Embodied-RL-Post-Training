"""
ON_SO2.py — Online Fine-tuning with Target Policy Smoothing & Buffer Pre-fill.

Key differences from ON_BRC_BC:
  1. UTD forced to 10
  2. Target policy smoothing (clipped Gaussian noise on next_action in critic target)
  3. Replay buffer pre-filled with P/N offline data (no real-time P/N mixing)
  4. BC loss with linear decay (same as BRC_BC)

Dual-agent Q-dist visualization (Agent A: training, Agent B: frozen reference).
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
from pathlib import Path
from omegaconf import DictConfig

from core.agent import RLAgent
from utils.common import set_seed_everywhere
from utils.offline_train_utils import setup_task_palette
from utils.online_train_utils import (
    get_decay_weight, setup_dual_agents, setup_online_infra,
    make_single_loader, load_batch_to_device,
    prefill_buffer, load_full_task_map,
    run_online_training,
)


# =====================================================================
# Hyperparameters  (only script-specific values)
# =====================================================================
UTD                = 10
NUM_BUFFER_PREFILL = 8
TARGET_NOISE_SIGMA = 0.2
TARGET_NOISE_CLIP  = 0.5
BC_ALPHA           = 2.5
BC_DECAY_STEPS     = 500_000
P_SUITE            = "Expert_Goal_Observable"
N_SUITE            = "Medium_Goal_Observable"


# =====================================================================
# Main
# =====================================================================

@hydra.main(config_path=CONFIG_DIR, config_name="config")
def train(cfg: DictConfig):
    device = torch.device(cfg.device)
    set_seed_everywhere(cfg.seed)

    train_task_names, global_to_local, _, g2l = setup_task_palette(cfg, device)
    num_tasks = len(train_task_names)
    full_task_map = load_full_task_map(cfg)

    agent, agent_ref = setup_dual_agents(cfg, device, RLAgent)
    agent.target_noise_sigma = TARGET_NOISE_SIGMA
    agent.target_noise_clip  = TARGET_NOISE_CLIP

    buffer, normalizer, start_training, data_root = setup_online_infra(
        cfg, device, num_tasks, global_to_local, train_task_names, agent,
        p_suite=P_SUITE, n_suite=N_SUITE)

    print("Pre-filling replay buffer with P + N offline data...")
    prefill_buffer(buffer, data_root, train_task_names, full_task_map,
                   suite_p=P_SUITE, suite_n=N_SUITE,
                   num_episodes=NUM_BUFFER_PREFILL)

    batch_size = cfg.train.rl.batch_size
    p_iter = make_single_loader(data_root, cfg, train_task_names, batch_size, P_SUITE)

    chunk_size   = cfg.agent.action_chunking.num_queries
    gamma        = cfg.train.rl.discount
    update_every = max(1, num_tasks // UTD)
    min_buf      = batch_size + chunk_size

    print(f"=== SO2 Online Fine-tuning | {Path.cwd()} ===")
    print(f"  batch={batch_size} UTD={UTD} chunk={chunk_size} gamma={gamma} "
          f"V=[{agent.v_min},{agent.v_max}]")
    print(f"  target_noise: sigma={TARGET_NOISE_SIGMA} clip={TARGET_NOISE_CLIP}")
    print(f"  buffer_prefill: {NUM_BUFFER_PREFILL} eps/task (P+N) -> buf={buffer.size}")
    print(f"  encoder={'unfrozen' if not agent.freeze_encoder else 'frozen'} "
          f"BC: alpha={BC_ALPHA} decay_steps={BC_DECAY_STEPS}")

    # ----- Gradient update callback -----
    def update_fn(global_step):
        bc_w = get_decay_weight(global_step, BC_DECAY_STEPS)

        b = buffer.sample_chunk(batch_size, chunk_size, gamma)
        b_imgs, b_acts, b_rews, b_next, b_dones, b_gids = b
        b_imgs = b_imgs.float().div_(255.0)
        b_next = b_next.float().div_(255.0)
        b_rews_n = normalizer.normalize(b_rews, g2l[b_gids], agent.alpha)

        bc_batch = None
        if bc_w > 0:
            bp = load_batch_to_device(next(p_iter), device)
            bc_batch = (bp[0], bp[2], None, None, None, bp[5])

        return agent.update(
            batch=(b_imgs, b_acts, b_rews_n, b_next, b_dones, b_gids),
            bc_alpha=BC_ALPHA, use_bc_loss=(bc_w > 0),
            bc_batch=bc_batch, bc_weight=bc_w)

    run_online_training(
        agent=agent, agent_ref=agent_ref, cfg=cfg,
        train_task_names=train_task_names, g2l=g2l,
        buffer=buffer, normalizer=normalizer,
        start_training=start_training, update_every=update_every, min_buf=min_buf,
        checkpoint_prefix='snapshot_so2post',
        update_fn=update_fn,
        done_message="SO2 online fine-tuning finished.")


if __name__ == '__main__':
    train()
