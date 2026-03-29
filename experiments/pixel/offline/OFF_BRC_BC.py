"""
Offline SAC+BC Training with Q-Distribution Visualization.

Actor_loss = MSE_bc(mu(s), a_data) + lambda * SAC_actor_loss
lambda = BC_ALPHA / mean(|Q(s, pi(s))|)
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
from omegaconf import DictConfig

from core.agent import RLAgent
from utils.common import set_seed_everywhere
from utils.offline_train_utils import (
    setup_task_palette, setup_offline_infra, run_offline_training,
)

BC_ALPHA = 2.5


@hydra.main(config_path=CONFIG_DIR, config_name="config")
def train(cfg: DictConfig):
    device = torch.device(cfg.device)
    set_seed_everywhere(cfg.seed)

    train_task_names, global_to_local, gid_to_name, g2l_tensor = \
        setup_task_palette(cfg, device)

    agent = RLAgent(device, cfg)

    normalizer, p_iter, n_iter, B_p, B_n, batch_size = \
        setup_offline_infra(cfg, device, train_task_names, global_to_local, agent)

    print(f"=== Offline SAC+BC | BC_ALPHA={BC_ALPHA} | batch={batch_size}(P={B_p},N={B_n}) ===\n")

    def update_fn(sac, bc_batch, rewards_norm, mc_returns_norm, capture, idx):
        return agent.update(
            batch=(sac['pixels'], sac['flat_actions'], rewards_norm,
                   sac['next_pixels'], sac['dones'], sac['task_ids']),
            bc_alpha=BC_ALPHA,
            return_dists=capture, dist_sample_idx=idx,
            bc_batch=bc_batch)

    run_offline_training(
        agent=agent, cfg=cfg,
        train_task_names=train_task_names, g2l_tensor=g2l_tensor,
        gid_to_name=gid_to_name,
        normalizer=normalizer, p_iter=p_iter, n_iter=n_iter,
        B_p=B_p, B_n=B_n, batch_size=batch_size,
        checkpoint_prefix='snapshot_brc',
        update_fn=update_fn,
        done_message="Offline SAC+BC Training finished.")


if __name__ == '__main__':
    train()
