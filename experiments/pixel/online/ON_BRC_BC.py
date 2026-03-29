"""
ON_BRC_BC.py — RLPD-style Online Fine-tuning (BC + SAC with linear BC decay).

Uses 50% online + 50% offline (P+N) mixed batches.
Actor loss = bc_weight * BC_loss + lambda * SAC_loss, bc_weight decays to 0.
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
    make_offline_loaders, mix_online_offline, load_batch_to_device,
    run_online_training,
)


# =====================================================================
# Hyperparameters  (only script-specific values)
# =====================================================================
OFF_RATIO      = 0.5
SAC_P_RATIO    = 0.25
BC_ALPHA       = 2.5
BC_DECAY_STEPS = 100000
P_SUITE        = "Expert_Goal_Observable"
N_SUITE        = "Medium_Goal_Observable"


# =====================================================================
# Main
# =====================================================================

@hydra.main(config_path=CONFIG_DIR, config_name="config")
def train(cfg: DictConfig):
    device = torch.device(cfg.device)
    set_seed_everywhere(cfg.seed)

    train_task_names, global_to_local, _, g2l = setup_task_palette(cfg, device)
    num_tasks = len(train_task_names)

    agent, agent_ref = setup_dual_agents(cfg, device, RLAgent)

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

    print(f"=== RLPD Online Fine-tuning | {Path.cwd()} ===")
    print(f"  batch={batch_size}(off={off_bs}[P={off_p}+N={off_n}]+on={on_bs}) UTD={UTD} "
          f"chunk={chunk_size} gamma={gamma} V=[{agent.v_min},{agent.v_max}] "
          f"encoder={'unfrozen' if not agent.freeze_encoder else 'frozen'} "
          f"BC: alpha={BC_ALPHA} decay_steps={BC_DECAY_STEPS}")

    # ----- Gradient update callback (closure captures local state) -----
    def update_fn(global_step):
        bc_w = get_decay_weight(global_step, BC_DECAY_STEPS)

        on = buffer.sample_chunk(on_bs, chunk_size, gamma)
        on_imgs, on_acts, on_rews, on_next, on_dones, on_gids = on
        on_imgs = on_imgs.float().div_(255.0)
        on_next = on_next.float().div_(255.0)
        on_rews_n = normalizer.normalize(on_rews, g2l[on_gids], agent.alpha)

        if off_bs > 0:
            bp = load_batch_to_device(next(p_iter), device)
            bn = load_batch_to_device(next(n_iter), device)
            pix_m, act_m, rew_m, nxt_m, don_m, gid_m = mix_online_offline(
                (on_imgs, on_acts, on_rews_n, on_next, on_dones, on_gids),
                bp, bn, off_p, off_n, normalizer, g2l, agent.alpha)
            bc_batch = (bp[0], bp[2], None, None, None, bp[5]) if bc_w > 0 else None
        else:
            pix_m, act_m, rew_m = on_imgs, on_acts, on_rews_n
            nxt_m, don_m, gid_m = on_next, on_dones, on_gids
            bc_batch = None

        return agent.update(
            batch=(pix_m, act_m, rew_m, nxt_m, don_m, gid_m),
            bc_alpha=BC_ALPHA, use_bc_loss=(bc_w > 0),
            bc_batch=bc_batch, bc_weight=bc_w)

    run_online_training(
        agent=agent, agent_ref=agent_ref, cfg=cfg,
        train_task_names=train_task_names, g2l=g2l,
        buffer=buffer, normalizer=normalizer,
        start_training=start_training, update_every=update_every, min_buf=min_buf,
        checkpoint_prefix='snapshot_rlpost',
        update_fn=update_fn,
        done_message="RLPD fine-tuning finished.")


if __name__ == '__main__':
    train()
