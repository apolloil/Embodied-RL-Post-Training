"""
ON_BRC.py — Online SAC (learn from scratch, pixel-based).

No pretrained model loading; encoder trained end-to-end from random init.
Supports optional RLPD-style offline data mixing (set OFF_RATIO > 0).
No BC loss — pure SAC actor objective throughout.
Dual-agent Q-dist visualization (Agent A: training, Agent B: frozen at random init).
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
from omegaconf import DictConfig, read_write

from core.agent import RLAgent
from utils.common import set_seed_everywhere
from utils.offline_train_utils import setup_task_palette
from utils.online_train_utils import (
    freeze_agent, setup_online_infra,
    make_offline_loaders, mix_online_offline, load_batch_to_device,
    run_online_training,
)


# =====================================================================
# Hyperparameters  (only script-specific values)
# =====================================================================
OFF_RATIO   = 0.5
SAC_P_RATIO = 0.25
P_SUITE     = "Expert_Goal_Observable"
N_SUITE     = "Medium_Goal_Observable"


# =====================================================================
# Logging
# =====================================================================

def _format_log(m, global_step, buf_size):
    return (
        f"  [S={global_step:>7d}] "
        f"cri={m.get('critic_loss',0):.3f} "
        f"act={m.get('actor_loss',0):.3f} "
        f"Qd={m.get('q_data',0):+.1f} "
        f"Qp={m.get('q_pi',0):+.1f} "
        f"a={m.get('alpha',0):.1e} "
        f"buf={buf_size}")


# =====================================================================
# Main
# =====================================================================

@hydra.main(config_path=CONFIG_DIR, config_name="config")
def train(cfg: DictConfig):
    device = torch.device(cfg.device)
    set_seed_everywhere(cfg.seed)

    train_task_names, global_to_local, _, g2l = setup_task_palette(cfg, device)
    num_tasks = len(train_task_names)

    # --- Agents (learn from scratch, no pretrained snapshot) ---
    with read_write(cfg):
        cfg.train.rl.encoder_freeze = False
    agent = RLAgent(device, cfg)
    agent.use_min_q = False
    agent.target_entropy = -agent.flat_act_dim / 2

    with read_write(cfg):
        cfg.train.rl.encoder_freeze = True
    agent_ref = RLAgent(device, cfg)
    with read_write(cfg):
        cfg.train.rl.encoder_freeze = False
    freeze_agent(agent_ref)

    # --- Online infra ---
    use_offline = OFF_RATIO > 0
    buffer, normalizer, start_training, data_root = setup_online_infra(
        cfg, device, num_tasks, global_to_local, train_task_names, agent,
        p_suite=P_SUITE if use_offline else None,
        n_suite=N_SUITE if use_offline else None)

    batch_size = cfg.train.rl.batch_size
    off_bs = int(batch_size * OFF_RATIO)
    on_bs  = batch_size - off_bs
    off_p  = int(off_bs * SAC_P_RATIO)
    off_n  = off_bs - off_p

    if use_offline:
        p_iter, n_iter = make_offline_loaders(
            data_root, cfg, train_task_names, off_bs, P_SUITE, N_SUITE)

    chunk_size   = cfg.agent.action_chunking.num_queries
    gamma        = cfg.train.rl.discount
    UTD          = cfg.train.rl.UTD
    update_every = max(1, num_tasks // UTD)
    min_buf      = on_bs + chunk_size

    print(f"=== Online SAC (learn from scratch) | {Path.cwd()} ===")
    if use_offline:
        print(f"  batch={batch_size}(off={off_bs}[P={off_p}+N={off_n}]+on={on_bs}) UTD={UTD} "
              f"chunk={chunk_size} gamma={gamma} V=[{agent.v_min},{agent.v_max}] "
              f"encoder={'unfrozen' if not agent.freeze_encoder else 'frozen'}")
    else:
        print(f"  batch={batch_size} (pure online) UTD={UTD} "
              f"chunk={chunk_size} gamma={gamma} V=[{agent.v_min},{agent.v_max}] "
              f"encoder={'unfrozen' if not agent.freeze_encoder else 'frozen'}")
    print(f"  Actor: pure SAC (no BC)")

    # ----- Gradient update callback -----
    def update_fn(global_step):
        on = buffer.sample_chunk(on_bs, chunk_size, gamma)
        on_imgs, on_acts, on_rews, on_next, on_dones, on_gids = on
        on_imgs = on_imgs.float().div_(255.0)
        on_next = on_next.float().div_(255.0)
        on_rews_n = normalizer.normalize(on_rews, g2l[on_gids], agent.alpha)

        if use_offline:
            bp = load_batch_to_device(next(p_iter), device)
            bn = load_batch_to_device(next(n_iter), device)
            pix_m, act_m, rew_m, nxt_m, don_m, gid_m = mix_online_offline(
                (on_imgs, on_acts, on_rews_n, on_next, on_dones, on_gids),
                bp, bn, off_p, off_n, normalizer, g2l, agent.alpha)
        else:
            pix_m, act_m, rew_m = on_imgs, on_acts, on_rews_n
            nxt_m, don_m, gid_m = on_next, on_dones, on_gids

        return agent.update(
            batch=(pix_m, act_m, rew_m, nxt_m, don_m, gid_m),
            bc_alpha=0, use_bc_loss=False, bc_batch=None, bc_weight=0)

    run_online_training(
        agent=agent, agent_ref=agent_ref, cfg=cfg,
        train_task_names=train_task_names, g2l=g2l,
        buffer=buffer, normalizer=normalizer,
        start_training=start_training, update_every=update_every, min_buf=min_buf,
        checkpoint_prefix='snapshot_online',
        update_fn=update_fn,
        format_log_fn=_format_log,
        done_message="Online SAC training finished.")


if __name__ == '__main__':
    train()
