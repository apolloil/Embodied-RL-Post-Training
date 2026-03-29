"""
State-based training scripts' shared hyperparameters.

This module is the single source of truth for:
  - Serial_Bro
  - Serial_TF
  - Parallel_Bro
  - Parallel_TF
"""

# Algorithm
DISCOUNT            = 0.97
V_MIN               = 0
V_MAX               = 20
NUM_BINS            = 101
INIT_TEMPERATURE    = 0.1
TAU                 = 0.005
WEIGHT_DECAY        = 0.0

# Training
BATCH_SIZE          = 1024
UTD                 = 2
CAPACITY_PER_TASK   = 1_000_000
MAX_STEPS           = 1_000_000
MAX_EP_STEPS        = 200

# Warmup: training starts after WARMUP_PER_TASK steps per task
WARMUP_PER_TASK     = 5000

# Schedule (parallel-step equivalents)
EVAL_INTERVAL       = 50000
SAVE_INTERVAL       = 50000
LOG_INTERVAL        = 1000
EVAL_EPISODES       = 10

# Offline data
OFF_RATIO           = 0
SAC_P_RATIO         = 0.25
P_SUITE             = "Expert_Goal_Observable"
N_SUITE             = "Medium_Goal_Observable"

# Optimizer
ACTOR_LR            = 3e-4
CRITIC_LR           = 3e-4
TEMP_LR             = 3e-4
TEMP_ADAM_B1        = 0.5

# Environment
STATE_DIM           = 39
ACT_DIM             = 4

# Critic architecture (shared across all four scripts)
CRITIC_WIDTH        = 4096
CRITIC_DEPTH        = 2
ENSEMBLE_SIZE       = 2
