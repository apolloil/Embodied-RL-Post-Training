# 像素在线训练与微调脚本对比

本文档对应目录 [`experiments/pixel/online/`](../experiments/pixel/online)，总结当前 5 个像素在线入口脚本的训练范式、数据组织方式与脚本级超参数。

## 共同基础

5 个脚本共享的在线训练基础设施位于 [`utils/online_train_utils.py`](../utils/online_train_utils.py)。

公共部分包括：

- replay buffer 与 reward normalizer 初始化
- 统一的在线 roll 循环
- 在线评估与 checkpoint 保存
- 双 agent 的 Q-distribution 可视化

当前共享设置如下。

| 项目 | 当前实现 |
| --- | --- |
| 入口目录 | `experiments/pixel/online/` |
| 共享训练循环 | `utils/online_train_utils.py` |
| 默认 batch size | `cfg.train.rl.batch_size`，当前默认值为 `256` |
| 默认折扣因子 | `cfg.train.rl.discount = 0.97` |
| 默认 UTD | `cfg.train.rl.UTD = 3`，仅 `ON_SO2.py` 强制使用 `10` |
| replay buffer 容量 | `50_000 / task` |
| 开始训练阈值 | `5000 * num_tasks` |
| 日志 / 评估 / 保存间隔 | `1000 / 5000 / 5000` |
| 可视化采样间隔 | `dist_capture_interval=5` |
| 可视化视频频率 | `dist_video_roll_freq=1` |

## 脚本差异总览

| 脚本 | 训练起点 | 数据组成 | Actor 目标 | 额外机制 | checkpoint 前缀 |
| --- | --- | --- | --- | --- | --- |
| `ON_BRC.py` | 从零开始 | 在线 + 可选离线混合 | 纯 SAC | 无 BC，无 snapshot 依赖 | `snapshot_online` |
| `ON_BRC_BC.py` | 从 snapshot 微调 | 50% 在线 + 50% P/N 离线 | BC + SAC，BC 线性衰减 | RLPD 风格微调 | `snapshot_rlpost` |
| `ON_CalQL.py` | 从 snapshot 微调 | 50% 在线 + 50% P/N 离线 | 纯 SAC | CQL 正则线性衰减 | `snapshot_calql_post` |
| `ON_SO2.py` | 从 snapshot 微调 | replay buffer 预填充 P/N，再继续在线采样 | BC + SAC，BC 线性衰减 | target policy smoothing，UTD=10 | `snapshot_so2post` |
| `ON_TD3BC.py` | 从 snapshot 微调 | 50% 在线 + 50% P/N 离线 | BC + TD3，BC 线性衰减 | 确定性策略，延迟 actor 更新 | `snapshot_td3bc_post` |

## 各脚本关键参数

### `ON_BRC.py`

这是像素在线训练的从零开始版本，不依赖预训练 checkpoint。

| 参数 | 当前值 |
| --- | --- |
| `OFF_RATIO` | `0.5` |
| `SAC_P_RATIO` | `0.25` |
| Actor 目标 | 纯 SAC |
| snapshot 依赖 | 否 |

说明：

- 虽然脚本支持离线数据混合，但不加载预训练模型
- 参考 agent `agent_ref` 仅用于可视化，对应随机初始化并冻结的网络

### `ON_BRC_BC.py`

标准的 RLPD 风格在线微调入口。

| 参数 | 当前值 |
| --- | --- |
| `OFF_RATIO` | `0.5` |
| `SAC_P_RATIO` | `0.25` |
| `BC_ALPHA` | `2.5` |
| `BC_DECAY_STEPS` | `100000` |

说明：

- batch 固定拆成 50% 在线数据、50% 离线 P/N 数据
- BC 约束会随训练步数线性衰减到 0
- 需要显式传入 `train.rl.rl_snapshot=/path/to/checkpoint.pt`

### `ON_CalQL.py`

在线版本的 Cal-QL 微调入口，在 critic 更新中只对离线分支施加 CQL 正则。

| 参数 | 当前值 |
| --- | --- |
| `OFF_RATIO` | `0.5` |
| `SAC_P_RATIO` | `0.25` |
| `CQL_N_ACTIONS` | `10` |
| `CQL_TEMP` | `1.0` |
| `CQL_CONSERVATIVE_WEIGHT` | `0.1` |
| `CQL_TARGET_ACTION_GAP` | `4.693` |
| `CQL_ALPHA_LR` | `3e-4` |
| `CQL_DECAY_STEPS` | `500000` |

说明：

- Actor 始终是纯 SAC 更新，不带 BC
- CQL 权重按步数线性衰减
- 需要显式传入初始化 checkpoint

### `ON_SO2.py`

该脚本不在每个 update 中实时混入离线 batch，而是先把离线数据写进 replay buffer，再继续在线训练。

| 参数 | 当前值 |
| --- | --- |
| `UTD` | `10` |
| `NUM_BUFFER_PREFILL` | `8` episodes / task |
| `TARGET_NOISE_SIGMA` | `0.2` |
| `TARGET_NOISE_CLIP` | `0.5` |
| `BC_ALPHA` | `2.5` |
| `BC_DECAY_STEPS` | `500000` |

说明：

- target policy smoothing 只在该脚本中显式启用
- BC batch 只从 expert loader 读取
- 需要显式传入初始化 checkpoint

### `ON_TD3BC.py`

这是 TD3+BC 的在线微调版本。

| 参数 | 当前值 |
| --- | --- |
| `OFF_RATIO` | `0.5` |
| `SAC_P_RATIO` | `0.25` |
| `BC_ALPHA` | 继承自 `OFF_TD3BC.py`，当前为 `2.5` |
| `POLICY_NOISE` | `0.2` |
| `NOISE_CLIP` | `0.2` |
| `POLICY_FREQ` | `2` |
| `BC_DECAY_STEPS` | `500000` |

说明：

- 与 `ON_BRC_BC.py` 一样使用 50% 在线 + 50% 离线混合批次
- Actor 采用确定性策略，不进行温度学习
- 需要显式传入初始化 checkpoint

## 选择建议

- 需要从零开始训练像素 agent：使用 `ON_BRC.py`。
- 需要标准的 offline-to-online 微调基线：使用 `ON_BRC_BC.py`。
- 需要在微调早期控制 Q 值过估计：使用 `ON_CalQL.py`。
- 需要更高 UTD 与 buffer 预填充策略：使用 `ON_SO2.py`。
- 需要 TD3 风格确定性微调：使用 `ON_TD3BC.py`。

## 推荐命令

```bash
python experiments/pixel/online/ON_BRC.py IS=ON_BRC env.suite_name=Pixel_Online

python experiments/pixel/online/ON_BRC_BC.py \
  IS=ON_BRC_BC \
  env.suite_name=Pixel_Online \
  train.rl.rl_snapshot=/path/to/snapshot.pt

python experiments/pixel/online/ON_CalQL.py \
  IS=ON_CalQL \
  env.suite_name=Pixel_Online \
  train.rl.rl_snapshot=/path/to/snapshot.pt

python experiments/pixel/online/ON_SO2.py \
  IS=ON_SO2 \
  env.suite_name=Pixel_Online \
  train.rl.rl_snapshot=/path/to/snapshot.pt

python experiments/pixel/online/ON_TD3BC.py \
  IS=ON_TD3BC \
  env.suite_name=Pixel_Online \
  train.rl.rl_snapshot=/path/to/snapshot.pt
```
