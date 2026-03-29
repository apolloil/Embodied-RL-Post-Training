# 像素离线训练脚本对比

本文档对应目录 [`experiments/pixel/offline/`](../experiments/pixel/offline)，用于概览当前仓库中 4 个像素离线 RL 入口脚本的共同部分与关键差异。

## 共同基础

4 个脚本共享同一套离线训练基础设施，核心逻辑集中在 [`utils/offline_train_utils.py`](../utils/offline_train_utils.py)：

- 任务集合与全局任务 ID 映射
- P / N 双数据集加载与按比例混合
- Reward normalizer 预热
- 统一的离线训练循环、评估、checkpoint 保存与分布可视化

当前实现中的共享设置如下。

| 项目 | 当前实现 |
| --- | --- |
| 入口目录 | `experiments/pixel/offline/` |
| 共享训练循环 | `utils/offline_train_utils.py` |
| Expert 数据集 | `Expert_Goal_Observable` |
| Medium 数据集 | `Medium_Goal_Observable` |
| SAC 批次中 P:N 比例 | `0.25 : 0.75` |
| 默认 batch size | `cfg.train.rl.batch_size`，`conf/config.yaml` 当前默认值为 `256` |
| 默认折扣因子 | `cfg.train.rl.discount = 0.97` |
| C51 取值范围 | `v_min=0.0`，`v_max=20.0`，`num_bins=101` |
| 日志 / 评估 / 保存间隔 | `1000 / 25000 / 25000` |
| 默认 critic 集成数 | `2`，仅 EDAC 会改为 `10` |

## 脚本差异总览

| 脚本 | Actor 目标 | Actor 形态 | Critic 额外项 | 熵正则 | 延迟策略更新 | checkpoint 前缀 |
| --- | --- | --- | --- | --- | --- | --- |
| `OFF_BRC_BC.py` | BC + SAC | 随机策略 | 无 | 是 | 否 | `snapshot_brc` |
| `OFF_CalQL.py` | 纯 SAC | 随机策略 | CQL + Cal-QL 校准 | 是 | 否 | `snapshot_calql` |
| `OFF_EDAC.py` | 纯 SAC | 随机策略 | EDAC 多 Q 多样性正则 | 是 | 否 | `snapshot_edac` |
| `OFF_TD3BC.py` | BC + TD3 | 确定性策略 | 无 | 否 | 是，`policy_freq=2` | `snapshot_td3bc` |

## 各脚本关键参数

### `OFF_BRC_BC.py`

这是当前最直接的离线 SAC+BC 基线。

| 参数 | 当前值 |
| --- | --- |
| `BC_ALPHA` | `2.5` |
| Actor 损失 | `BC loss + lambda * SAC actor loss` |
| BC 数据来源 | 纯 Expert 批次 |

适用场景：

- 需要一个稳定、结构简单的离线基线
- 希望策略在学习过程中持续被 expert 动作约束

### `OFF_CalQL.py`

在 C51 critic 基础上加入 CQL 正则与 Cal-QL 的 MC return 校准。

| 参数 | 当前值 |
| --- | --- |
| `BC_ALPHA` | `0` |
| `CQL_N_ACTIONS` | `10` |
| `CQL_TEMP` | `1.0` |
| `CQL_CONSERVATIVE_WEIGHT` | 默认 `0.1`，可由环境变量 `CQL_WEIGHT` 覆盖 |
| `CQL_TARGET_ACTION_GAP` | 默认 `4.693`，可由环境变量 `CQL_GAP` 覆盖 |
| `CQL_ALPHA_LR` | `3e-4` |

实现要点：

- Actor 端不使用 BC 正则，保持纯 SAC 更新
- CQL 惩罚直接叠加在 critic loss 上
- 通过 `alpha_prime` 自动调节保守项强度
- 仅该脚本显式使用 MC return 参与 Cal-QL 校准

### `OFF_EDAC.py`

EDAC 版本主要改动在 critic 端：把 Q 集成数扩大到 10，并引入动作梯度多样性正则。

| 参数 | 当前值 |
| --- | --- |
| `EDAC_NUM_QS` | `10` |
| `EDAC_ETA` | `1.0` |
| Actor 损失 | 纯 SAC |
| Critic 集成数 | 强制改为 `10` |

适用场景：

- 希望增强 Q 集成的不确定性表达
- 希望在离线训练中测试更强的 critic 集成策略

### `OFF_TD3BC.py`

TD3+BC 版本保留 C51 critic，但把 actor 更新逻辑改成了 TD3 风格。

| 参数 | 当前值 |
| --- | --- |
| `BC_ALPHA` | `2.5` |
| `POLICY_NOISE` | `0.2` |
| `NOISE_CLIP` | `0.2` |
| `POLICY_FREQ` | `2` |

实现要点：

- 使用确定性策略，不进行温度学习
- critic target 中加入 target policy smoothing
- actor / target network 采用延迟更新

## 选择建议

- 优先从 `OFF_BRC_BC.py` 开始：结构最直接，适合做离线基线。
- 如果需要更强的保守性约束，使用 `OFF_CalQL.py`。
- 如果主要关注多 Q 集成与不确定性建模，使用 `OFF_EDAC.py`。
- 如果希望测试确定性策略与 TD3 风格更新，使用 `OFF_TD3BC.py`。

## 推荐命令

```bash
python experiments/pixel/offline/OFF_BRC_BC.py IS=OFF_BRC_BC env.suite_name=Pixel_Offline
python experiments/pixel/offline/OFF_TD3BC.py IS=OFF_TD3BC env.suite_name=Pixel_Offline
python experiments/pixel/offline/OFF_CalQL.py IS=OFF_CalQL env.suite_name=Pixel_Offline
python experiments/pixel/offline/OFF_EDAC.py IS=OFF_EDAC env.suite_name=Pixel_Offline train.rl.critic.hidden_dim=512
```
