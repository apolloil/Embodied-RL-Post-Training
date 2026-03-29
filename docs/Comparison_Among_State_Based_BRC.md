# 状态输入训练脚本对比

本文档对应目录 [`experiments/state/`](../experiments/state)，用于总结当前 4 个状态输入训练脚本的结构差异、训练循环差异以及默认配置。

## 共享配置

状态输入脚本的共享常量集中在 [`experiments/state/shared_config.py`](../experiments/state/shared_config.py)。

当前默认值如下。

| 项目 | 当前值 |
| --- | --- |
| `DISCOUNT` | `0.97` |
| `V_MIN / V_MAX` | `0 / 20` |
| `NUM_BINS` | `101` |
| `INIT_TEMPERATURE` | `0.1` |
| `TAU` | `0.005` |
| `WEIGHT_DECAY` | `0.0` |
| `BATCH_SIZE` | `1024` |
| `UTD` | `2` |
| `CAPACITY_PER_TASK` | `1_000_000` |
| `MAX_STEPS` | `1_000_000` |
| `MAX_EP_STEPS` | `200` |
| `WARMUP_PER_TASK` | `5000` |
| `OFF_RATIO` | `0` |
| `P_SUITE / N_SUITE` | `Expert_Goal_Observable` / `Medium_Goal_Observable` |
| `STATE_DIM / ACT_DIM` | `39 / 4` |

这意味着按当前仓库默认配置，4 个状态输入脚本都以纯在线训练为主；只有当手动把 `OFF_RATIO` 改为正数时，离线数据才会参与 batch mixing。需要注意的是，`Serial_TF.py` 仍会在启动时读取 P/N 数据来预热 normalizer。

## 四个脚本的角色定位

| 脚本 | 训练循环 | 策略结构 | 状态编码 | Replay Buffer | mask / 终止逻辑 | checkpoint 前缀 |
| --- | --- | --- | --- | --- | --- | --- |
| `Serial_Bro.py` | 串行 roll-based | BroNet | 无 | `ParallelFeatureReplayBuffer` | success-aware，`terminated` 会传入更新 | `snapshot_brc` |
| `Serial_TF.py` | 串行 roll-based | GPT actor | `StateEncoder(39->512)` | `ParallelFeatureReplayBuffer` | success-aware，`terminated` 会传入更新 | `snapshot_serial_tf` |
| `Parallel_Bro.py` | 并行 lockstep | BroNet | 无 | `BRCReplayBuffer` | `mask=1` 风格，按 BRC 约定始终 bootstrap | `snapshot_parallel_bro` |
| `Parallel_TF.py` | 并行 lockstep | GPT actor | `StateEncoder(39->512)` | `BRCReplayBuffer` | `mask=1` 风格，按 BRC 约定始终 bootstrap | `snapshot_parallel_tf` |

## 架构差异

### Bro 系列

`Serial_Bro.py` 与 `Parallel_Bro.py` 都采用 BRC 风格结构：

- `BRCTaskEmbedding`，维度为 `32`
- BroNet actor
- BroNet + C51 critic
- 不额外引入状态 encoder

### TF 系列

`Serial_TF.py` 与 `Parallel_TF.py` 使用当前仓库的 GPT actor 方案：

- `TaskEmbedding` 维度为 `512`
- `StateEncoder` 将 39 维状态编码到 512 维表示
- actor 使用 GPT backbone
- critic 仍为 distributional critic，但输入表示来自 `StateEncoder`

## 训练循环差异

### `Serial_Bro.py`

- 使用 [`core/wrapper_new.py`](../core/wrapper_new.py) 构建环境
- roll-based 串行采样
- `terminated` 直接进入 buffer 与更新过程
- 当 `OFF_RATIO=0` 时，不加载离线 buffer，也不会用离线奖励预热 normalizer

### `Serial_TF.py`

- 同样使用 [`core/wrapper_new.py`](../core/wrapper_new.py)
- 支持 action chunking，但当前默认 `num_queries=1`
- 默认仍会读取 P/N 数据并用奖励序列预热 normalizer
- 只有 `OFF_RATIO>0` 时，离线数据才会真正进入训练 batch

### `Parallel_Bro.py`

- 使用 [`core/wrapper_parallel.py`](../core/wrapper_parallel.py) 的并行环境
- 在入口处固定：
  - `cfg.env.action_repeat = 1`
  - `cfg.agent.action_chunking.num_queries = 1`
  - `cfg.agent.action_chunking.num_exec = 1`
- 使用 `BRCRewardNormalizer.init_parallel(...)`
- 在线数据的 `mask` 由并行环境生成；离线数据在混入时被显式改为全 1 mask

### `Parallel_TF.py`

- 与 `Parallel_Bro.py` 共用并行 lockstep 框架
- 仅把 actor / task embedding / state encoder 替换为 TF 版本
- 同样固定 `action_repeat=1`、`num_queries=1`、`num_exec=1`
- actor 使用 critic 更新后的 fresh task embedding

## 离线数据处理方式

| 脚本 | 当前默认 `OFF_RATIO=0` 时的行为 |
| --- | --- |
| `Serial_Bro.py` | 不加载离线 buffer，也不预热 normalizer |
| `Serial_TF.py` | 读取 P/N 数据并预热 normalizer，但不参与 batch mixing |
| `Parallel_Bro.py` | 不加载离线 buffer；normalizer 走纯在线初始化 |
| `Parallel_TF.py` | 不加载离线 buffer；normalizer 走纯在线初始化 |

当 `OFF_RATIO>0` 时：

- `Serial_Bro.py` / `Serial_TF.py` 会把在线 batch 与离线 batch 拼接后共同更新
- `Parallel_Bro.py` / `Parallel_TF.py` 会在每次 UTD 更新时采样在线批次，并可选拼接离线批次

## 使用建议

- 需要更贴近 BRC 风格结构的串行实现：使用 `Serial_Bro.py`。
- 需要状态输入 + GPT actor 的串行版本：使用 `Serial_TF.py`。
- 需要并行 lockstep 训练框架：优先使用 `Parallel_Bro.py` 或 `Parallel_TF.py`。
- 需要在相同并行框架下比较 BroNet 与 GPT actor：对照使用 `Parallel_Bro.py` 与 `Parallel_TF.py`。

## 推荐命令

```bash
python experiments/state/Serial_Bro.py IS=Serial_Bro env.suite_name=State_Based
python experiments/state/Serial_TF.py IS=Serial_TF env.suite_name=State_Based
python experiments/state/Parallel_Bro.py IS=Parallel_Bro env.suite_name=State_Based
python experiments/state/Parallel_TF.py IS=Parallel_TF env.suite_name=State_Based
```
