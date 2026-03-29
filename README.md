# Embodied-RL-Post-Training

Embodied-RL-Post-Training 是一个面向 MetaWorld 的多任务强化学习研究代码仓库，围绕 GPT policy backbone + C51 distributional critic，提供覆盖数据、训练、评估与可视化的完整实验 pipeline，以及用于系统比较不同训练策略的多分支脚本。

仓库同时包含像素输入与状态输入两条实验线，覆盖离线训练、offline-to-online 微调、串行训练与并行训练，并将 BRC、TD3+BC、CalQL、EDAC、SO2 等方法整理到统一的配置与工具链下，便于复现实验、横向对比和扩展新方法。

## 主要内容

- 像素离线训练：`OFF_BRC_BC`、`OFF_TD3BC`、`OFF_CalQL`、`OFF_EDAC`
- 像素在线训练与微调：`ON_BRC`、`ON_BRC_BC`、`ON_CalQL`、`ON_SO2`、`ON_TD3BC`
- 状态输入训练：`Serial_Bro`、`Serial_TF`、`Parallel_Bro`、`Parallel_TF`
- 配套工具：评估、数据采集、checkpoint 可视化、批量启动脚本

## 仓库结构

```text
Embodied-RL-Post-Training/
├── conf/
│   ├── config.yaml
│   └── task_palette.yaml
├── core/
├── docs/
│   ├── Comparison_Among_Offline_Scripts.md
│   ├── Comparison_Among_Online_Scripts.md
│   └── Comparison_Among_State_Based_BRC.md
├── experiments/
│   ├── pixel/
│   │   ├── offline/
│   │   └── online/
│   └── state/
│       ├── Parallel_Bro.py
│       ├── Parallel_TF.py
│       ├── Serial_Bro.py
│       ├── Serial_TF.py
│       └── shared_config.py
├── scripts/
│   └── launch/
│       ├── calql_sweep.sh
│       ├── pixel_offline.sh
│       ├── pixel_online.sh
│       ├── plot_snapshot.sh
│       └── state_based.sh
├── tools/
│   ├── Data_Collect_Expert.py
│   ├── Data_Collect_Medium.py
│   ├── Eval.py
│   └── Plot_Snapshot.py
├── utils/
├── environment.yml
└── README.md
```

## 环境安装

仓库提供了最小可用的 Conda 环境文件，环境名为 `baku`。

```bash
conda env create -f environment.yml
conda activate baku
```

当前环境文件基于仓库实际运行环境整理，核心依赖包括：

- Python 3.9
- PyTorch 2.8.0
- Hydra 1.1.0
- MetaWorld 2.0.0
- MuJoCo 2.3.7

## 配置方式

主配置文件为 [`conf/config.yaml`](conf/config.yaml)。所有训练与工具脚本都通过 Hydra 接收命令行覆盖参数，例如：

```bash
python experiments/pixel/offline/OFF_BRC_BC.py \
  IS=OFF_BRC_BC \
  env.suite_name=Pixel_Offline \
  train.rl.batch_size=512
```

默认实验输出目录为：

```text
exp_local/${env.suite_name}/${date}/${time}_${IS}
```

## 数据组织

默认数据根目录为 `datasets/`，当前代码使用的两套离线数据目录名如下：

- `datasets/Expert_Goal_Observable/`
- `datasets/Medium_Goal_Observable/`

推荐目录结构：

```text
datasets/
├── Expert_Goal_Observable/
│   ├── basketball-v2.pkl
│   ├── bin-picking-v2.pkl
│   └── ...
└── Medium_Goal_Observable/
    ├── basketball-v2.pkl
    ├── bin-picking-v2.pkl
    └── ...
```

像素训练脚本读取 `observations[*]["pixels"]`，状态输入脚本读取 `observations[*]["global_state"]`。

## 训练

### 像素离线训练

```bash
python experiments/pixel/offline/OFF_BRC_BC.py IS=OFF_BRC_BC env.suite_name=Pixel_Offline
python experiments/pixel/offline/OFF_TD3BC.py IS=OFF_TD3BC env.suite_name=Pixel_Offline
python experiments/pixel/offline/OFF_CalQL.py IS=OFF_CalQL env.suite_name=Pixel_Offline
python experiments/pixel/offline/OFF_EDAC.py IS=OFF_EDAC env.suite_name=Pixel_Offline train.rl.critic.hidden_dim=512
```

### 像素在线训练与微调

从零开始训练：

```bash
python experiments/pixel/online/ON_BRC.py IS=ON_BRC env.suite_name=Pixel_Online
```

从已有 checkpoint 微调：

```bash
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

### 状态输入训练

```bash
python experiments/state/Serial_Bro.py IS=Serial_Bro env.suite_name=State_Based
python experiments/state/Serial_TF.py IS=Serial_TF env.suite_name=State_Based
python experiments/state/Parallel_Bro.py IS=Parallel_Bro env.suite_name=State_Based
python experiments/state/Parallel_TF.py IS=Parallel_TF env.suite_name=State_Based
```

## 启动脚本

仓库在 `scripts/launch/` 下提供了面向开源使用场景整理过的启动脚本。

```bash
bash scripts/launch/pixel_offline.sh OFF_BRC_BC
bash scripts/launch/pixel_online.sh ON_BRC
SNAPSHOT_PATH=/path/to/snapshot.pt bash scripts/launch/pixel_online.sh ON_BRC_BC
bash scripts/launch/state_based.sh Parallel_TF
CALQL_PAIRS="0.1:4.693 0.2:5.0" bash scripts/launch/calql_sweep.sh
```

这些脚本支持通过环境变量指定 GPU、日志目录、运行模式和 checkpoint 路径。可通过 `-h` 查看各脚本的完整用法。

## 工具脚本

### 评估

```bash
python tools/Eval.py eval.snapshot_path=/path/to/checkpoint.pt
```

### 数据采集

```bash
python tools/Data_Collect_Expert.py eval.snapshot_path=/path/to/checkpoint.pt
python tools/Data_Collect_Medium.py eval.snapshot_path=/path/to/checkpoint.pt
```

### checkpoint 可视化

```bash
bash scripts/launch/plot_snapshot.sh /path/to/snapshot.pt

IS_PIXEL_BASED=false \
  bash scripts/launch/plot_snapshot.sh /path/to/state_snapshot.pt state_debug
```

## 说明文档

仓库同时提供 3 份脚本对比文档，便于快速理解不同实验入口之间的实现差异：

- [`docs/Comparison_Among_Offline_Scripts.md`](docs/Comparison_Among_Offline_Scripts.md)
- [`docs/Comparison_Among_Online_Scripts.md`](docs/Comparison_Among_Online_Scripts.md)
- [`docs/Comparison_Among_State_Based_BRC.md`](docs/Comparison_Among_State_Based_BRC.md)
