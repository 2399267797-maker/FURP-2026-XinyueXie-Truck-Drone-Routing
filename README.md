# Truck-Drone Routing: P-ACO vs NSGA-II

基于 **Solomon 基准数据集**，对**卡车-无人机协同车辆路径问题**（Truck-Drone Routing）进行双目标优化（最小化运输成本 & 最小化时间窗违反惩罚），对比 **P-ACO + ALNS** 与 **NSGA-II** 两种元启发式算法的性能。

## 项目结构

```
├── src/
│   ├── experiments/
│   │   ├── PACO+ALNS/           # P-ACO + ALNS 算法实现
│   │   │   ├── PACO+ALNSW5.py   # W5 版本（基础版）
│   │   │   ├── PACO+ALNSW6.py   # W6 版本（改进版）
│   │   │   ├── PACO+ALNSW7.py   # W7 版本（最新版）
│   │   │   ├── run_paco_alns_W5.py
│   │   │   ├── run_paco_alns_W6.py
│   │   │   └── plot_result.py   # 结果可视化
│   │   ├── NSGA2/               # NSGA-II 算法实现
│   │   │   ├── nsga2_vrp.py     # 核心算法
│   │   │   ├── nsga2_imp.py     # 改进版本
│   │   │   └── models/vrp_model.py
│   │   ├── PACO_vs_NSGA2/       # 对比实验
│   │   │   ├── compare_paco_nsga2_solomon.py  # 主对比脚本
│   │   │   ├── data/            # Solomon 基准数据集
│   │   │   ├── models/vrp_model.py
│   │   │   ├── utils/
│   │   │   │   ├── visualizer.py  # 路径可视化
│   │   │   │   └── evaluator.py   # 评价指标
│   │   │   └── results/         # 实验结果与图表
│   │   ├── CVRP_POMO/           # POMO 强化学习方法（基线对比）
│   │   ├── E-VRPTW/             # 遗传算法 VRPTW（基线对比）
│   │   └── ETRD-NL/             # 混合整数规划求解器（基线对比）
│   └── README.md
└── docs/                        # 文档与周报
```

## 算法

### P-ACO + ALNS（W5 / W6 / W7）

基于种群的自适应蚁群优化（Population-based Ant Colony Optimization）结合自适应大邻域搜索（Adaptive Large Neighborhood Search），专为卡车-无人机协同场景设计：

- 多蚁群分权重引导信息素更新
- 无人机感知的启发式构造（savings-based drone heuristic）
- 多算子 ALNS 局部搜索（4-6 破坏算子 + 3-4 修复算子）
- 帕累托非支配接受准则 + 拥挤度距离裁剪
- 软时间窗约束（3 分钟最大等待容忍）

### NSGA-II

基于非支配排序的遗传算法，采用与 P-ACO 相同的目标函数和约束：

- 锦标赛选择 + 拥挤度距离
- 模拟二进制交叉（SBX）与多项式变异
- 卡车-无人机协同解码（搜索所有可能的发射/回收点）

## 实验配置

| 参数 | 取值 |
|------|------|
| 客户规模 | 25 / 50 / 100 |
| 数据集 | Solomon R / C / RC 系列 |
| 无人机续航 | 4 km（medium）/ 6 km（high）|
| 地图尺寸 | 12 × 12 km |
| 车辆配置 | 25c: 2 卡车+2 无人机; 50c: 2+2 / 4+4 / 6+6 |
| 目标函数 | (Cost, Tardiness) 双目标最小化 |
| 重复次数 | 10 runs |

## 快速开始

### 环境要求

- Python 3.10+
- 依赖：`numpy`, `matplotlib`, `scipy`, `deap`（NSGA-II）

### 运行 P-ACO + ALNS

```bash
cd src/experiments/PACO+ALNS
python run_paco_alns_W6.py --instance RC101 --n_customers 25 --endurance high
```

### 运行 NSGA-II

```bash
cd src/experiments/NSGA2
python nsga2_vrp.py
```

### 运行对比实验

```bash
cd src/experiments/PACO_vs_NSGA2
python compare_paco_nsga2_solomon.py
```

### 可视化结果

```bash
cd src/experiments/PACO+ALNS
python plot_result.py --solution 5 --pareto-only --outdir ./plots
```

## 结果

实验结果（帕累托前沿图、路径图、统计摘要）保存在各实验目录的 `results/` 文件夹中。

主要评价指标：
- **Mean Cost ± Std**：多运行平均运输成本
- **Mean Tardiness ± Std**：多运行平均时间窗违反惩罚
- **Hypervolume（HV）**：以固定参考点 (170, 140) 计算的超体积

## 许可证

本项目仅供学术研究使用。