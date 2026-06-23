# ETRD-NL: Electric Truck and Robot Delivery with Non-Linear Charging

电动卡车+地面机器人配送问题，具有非线性充电特性，采用OR+ALNS混合求解框架。

## 项目结构

```
ETRD-NL/
├── data/                          # 数据文件
│   ├── instance_generator.py      # 算例生成器
│   ├── converter.py               # 实例格式转换器
│   └── json/                      # 测试实例
│       ├── tiny_instance.json     # tiny算例（7客户）
│       ├── small_instance.json    # small算例（15客户）
│       ├── medium_instance.json   # medium算例（30客户）
│       └── large_instance.json    # large算例（60客户）
├── or_solver/                     # OR求解器（MILP）
│   └── milp_model_pulp.py        # PuLP MILP模型
├── alns_solver/                   # ALNS求解器
│   └── alns_solver.py             # 自适应大邻域搜索
├── hybrid_solver/                  # 混合求解框架
│   └── hybrid_solver.py           # MILP + ALNS混合策略
├── test_complete.py               # 完整测试脚本
├── test_hybrid.py                 # 混合求解器测试
├── compare_three_methods.py       # 三方法对比脚本
└── README.md                      # 说明文档
```

## 问题特性

### 核心约束

1. **路径约束**：卡车和机器人的路径规划
2. **时间约束**：到达时间、服务时间、充电时间
3. **能量约束**：电池容量、行驶能耗
4. **非线性充电**：分段线性化的充电曲线
5. **同步约束**：卡车-机器人会合点的时间同步

### 车辆参数

| 参数 | 卡车 (T) | 机器人 (R) |
|------|---------|-----------|
| 行驶速度 | 25 km/h | 15 km/h |
| 单位能耗 | 0.2 kWh/km | 0.1 kWh/km |
| 电池容量 | 60-120 kWh | 3-6 kWh |

### 非线性充电

充电曲线分为4段，每段具有不同的充电功率：

**卡车充电分段：**
- 0-30% SOC: 50 kW
- 30-60% SOC: 40 kW
- 60-80% SOC: 30 kW
- 80-100% SOC: 20 kW

**机器人充电分段：**
- 0-40% SOC: 10 kW
- 40-70% SOC: 8 kW
- 70-90% SOC: 5 kW
- 90-100% SOC: 3 kW

## 使用方法

### 1. 安装依赖

```bash
pip install pulp numpy
```

### 2. 运行测试

```bash
cd src/experiments/ETRD-NL

# 运行完整测试
python test_complete.py

# 运行混合求解器测试
python test_hybrid.py
```

### 3. 预期输出

```
======================================================================
ETRD-NL 混合求解器
======================================================================
问题规模: 15 个客户
选择方法: milp
时间限制: 120秒
======================================================================

使用MILP精确求解器...
MILP求解完成!
目标值: 45.23
求解时间: 12.34秒
状态: optimal

======================================================================
SOLUTION SUMMARY
======================================================================
方法: milp
求解时间: 12.34秒
目标值 (Makespan): 45.23

Truck Route:
  1. 节点 0 -> 节点 3
  2. 节点 3 -> 节点 5
  3. 节点 5 -> 节点 8
  ...

======================================================================
```

## 求解策略

### 自动选择策略

- **小规模问题（≤30客户）**：使用MILP精确求解器
- **大规模问题（>30客户）**：使用ALNS启发式求解器
- **混合策略**：先用MILP快速求解，再用ALNS改进

### ALNS算子

**破坏算子（Destroy）：**
- `random`：随机移除客户
- `worst`：移除造成最大绕路的客户
- `cluster`：移除相邻的客户
- `distance`：移除距离仓库最远的客户

**修复算子（Repair）：**
- `greedy`：贪心插入最佳位置
- `regret`：考虑未来成本的后悔插入
- `random_insert`：随机插入

### 混合求解框架

```python
from hybrid_solver.hybrid_solver import ETRD_Hybrid_Solver, solve_etrn_nl

# 方法1：使用便捷函数
solution = solve_etrn_nl(instance, strategy='auto', time_limit=300)

# 方法2：使用求解器类
solver = ETRD_Hybrid_Solver(instance, strategy='hybrid')
solution = solver.solve(time_limit=300)
solver.print_solution()
solver.save_solution('solution.json')
```

## 三方法对比框架

支持与E-VRPTW、CVRP-POMO使用相同的Solomon标准算例进行方法对比。

### 使用方法

```bash
# 运行三方法对比（使用R101实例）
python compare_three_methods.py
```

### 对比方法

| 方法 | 问题 | 约束 | 求解器 |
|------|------|------|--------|
| CVRP-POMO | CVRP | 容量(C) | 强化学习(POMO) |
| E-VRPTW | VRPTW | 容量+时间窗(TW) | 强化学习(RL) |
| ETRD-NL | ETRD-NL | 容量+时间窗+电池+非线性充电 | MILP+ALNS |

### 实例转换

将E-VRPTW格式转换为ETRD-NL格式：

```python
from data.converter import load_evrptw_instance

# 加载并转换E-VRPTW实例（如R101）到ETRD-NL格式
instance = load_evrptw_instance('R101')

# 保存转换后的实例
from data.converter import ETRDInstanceConverter
converter = ETRDInstanceConverter()
converter.save_instance(instance, 'R101_etrdnl.json')
```

## 求解性能

| 规模 | 客户数 | 推荐方法 | 预期时间 | 预期质量 |
|------|--------|---------|---------|---------|
| Tiny | 7 | MILP | < 1分钟 | 最优解 |
| Small | 15 | MILP | < 5分钟 | 最优/近最优 |
| Medium | 30 | ALNS/MILP | < 10分钟 | 近最优 |
| Large | 60 | ALNS | < 5分钟 | 启发式解 |

## 项目说明

### 核心模块

1. **data/instance_generator.py**：生成不同规模的测试实例
2. **or_solver/milp_model_pulp.py**：基于PuLP的MILP精确模型
3. **alns_solver/alns_solver.py**：自适应大邻域搜索启发式算法
4. **hybrid_solver/hybrid_solver.py**：MILP + ALNS混合求解框架

### 下一步计划

1. ✅ 实现ALNS求解器
2. ✅ 创建混合求解框架
3. ⏳ 测试和验证求解器性能
4. ⏳ 优化ALNS算子和参数调优
5. ⏳ 添加结果可视化功能

## 参考文献

基于论文复现：Electric Truck and Robot Delivery with Non-Linear Charging