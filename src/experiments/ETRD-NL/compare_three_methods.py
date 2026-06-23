"""
ETRD-NL Comparison Framework
在同一实例上比较：CVRP-POMO, E-VRPTW, ETRD-NL
"""
import sys
import os

# 设置路径
script_dir = r'C:\Users\23992\FURP-2026-XinyueXie-Truck-Drone-Routing\src\experiments\ETRD-NL'
sys.path.insert(0, script_dir)
sys.path.insert(0, os.path.dirname(script_dir))

print("="*70)
print("E-VRPTW / CVRP-POMO / ETRD-NL 三方法对比框架")
print("="*70)

# =============================================================================
# 1. 加载E-VRPTW实例（R101格式）
# =============================================================================
print("\n1. 加载E-VRPTW实例...")

from data.converter import load_evrptw_instance, ETRDInstanceConverter

# 实例名称（对应Solomon标准算例）
INSTANCE_NAME = 'R101'

# 转换为ETRD-NL格式
instance = load_evrptw_instance(INSTANCE_NAME)

print(f"   实例: {INSTANCE_NAME}")
print(f"   客户数: {instance['n_customers']}")
print(f"   仓库: ({instance['depot'][0]:.1f}, {instance['depot'][1]:.1f})")
print(f"   充电站数: {instance['n_charging_stations']}")
print(f"   车辆容量: {instance['vehicle_capacity']}")
print("   ✓ 实例加载成功!")

# =============================================================================
# 2. 运行ETRD-NL求解器
# =============================================================================
print("\n2. 运行ETRD-NL求解器...")

from hybrid_solver.hybrid_solver import ETRD_Hybrid_Solver

# 创建求解器（使用混合策略）
solver = ETRD_Hybrid_Solver(instance, strategy='auto')

# 求解（时间限制根据规模调整）
time_limit = 300 if instance['n_customers'] <= 30 else 600
etrdnl_solution = solver.solve(time_limit=time_limit)

if etrdnl_solution:
    print(f"   ✓ ETRD-NL求解成功!")
    print(f"   目标值 (Makespan): {etrdnl_solution['makespan']:.2f}")
    print(f"   方法: {solver.stats['method_used']}")
    print(f"   求解时间: {solver.stats['solve_time']:.2f}秒")
else:
    print("   ✗ ETRD-NL求解失败")

# =============================================================================
# 3. 显示ETRD-NL求解结果
# =============================================================================
print("\n3. ETRD-NL求解结果:")
solver.print_solution()

# =============================================================================
# 4. 说明其他两种方法的对比方式
# =============================================================================
print("\n" + "="*70)
print("CVRP-POMO 和 E-VRPTW 对比说明")
print("="*70)

print(f"""
实例: {INSTANCE_NAME} (Solomon标准算例)
ETRD-NL转换格式: ETRD-NL/data/json/{INSTANCE_NAME}_etrdnl.json

对比方式:
----------

1. E-VRPTW (强化学习方法):
   - 直接使用E-VRPTW/data/json/{INSTANCE_NAME}.json
   - 包含时间窗(TW)和电池能量(E)约束
   - 运行: cd E-VRPTW && python sample_R101.py

2. CVRP-POMO (强化学习方法):
   - 直接使用E-VRPTW/data/json/{INSTANCE_NAME}.json（与E-VRPTW相同实例）
   - 仅包含容量约束(C)，无时间窗
   - 运行: cd CVRP_POMO/CVRP/POMO && python test_n100.py

3. ETRD-NL (MILP+ALNS混合方法):
   - 使用转换后的ETRD-NL格式实例
   - 包含时间窗、电池能量、非线性充电约束
   - 本脚本已运行

目标函数对比:
--------------
- E-VRPTW: 最小化总行驶距离（可能含时间窗惩罚）
- CVRP-POMO: 最小化总行驶距离
- ETRD-NL: 最小化完工时间(Makespan)

约束对比:
----------
- CVRP: 容量约束(C)
- E-VRPTW: 容量约束 + 电池能量(E) + 时间窗(TW)
- ETRD-NL: 容量约束 + 电池能量(E) + 时间窗(TW) + 非线性充电
""")

print("="*70)
print("对比框架设置完成!")
print("="*70)