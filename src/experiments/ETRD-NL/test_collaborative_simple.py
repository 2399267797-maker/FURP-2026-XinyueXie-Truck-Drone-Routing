"""
简单测试 Truck-Robot 协同 ALNS 求解器
使用内置的小规模实例
展示 MILP + ALNS 混合求解策略
"""
import sys
import os

# 设置路径
script_dir = r'C:\Users\23992\FURP-2026-XinyueXie-Truck-Drone-Routing\src\experiments\ETRD-NL'
sys.path.insert(0, script_dir)
sys.path.insert(0, os.path.dirname(script_dir))

print("="*70)
print("Truck-Robot 协同 ALNS 求解器测试")
print("="*70)

# 加载测试实例
print("\n加载测试实例...")
try:
    import json
    
    # 使用 tiny 实例（7个客户）
    instance_path = os.path.join(script_dir, 'data', 'json', 'tiny_instance.json')
    
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    
    # 转换 service_times 从列表到字典
    if isinstance(instance.get('service_times'), list):
        instance['service_times'] = {i+1: t for i, t in enumerate(instance['service_times'])}
    
    # 添加空的 time_windows
    if 'time_windows' not in instance:
        instance['time_windows'] = {i: (0, float('inf')) for i in range(1, instance['n_customers'] + 1)}
    
    print(f"✓ 实例加载成功!")
    print(f"  实例名称: {instance.get('name', 'Unknown')}")
    print(f"  客户数: {instance['n_customers']}")
    print(f"  充电站数: {instance['n_charging_stations']}")

except Exception as e:
    print(f"✗ 加载失败: {e}")
    sys.exit(1)

# 测试1: MILP 求解（小规模实例适合 MILP）
print("\n" + "="*70)
print("测试1: MILP 精确求解")
print("="*70)

try:
    from or_solver.milp_model_pulp import ETRD_MILP_Solver_PULP
    
    milp_solver = ETRD_MILP_Solver_PULP(instance)
    milp_solver.build_model()
    
    print("\n开始 MILP 求解 (60秒)...")
    milp_solution = milp_solver.solve(time_limit=60)
    
    if milp_solution:
        print(f"\nMILP 结果:")
        print(f"  目标值 (Makespan): {milp_solution['makespan']:.2f}")
        print(f"  求解时间: {milp_solution['solve_time']:.2f}秒")
        print(f"  状态: {milp_solution['status']}")
    else:
        print("  MILP 未能找到可行解")
        milp_solution = None
        
except Exception as e:
    print(f"  ✗ MILP 求解失败: {e}")
    milp_solution = None

# 测试2: ALNS 求解
print("\n" + "="*70)
print("测试2: ALNS 启发式求解")
print("="*70)

try:
    from alns_solver.alns_solver_drone import ETRD_ALNS_Collaborative_Solver
    print("✓ 使用 Drone ALNS 求解器（Truck-Robot协同配送）")
    
    alns_solver = ETRD_ALNS_Collaborative_Solver(instance)
    
    # 求解（60秒）
    print("\n开始 ALNS 求解 (60秒)...")
    alns_solution = alns_solver.solve(time_limit=60)
    
    print(f"\nALNS 结果:")
    print(f"  目标值 (Makespan): {alns_solution['makespan']:.2f}")
    print(f"  访问客户数: {len(alns_solution['visited_customers'])}/{instance['n_customers']}")
    print(f"  Truck客户: {len(alns_solution['truck_customers'])}")
    print(f"  Robot客户: {len(alns_solution['robot_customers'])}")
    print(f"  交接点: 充电站 {alns_solution['handoff_point']}")
    
except Exception as e:
    print(f"✗ ALNS 求解失败: {e}")
    import traceback
    traceback.print_exc()
    alns_solution = None

# 测试3: 混合求解（MILP + ALNS）
print("\n" + "="*70)
print("测试3: MILP + ALNS 混合求解")
print("="*70)

try:
    from hybrid_solver.hybrid_solver import ETRD_Hybrid_Solver
    
    hybrid_solver = ETRD_Hybrid_Solver(instance, strategy='hybrid')
    
    print("\n开始混合求解 (120秒)...")
    hybrid_solution = hybrid_solver.solve(time_limit=120)
    
    if hybrid_solution:
        print(f"\n混合求解 结果:")
        print(f"  目标值 (Makespan): {hybrid_solution['makespan']:.2f}")
    
except Exception as e:
    print(f"✗ 混合求解失败: {e}")
    import traceback
    traceback.print_exc()

# 对比结果
print("\n" + "="*70)
print("对比结果")
print("="*70)

if milp_solution and alns_solution:
    gap = (alns_solution['makespan'] - milp_solution['makespan']) / milp_solution['makespan'] * 100
    print(f"MILP 目标值: {milp_solution['makespan']:.2f}")
    print(f"ALNS 目标值: {alns_solution['makespan']:.2f}")
    print(f"差距: {gap:.2f}%")
    
    if gap < 1:
        print("结论: ALNS 找到最优解 ✓")
    elif gap < 10:
        print("结论: ALNS 解质量优秀")
    else:
        print("结论: ALNS 解质量较差")

# 显示详细解
print("\n" + "="*70)
print("ALNS 详细解")
print("="*70)
if alns_solution:
    alns_solver.print_solution(alns_solution)

print("\n✓ 测试完成!")