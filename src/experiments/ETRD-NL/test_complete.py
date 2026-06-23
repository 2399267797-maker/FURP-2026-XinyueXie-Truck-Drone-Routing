"""
ETRD-NL Complete Test
完整测试ETRD-NL项目
"""
import sys
import os

# 设置路径
script_dir = r'C:\Users\23992\FURP-2026-XinyueXie-Truck-Drone-Routing\src\experiments\ETRD-NL'
sys.path.insert(0, script_dir)

print("="*60, flush=True)
print("ETRD-NL Complete Test", flush=True)
print("="*60, flush=True)

# 1. 导入实例生成器
print("\n1. 导入实例生成器...", flush=True)
from data.instance_generator import ETRDInstanceGenerator
print("   ✓ ETRDInstanceGenerator导入成功", flush=True)

# 2. 生成tiny实例
print("\n2. 生成tiny实例...", flush=True)
gen = ETRDInstanceGenerator(seed=42)
instance = gen.generate_tiny_instance()
print(f"   ✓ 客户数: {instance['n_customers']}", flush=True)
print(f"   ✓ 充电站数: {instance['n_charging_stations']}", flush=True)
print(f"   ✓ 卡车电池容量: {instance['truck']['battery_capacity']}", flush=True)

# 3. 导入PuLP
print("\n3. 导入PuLP...", flush=True)
import pulp
print(f"   ✓ PuLP版本: {pulp.__version__}", flush=True)

# 4. 创建简单的MILP模型
print("\n4. 创建简单的MILP模型...", flush=True)
prob = pulp.LpProblem("Test", pulp.LpMinimize)
x = pulp.LpVariable("x", lowBound=0)
y = pulp.LpVariable("y", lowBound=0)
prob += x + y
prob += 2*x + y >= 5
prob += x + 2*y >= 5
print("   ✓ 简单模型创建成功", flush=True)

# 5. 求解简单模型
print("\n5. 求解简单模型...", flush=True)
status = prob.solve()
print(f"   ✓ 状态: {pulp.LpStatus[status]}", flush=True)
print(f"   ✓ x = {pulp.value(x):.2f}", flush=True)
print(f"   ✓ y = {pulp.value(y):.2f}", flush=True)

print("\n" + "="*60, flush=True)
print("基础测试完成！", flush=True)
print("="*60, flush=True)

print("\n接下来测试MILP求解器...", flush=True)

# 6. 尝试导入MILP求解器
print("\n6. 导入MILP求解器...", flush=True)
try:
    # 直接导入模块，避免__init__.py的问题
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "milp_model_pulp",
        os.path.join(script_dir, "or_solver", "milp_model_pulp.py")
    )
    milp_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(milp_module)
    ETRD_MILP_Solver_PULP = milp_module.ETRD_MILP_Solver_PULP
    print("   ✓ ETRD_MILP_Solver_PULP导入成功", flush=True)
    
    # 7. 创建求解器
    print("\n7. 创建MILP求解器...", flush=True)
    solver = ETRD_MILP_Solver_PULP(instance)
    print("   ✓ 求解器创建成功", flush=True)
    
    # 8. 构建模型
    print("\n8. 构建MILP模型...", flush=True)
    solver.build_model()
    print(f"   ✓ 变量数: {len(solver.model.variables())}", flush=True)
    print(f"   ✓ 约束数: {len(solver.model.constraints)}", flush=True)
    
    # 9. 求解模型
    print("\n9. 求解MILP模型（时间限制: 60秒）...", flush=True)
    result = solver.solve(time_limit=60)
    
    if result:
        print("\n" + "="*60, flush=True)
        print("✓ 求解成功！", flush=True)
        print("="*60, flush=True)
        print(f"完工时间: {result['makespan']:.2f}", flush=True)
        print(f"求解时间: {result['solve_time']:.2f}秒", flush=True)
        print(f"状态: {result['status']}", flush=True)
        
        print("\n卡车路径:", flush=True)
        for i, (from_node, to_node) in enumerate(result['truck_route']):
            print(f"  {i+1}. 节点 {from_node} -> 节点 {to_node}", flush=True)
        
        print("\n" + "="*60, flush=True)
        print("测试完成！", flush=True)
        print("="*60, flush=True)
    else:
        print("\n✗ 未找到可行解", flush=True)
        
except Exception as e:
    print(f"\n✗ 错误: {str(e)}", flush=True)
    import traceback
    traceback.print_exc()