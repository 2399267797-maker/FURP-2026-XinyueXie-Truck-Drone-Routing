"""
ETRD-NL Hybrid Solver
混合求解框架：MILP + ALNS
小规模用MILP精确求解，大规模用ALNS启发式求解
"""
import numpy as np
import time
from typing import Dict, Optional
import os

# 尝试导入求解器
try:
    from or_solver.milp_model_pulp import ETRD_MILP_Solver_PULP
    MILP_AVAILABLE = True
except ImportError:
    MILP_AVAILABLE = False
    print("Warning: PuLP not available, ALNS only mode")

try:
    from alns_solver.alns_solver_drone import ETRD_ALNS_Collaborative_Solver
    ALNS_AVAILABLE = True
except ImportError:
    ALNS_AVAILABLE = False
    print("Warning: ALNS solver not available")


class ETRD_Hybrid_Solver:
    """ETRD-NL混合求解器"""
    
    def __init__(self, instance: Dict, strategy: str = 'auto'):
        """
        初始化混合求解器
        
        Args:
            instance: 问题实例
            strategy: 求解策略
                - 'milp': 强制使用MILP
                - 'alns': 强制使用ALNS
                - 'hybrid': 混合策略（根据问题规模自动选择）
                - 'auto': 自动选择（默认）
        """
        self.instance = instance
        self.strategy = strategy
        
        # 问题规模
        self.n_customers = instance['n_customers']
        
        # 求解器选择阈值
        self.milp_threshold = 30  # 30个客户以下用MILP
        
        # 求解器
        self.milp_solver = None
        self.alns_solver = None
        
        # 解
        self.solution = None
        
        # 统计
        self.stats = {
            'method_used': None,
            'solve_time': 0,
            'solution_quality': None
        }
    
    def solve(self, time_limit: int = 300) -> Dict:
        """
        求解ETRD-NL问题
        
        Args:
            time_limit: 时间限制（秒）
        
        Returns:
            最优解
        """
        start_time = time.time()
        
        # 选择求解方法
        method = self._select_method()
        self.stats['method_used'] = method
        
        print(f"\n{'='*60}")
        print(f"ETRD-NL 混合求解器")
        print(f"{'='*60}")
        print(f"问题规模: {self.n_customers} 个客户")
        print(f"选择方法: {method}")
        print(f"时间限制: {time_limit}秒")
        print(f"{'='*60}\n")
        
        if method == 'milp':
            self.solution = self._solve_milp(time_limit)
        elif method == 'alns':
            self.solution = self._solve_alns(time_limit)
        elif method == 'hybrid':
            self.solution = self._solve_hybrid(time_limit)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.stats['solve_time'] = time.time() - start_time
        self.stats['solution_quality'] = self.solution.get('makespan', float('inf'))
        
        return self.solution
    
    def _select_method(self) -> str:
        """选择求解方法"""
        if self.strategy == 'milp':
            return 'milp'
        elif self.strategy == 'alns':
            return 'alns'
        elif self.strategy == 'hybrid':
            # 根据问题规模选择
            if self.n_customers <= self.milp_threshold:
                return 'milp'
            else:
                return 'alns'
        elif self.strategy == 'auto':
            # 自动选择：优先使用协同ALNS求解器
            if self.n_customers <= self.milp_threshold and MILP_AVAILABLE:
                return 'milp'
            elif ALNS_AVAILABLE:
                return 'alns'
            elif MILP_AVAILABLE:
                return 'milp'
            else:
                raise RuntimeError("No solver available!")
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _solve_milp(self, time_limit: int) -> Dict:
        """使用MILP求解"""
        if not MILP_AVAILABLE:
            raise RuntimeError("MILP solver not available!")
        
        print("使用MILP精确求解器...")
        
        solver = ETRD_MILP_Solver_PULP(self.instance)
        solver.build_model()
        
        solution = solver.solve(time_limit=time_limit)
        
        if solution:
            print(f"MILP求解完成!")
            print(f"目标值: {solution['makespan']:.2f}")
            print(f"求解时间: {solution['solve_time']:.2f}秒")
            print(f"状态: {solution['status']}")
        else:
            print("MILP未能找到可行解，切换到ALNS...")
            return self._solve_alns(time_limit)
        
        return solution
    
    def _solve_alns(self, time_limit: int) -> Dict:
        """使用ALNS协同求解器（Truck-Robot协同配送）"""
        if not ALNS_AVAILABLE:
            raise RuntimeError("ALNS solver not available!")
        
        print("使用ALNS协同求解器（Truck-Robot协同配送）...")
        solver = ETRD_ALNS_Collaborative_Solver(self.instance)
        
        # 运行ALNS
        solution = solver.solve(time_limit=time_limit)
        
        print(f"ALNS求解完成!")
        print(f"目标值: {solution['makespan']:.2f}")
        
        return solution
    
    def _solve_hybrid(self, time_limit: int) -> Dict:
        """混合策略：先用MILP快速求解，然后用ALNS协同求解器改进"""
        if not MILP_AVAILABLE:
            # 只有ALNS可用
            if ALNS_AVAILABLE:
                return self._solve_alns(time_limit)
            else:
                raise RuntimeError("No solver available!")
        
        print("使用混合策略（MILP + ALNS）...")
        
        # 第一阶段：MILP快速求解（较短时间）
        milp_time = min(time_limit // 3, 60)  # 最多用1/3时间，最多60秒
        print(f"\n第一阶段: MILP求解 ({milp_time}秒)")
        
        milp_solver = ETRD_MILP_Solver_PULP(self.instance)
        milp_solver.build_model()
        milp_solution = milp_solver.solve(time_limit=milp_time)
        
        # 第二阶段：ALNS改进
        alns_time = time_limit - milp_time
        print(f"\n第二阶段: ALNS协同求解 ({alns_time}秒)")

        if ALNS_AVAILABLE:
            alns_solver = ETRD_ALNS_Collaborative_Solver(self.instance)
        else:
            raise RuntimeError("ALNS solver not available!")
        
        initial_solution = milp_solution if milp_solution else None
        final_solution = alns_solver.solve(initial_solution, time_limit=alns_time)
        
        # 比较两个解
        if milp_solution and milp_solution['makespan'] < final_solution['makespan']:
            print(f"\nMILP解更优，使用MILP解")
            return milp_solution
        else:
            print(f"\nALNS解更优或MILP无解，使用ALNS解")
            return final_solution
    
    def _generate_greedy_solution(self) -> Dict:
        """生成贪心初始解"""
        solution = {
            'truck_route': [],
            'robot_route': [],
            'truck_energy': {},
            'robot_energy': {},
            'truck_time': {},
            'robot_time': {},
            'makespan': 0.0
        }
        
        # 提取参数
        depot = np.array(self.instance['depot'])
        customers = np.array(self.instance['customers'])
        service_times = self.instance['service_times']
        truck_speed = self.instance['truck']['speed']
        truck_energy_rate = self.instance['truck']['energy_rate']
        battery_capacity = self.instance['truck']['battery_capacity']
        
        # 计算距离矩阵
        coords = np.vstack([depot.reshape(1, 2), customers, depot.reshape(1, 2)])
        n_nodes = len(coords)
        distances = np.zeros((n_nodes, n_nodes))
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    distances[i, j] = np.linalg.norm(coords[i] - coords[j])
        
        # 贪心构造
        n_customers = len(customers)
        visited = [False] * n_customers
        current_node = 0  # 从仓库出发
        current_energy = battery_capacity
        current_time = 0.0
        
        while not all(visited):
            best_customer = None
            best_score = float('inf')
            
            for i in range(n_customers):
                if not visited[i]:
                    customer_node = i + 1
                    dist = distances[current_node, customer_node]
                    energy_needed = dist * truck_energy_rate
                    
                    # 检查能量约束
                    if energy_needed <= current_energy:
                        score = dist
                        if score < best_score:
                            best_score = score
                            best_customer = i
            
            if best_customer is None:
                # 需要返回仓库充电
                solution['truck_route'].append((current_node, n_customers + 1))
                current_time += distances[current_node, n_customers + 1] / truck_speed
                current_node = n_customers + 1
                current_energy = battery_capacity
            else:
                customer_node = best_customer + 1
                solution['truck_route'].append((current_node, customer_node))
                dist = distances[current_node, customer_node]
                current_time += dist / truck_speed
                current_time += service_times.get(customer_node, 0.5)
                current_energy -= dist * truck_energy_rate
                current_node = customer_node
                visited[best_customer] = True
                
                solution['truck_time'][customer_node] = current_time
                solution['truck_energy'][customer_node] = current_energy
        
        # 返回仓库
        solution['truck_route'].append((current_node, n_nodes - 1))
        current_time += distances[current_node, n_nodes - 1] / truck_speed
        solution['makespan'] = current_time
        
        return solution
    
    def print_solution(self, solution: Dict = None):
        """打印解"""
        if solution is None:
            solution = self.solution
        
        if solution is None:
            print("No solution available!")
            return
        
        print(f"\n{'='*60}")
        print("SOLUTION SUMMARY")
        print(f"{'='*60}")
        print(f"方法: {self.stats['method_used']}")
        print(f"求解时间: {self.stats['solve_time']:.2f}秒")
        print(f"目标值 (Makespan): {self.stats['solution_quality']:.2f}")
        
        print("\n卡车路径:")
        for i, (from_node, to_node) in enumerate(solution.get('truck_route', [])):
            print(f"  {i+1}. 节点 {from_node} -> 节点 {to_node}")
        
        if solution.get('robot_route'):
            print("\n机器人路径:")
            for i, (from_node, to_node) in enumerate(solution.get('robot_route', [])):
                print(f"  {i+1}. 节点 {from_node} -> 节点 {to_node}")
        
        print(f"\n{'='*60}")
    
    def save_solution(self, solution: Dict = None, filename: str = 'solution.json'):
        """保存解到文件"""
        import json
        
        if solution is None:
            solution = self.solution
        
        if solution is None:
            print("No solution to save!")
            return
        
        # 转换解以便JSON序列化
        solution_json = {
            'makespan': solution.get('makespan', 0),
            'truck_route': [list(arc) for arc in solution.get('truck_route', [])],
            'robot_route': [list(arc) for arc in solution.get('robot_route', [])],
            'truck_energy': {str(k): v for k, v in solution.get('truck_energy', {}).items()},
            'robot_energy': {str(k): v for k, v in solution.get('robot_energy', {}).items()},
            'truck_time': {str(k): v for k, v in solution.get('truck_time', {}).items()},
            'robot_time': {str(k): v for k, v in solution.get('robot_time', {}).items()},
            'stats': self.stats
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(solution_json, f, indent=2, ensure_ascii=False)
        
        print(f"Solution saved to {filename}")


def solve_etrn_nl(instance: Dict, strategy: str = 'auto', time_limit: int = 300) -> Dict:
    """
    便捷函数：求解ETRD-NL问题
    
    Args:
        instance: 问题实例
        strategy: 求解策略 ('auto', 'milp', 'alns', 'hybrid')
        time_limit: 时间限制（秒）
    
    Returns:
        最优解
    """
    solver = ETRD_Hybrid_Solver(instance, strategy=strategy)
    solution = solver.solve(time_limit=time_limit)
    solver.print_solution()
    return solution