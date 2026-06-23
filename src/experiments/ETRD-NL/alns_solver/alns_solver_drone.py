"""
ETRD-NL ALNS Solver - Truck-Robot Collaborative Version
自适应大邻域搜索求解器 - 支持卡车机器人协同配送
"""
import numpy as np
import random
import copy
import time
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict


class ETRD_ALNS_Collaborative_Solver:
    """ETRD-NL ALNS协同求解器 - 支持Truck-Robot协同配送"""
    
    def __init__(self, instance: Dict):
        self.instance = instance
        self.current_solution = None
        self.best_solution = None
        
        # 提取数据
        self.depot = np.array(instance['depot'])
        self.customers = np.array(instance['customers'])
        self.charging_stations = np.array(instance['charging_stations'])
        self.service_times = instance['service_times']
        
        self.n_customers = instance['n_customers']
        self.n_cs = instance['n_charging_stations']
        
        # 节点索引: 
        # 0 = 起点仓库
        # 1~N = 客户
        # N+1~N+CS = 充电站（交接点）
        # N+CS+1 = 终点仓库
        self.depot_idx = 0
        self.end_depot_idx = self.n_customers + self.n_cs + 1
        self.N = list(range(1, self.n_customers + 1))  # 客户节点
        self.CS = list(range(self.n_customers + 1, self.n_customers + self.n_cs + 1))  # 充电站
        
        # 所有节点坐标
        self.coords = np.vstack([
            self.depot.reshape(1, 2),  # 0: 起点仓库
            self.customers,             # 1~N: 客户
            self.charging_stations,     # N+1~N+CS: 充电站
            self.depot.reshape(1, 2)    # N+CS+1: 终点仓库
        ])
        
        # 计算距离矩阵
        self.distances = self._compute_distance_matrix()
        
        # 车辆参数
        self.truck_params = instance['truck']
        self.robot_params = instance['robot']
        
        # 时间窗参数
        self.time_windows = instance.get('time_windows', {})
        for c in self.N:
            if c not in self.time_windows:
                self.time_windows[c] = (0, float('inf'))
        
        # ALNS参数
        self.max_iterations = 2000
        self.max_time = 600
        self.initial_temperature = 100.0
        self.cooling_rate = 0.997
        
        # 统计信息
        self.stats = {
            'iterations': 0,
            'accepted': 0,
            'improved': 0,
            'worsened': 0
        }
    
    def _compute_distance_matrix(self) -> np.ndarray:
        """计算距离矩阵"""
        n = len(self.coords)
        dist = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist[i, j] = np.linalg.norm(self.coords[i] - self.coords[j])
        return dist
    
    def _compute_travel_time_truck(self, from_node: int, to_node: int) -> float:
        """计算卡车行驶时间"""
        return self.distances[from_node, to_node] / self.truck_params['speed']
    
    def _compute_travel_time_robot(self, from_node: int, to_node: int) -> float:
        """计算机器人行驶时间"""
        return self.distances[from_node, to_node] / self.robot_params['speed']
    
    def _compute_energy_consumption_truck(self, from_node: int, to_node: int) -> float:
        """计算卡车能量消耗"""
        return self.distances[from_node, to_node] * self.truck_params['energy_rate']
    
    def _compute_energy_consumption_robot(self, from_node: int, to_node: int) -> float:
        """计算机器人能量消耗"""
        return self.distances[from_node, to_node] * self.robot_params['energy_rate']
    
    def _create_initial_solution(self) -> Dict:
        """创建初始解 - 使用贪心分配策略"""
        solution = {
            'assignment': {},           # {customer_id: 'truck'|'robot'}
            'truck_route': [],           # [(from, to), ...]
            'robot_route': [],           # [(from, to), ...] - robot在交接点之间移动
            'handoff_point': None,       # 交接点（充电站索引）
            'visited_customers': set(),
            'truck_customers': set(),
            'robot_customers': set(),
            'makespan': float('inf')
        }
        
        # Step 1: 基于贪心策略分配客户
        # 策略：距离仓库较远的客户由truck服务，较近的由robot服务
        customer_distances = []
        for i, customer in enumerate(self.customers, start=1):
            dist_to_depot = self.distances[0, i]
            customer_distances.append((i, dist_to_depot))
        
        # 按距离排序
        customer_distances.sort(key=lambda x: x[1])
        
        # 分配：前半部分给robot，后半部分给truck
        # 考虑robot电池容量限制
        robot_max_range = self.robot_params['battery_capacity'] / self.robot_params['energy_rate']
        mid_point = len(customer_distances) // 2
        
        for idx, (customer_id, dist) in enumerate(customer_distances):
            if idx < mid_point and dist < robot_max_range * 0.3:
                solution['assignment'][customer_id] = 'robot'
                solution['robot_customers'].add(customer_id)
            else:
                solution['assignment'][customer_id] = 'truck'
                solution['truck_customers'].add(customer_id)
        
        solution['visited_customers'] = set(solution['assignment'].keys())
        
        # Step 2: 选择交接点（充电站）
        # 选择距离客户区域中心最近的充电站
        center_x = np.mean(self.customers[:, 0])
        center_y = np.mean(self.customers[:, 1])
        
        min_dist = float('inf')
        best_cs = self.CS[0]
        for cs in self.CS:
            dist = np.linalg.norm(self.charging_stations[cs - self.n_customers - 1] - np.array([center_x, center_y]))
            if dist < min_dist:
                min_dist = dist
                best_cs = cs
        
        solution['handoff_point'] = best_cs
        
        # Step 3: 构建truck路线
        solution['truck_route'] = self._build_truck_route_greedy(solution)
        
        # Step 4: 构建robot路线
        solution['robot_route'] = self._build_robot_route_greedy(solution)
        
        # Step 5: 计算目标值
        solution['makespan'] = self._calculate_makespan(solution)
        
        return solution
    
    def _build_truck_route_greedy(self, solution: Dict) -> List[Tuple[int, int]]:
        """贪心构建truck路线"""
        truck_customers = sorted(solution['truck_customers'])
        if not truck_customers:
            return []
        
        route = []
        current_node = self.depot_idx
        
        while truck_customers:
            best_next = None
            best_cost = float('inf')
            
            for customer in truck_customers:
                cost = self.distances[current_node, customer]
                if cost < best_cost:
                    best_cost = cost
                    best_next = customer
            
            if best_next is not None:
                route.append((current_node, best_next))
                current_node = best_next
                truck_customers.remove(best_next)
        
        # 返回仓库
        route.append((current_node, self.end_depot_idx))
        return route
    
    def _build_robot_route_greedy(self, solution: Dict) -> List[Tuple[int, int]]:
        """贪心构建robot路线"""
        robot_customers = sorted(solution['robot_customers'])
        if not robot_customers:
            return []
        
        handoff = solution['handoff_point']
        route = []
        current_node = handoff
        
        while robot_customers:
            best_next = None
            best_cost = float('inf')
            
            for customer in robot_customers:
                cost = self.distances[current_node, customer]
                if cost < best_cost:
                    best_cost = cost
                    best_next = customer
            
            if best_next is not None:
                route.append((current_node, best_next))
                current_node = best_next
                robot_customers.remove(best_next)
        
        # 返回交接点
        route.append((current_node, handoff))
        return route
    
    def _calculate_makespan(self, solution: Dict) -> float:
        """计算目标值（makespan）- 卡车和机器人完成时间的最大值"""
        truck_time = self._calculate_truck_makespan(solution)
        robot_time = self._calculate_robot_makespan(solution)
        return max(truck_time, robot_time)
    
    def _calculate_truck_makespan(self, solution: Dict) -> float:
        """计算卡车的完工时间"""
        if not solution['truck_route']:
            return 0.0
        
        current_time = 0.0
        current_node = self.depot_idx
        
        for from_node, to_node in solution['truck_route']:
            travel_time = self._compute_travel_time_truck(from_node, to_node)
            service_time = self.service_times.get(to_node, 0)
            current_time += travel_time + service_time
            current_node = to_node
        
        return current_time
    
    def _calculate_robot_makespan(self, solution: Dict) -> float:
        """计算机器人的完工时间"""
        if not solution['robot_route']:
            return 0.0
        
        current_time = 0.0
        current_node = solution['handoff_point']
        
        for from_node, to_node in solution['robot_route']:
            travel_time = self._compute_travel_time_robot(from_node, to_node)
            service_time = self.service_times.get(to_node, 0)
            current_time += travel_time + service_time
            current_node = to_node
        
        return current_time
    
    def _destroy_random(self, solution: Dict, num_remove: int = 5) -> Dict:
        """破坏算子：随机移除客户"""
        partial = copy.deepcopy(solution)
        
        all_customers = list(partial['visited_customers'])
        if len(all_customers) <= num_remove:
            num_remove = len(all_customers) - 1
        
        to_remove = set(random.sample(all_customers, num_remove))
        
        for customer in to_remove:
            vehicle = partial['assignment'].pop(customer)
            partial['visited_customers'].remove(customer)
            
            if vehicle == 'truck':
                partial['truck_customers'].remove(customer)
            else:
                partial['robot_customers'].remove(customer)
        
        # 重建路线
        partial['truck_route'] = self._rebuild_truck_route(partial)
        partial['robot_route'] = self._rebuild_robot_route(partial)
        
        return partial
    
    def _destroy_worst(self, solution: Dict, num_remove: int = 5) -> Dict:
        """破坏算子：移除造成最大绕路的客户"""
        partial = copy.deepcopy(solution)
        
        # 计算每个客户的绕路成本
        customer_costs = []
        
        # Truck客户
        for customer in partial['truck_customers']:
            cost = self._calculate_detour_cost(solution, customer, 'truck')
            customer_costs.append((customer, cost, 'truck'))
        
        # Robot客户
        for customer in partial['robot_customers']:
            cost = self._calculate_detour_cost(solution, customer, 'robot')
            customer_costs.append((customer, cost, 'robot'))
        
        # 排序并移除成本最高的
        customer_costs.sort(key=lambda x: x[1], reverse=True)
        
        to_remove = set()
        for customer, cost, vehicle in customer_costs[:num_remove]:
            to_remove.add(customer)
        
        for customer in to_remove:
            partial['assignment'].pop(customer)
            partial['visited_customers'].remove(customer)
            
            if customer in partial['truck_customers']:
                partial['truck_customers'].remove(customer)
            else:
                partial['robot_customers'].remove(customer)
        
        # 重建路线
        partial['truck_route'] = self._rebuild_truck_route(partial)
        partial['robot_route'] = self._rebuild_robot_route(partial)
        
        return partial
    
    def _calculate_detour_cost(self, solution: Dict, customer: int, vehicle: str) -> float:
        """计算移除客户后的绕路成本"""
        # 简化版本：使用距离作为成本
        if vehicle == 'truck':
            base_dist = sum(self.distances[f, t] for f, t in solution['truck_route'])
            return self.distances[customer, self.depot_idx]
        else:
            return self.distances[customer, solution['handoff_point']]
    
    def _rebuild_truck_route(self, solution: Dict) -> List[Tuple[int, int]]:
        """重建truck路线"""
        truck_customers = sorted(solution['truck_customers'])
        if not truck_customers:
            return []
        
        route = []
        current_node = self.depot_idx
        
        while truck_customers:
            best_next = None
            best_cost = float('inf')
            
            for customer in truck_customers:
                cost = self.distances[current_node, customer]
                if cost < best_cost:
                    best_cost = cost
                    best_next = customer
            
            if best_next is not None:
                route.append((current_node, best_next))
                current_node = best_next
                truck_customers.remove(best_next)
        
        route.append((current_node, self.end_depot_idx))
        return route
    
    def _rebuild_robot_route(self, solution: Dict) -> List[Tuple[int, int]]:
        """重建robot路线"""
        robot_customers = sorted(solution['robot_customers'])
        if not robot_customers:
            return []
        
        handoff = solution['handoff_point']
        route = []
        current_node = handoff
        
        while robot_customers:
            best_next = None
            best_cost = float('inf')
            
            for customer in robot_customers:
                cost = self.distances[current_node, customer]
                if cost < best_cost:
                    best_cost = cost
                    best_next = customer
            
            if best_next is not None:
                route.append((current_node, best_next))
                current_node = best_next
                robot_customers.remove(best_next)
        
        route.append((current_node, handoff))
        return route
    
    def _repair_greedy(self, partial: Dict) -> Dict:
        """修复算子：贪心重新插入客户"""
        solution = copy.deepcopy(partial)
        
        unvisited = set(range(1, self.n_customers + 1)) - solution['visited_customers']
        if not unvisited:
            solution['makespan'] = self._calculate_makespan(solution)
            return solution
        
        # 逐个插入未访问的客户
        while unvisited:
            best_customer = None
            best_vehicle = None
            best_position = None
            best_cost_increase = float('inf')
            
            # 尝试插入到truck
            truck_route = solution['truck_route']
            for customer in unvisited:
                for pos in range(len(truck_route) + 1):
                    cost_increase = self._calculate_insertion_cost_truck(solution, customer, pos)
                    if cost_increase < best_cost_increase:
                        best_cost_increase = cost_increase
                        best_customer = customer
                        best_vehicle = 'truck'
                        best_position = pos
            
            # 尝试插入到robot
            robot_route = solution['robot_route']
            for customer in unvisited:
                for pos in range(len(robot_route) + 1):
                    cost_increase = self._calculate_insertion_cost_robot(solution, customer, pos)
                    if cost_increase < best_cost_increase:
                        best_cost_increase = cost_increase
                        best_customer = customer
                        best_vehicle = 'robot'
                        best_position = pos
            
            if best_customer is None:
                break
            
            # 执行插入
            if best_vehicle == 'truck':
                solution['assignment'][best_customer] = 'truck'
                solution['truck_customers'].add(best_customer)
                solution['truck_route'] = self._insert_into_truck_route(
                    solution['truck_route'], best_customer, best_position
                )
            else:
                solution['assignment'][best_customer] = 'robot'
                solution['robot_customers'].add(best_customer)
                solution['robot_route'] = self._insert_into_robot_route(
                    solution['robot_route'], best_customer, best_position
                )
            
            solution['visited_customers'].add(best_customer)
            unvisited.remove(best_customer)
        
        solution['makespan'] = self._calculate_makespan(solution)
        return solution
    
    def _calculate_insertion_cost_truck(self, solution: Dict, customer: int, position: int) -> float:
        """计算在truck路线中插入客户的成本增加"""
        route = solution['truck_route']
        
        if len(route) == 0:
            # 空路线
            return (self.distances[self.depot_idx, customer] + 
                   self.distances[customer, self.end_depot_idx])
        
        if position == 0:
            # 插入到最前面
            old_cost = self.distances[self.depot_idx, route[0][0]]
            new_cost = self.distances[self.depot_idx, customer] + self.distances[customer, route[0][0]]
        elif position >= len(route):
            # 插入到最后面
            old_cost = self.distances[route[-1][1], self.end_depot_idx]
            new_cost = self.distances[route[-1][1], customer] + self.distances[customer, self.end_depot_idx]
        else:
            # 插入到中间
            from_node = route[position-1][1]
            to_node = route[position][0]
            old_cost = self.distances[from_node, to_node]
            new_cost = self.distances[from_node, customer] + self.distances[customer, to_node]
        
        return new_cost - old_cost
    
    def _calculate_insertion_cost_robot(self, solution: Dict, customer: int, position: int) -> float:
        """计算在robot路线中插入客户的成本增加"""
        route = solution['robot_route']
        handoff = solution['handoff_point']
        
        if len(route) == 0:
            return (self.distances[handoff, customer] * 2)
        
        if position == 0:
            old_cost = self.distances[handoff, route[0][0]]
            new_cost = self.distances[handoff, customer] + self.distances[customer, route[0][0]]
        elif position >= len(route):
            old_cost = self.distances[route[-1][1], handoff]
            new_cost = self.distances[route[-1][1], customer] + self.distances[customer, handoff]
        else:
            from_node = route[position-1][1]
            to_node = route[position][0]
            old_cost = self.distances[from_node, to_node]
            new_cost = self.distances[from_node, customer] + self.distances[customer, to_node]
        
        return new_cost - old_cost
    
    def _insert_into_truck_route(self, route: List[Tuple], customer: int, position: int) -> List[Tuple[int, int]]:
        """将客户插入truck路线"""
        if len(route) == 0:
            return [(customer, self.end_depot_idx)]
        
        if position == 0:
            return [(customer, route[0][0])] + route
        elif position >= len(route):
            return route[:-1] + [(route[-1][1], customer), (customer, self.end_depot_idx)]
        else:
            from_node = route[position-1][1]
            to_node = route[position][0]
            new_route = route[:position-1] + [(from_node, customer), (customer, to_node)] + route[position:]
            return new_route
    
    def _insert_into_robot_route(self, route: List[Tuple], customer: int, position: int) -> List[Tuple[int, int]]:
        """将客户插入robot路线"""
        handoff = self.current_solution['handoff_point'] if self.current_solution else self.CS[0]
        
        if len(route) == 0:
            return [(handoff, customer), (customer, handoff)]
        
        if position == 0:
            return [(handoff, customer)] + route
        elif position >= len(route):
            return route + [(route[-1][1], customer), (customer, handoff)]
        else:
            from_node = route[position-1][1]
            to_node = route[position][0]
            new_route = route[:position-1] + [(from_node, customer), (customer, to_node)] + route[position:]
            return new_route
    
    def _accept_solution(self, new_makespan: float, temperature: float) -> bool:
        """模拟退火接受准则"""
        if new_makespan < self.current_solution['makespan']:
            return True
        
        delta = new_makespan - self.current_solution['makespan']
        probability = np.exp(-delta / temperature)
        
        return random.random() < probability
    
    def solve(self, initial_solution: Optional[Dict] = None, time_limit: int = 600) -> Dict:
        """
        求解ETRD-NL问题
        
        Args:
            initial_solution: 初始解（可选）
            time_limit: 时间限制（秒）
        
        Returns:
            最优解
        """
        self.max_time = time_limit
        start_time = time.time()
        
        # 初始化
        if initial_solution is not None:
            self.current_solution = copy.deepcopy(initial_solution)
        else:
            self.current_solution = self._create_initial_solution()
        
        self.best_solution = copy.deepcopy(self.current_solution)
        
        print(f"ALNS协同求解开始")
        print(f"初始解目标值: {self.current_solution['makespan']:.2f}")
        print(f"Truck客户: {len(self.current_solution['truck_customers'])}")
        print(f"Robot客户: {len(self.current_solution['robot_customers'])}")
        print(f"交接点: {self.current_solution['handoff_point']}")
        
        temperature = self.initial_temperature
        iteration = 0
        
        destroy_operators = [self._destroy_random, self._destroy_worst]
        repair_operators = [self._repair_greedy]
        
        while (time.time() - start_time < self.max_time and 
               iteration < self.max_iterations):
            
            # 选择算子
            destroy_op = random.choice(destroy_operators)
            repair_op = random.choice(repair_operators)
            
            # 破坏
            partial = destroy_op(self.current_solution, num_remove=5)
            
            # 修复
            new_solution = repair_op(partial)
            
            # 接受准则
            if self._accept_solution(new_solution['makespan'], temperature):
                self.current_solution = new_solution
                self.stats['accepted'] += 1
                
                if new_solution['makespan'] < self.best_solution['makespan']:
                    self.best_solution = copy.deepcopy(new_solution)
                    self.stats['improved'] += 1
                    print(f"  迭代 {iteration}: 改进到 {self.best_solution['makespan']:.2f}")
                elif new_solution['makespan'] > self.current_solution['makespan']:
                    self.stats['worsened'] += 1
            
            # 降温
            temperature *= self.cooling_rate
            iteration += 1
            self.stats['iterations'] = iteration
        
        elapsed = time.time() - start_time
        print(f"ALNS求解完成!")
        print(f"求解时间: {elapsed:.2f}秒")
        print(f"迭代次数: {iteration}")
        print(f"最优目标值: {self.best_solution['makespan']:.2f}")
        
        return self.best_solution
    
    def print_solution(self, solution: Optional[Dict] = None):
        """打印解"""
        if solution is None:
            solution = self.best_solution
        
        print(f"\n{'='*60}")
        print("ETRD-NL ALNS SOLUTION (Truck-Robot Collaborative)")
        print(f"{'='*60}")
        print(f"目标值 (Makespan): {solution['makespan']:.2f}")
        print(f"Truck客户: {len(solution['truck_customers'])}/{self.n_customers}")
        print(f"Robot客户: {len(solution['robot_customers'])}/{self.n_customers}")
        print(f"交接点: {solution['handoff_point']}")
        
        print("\nTruck路径:")
        if solution['truck_route']:
            print(f"  仓库 -> 节点 {solution['truck_route'][0][0]}")
            for from_node, to_node in solution['truck_route']:
                print(f"  节点 {from_node} -> 节点 {to_node}")
        else:
            print("  (无)")
        
        print("\nRobot路径:")
        if solution['robot_route']:
            print(f"  交接点 {solution['handoff_point']} -> 节点 {solution['robot_route'][0][0]}")
            for from_node, to_node in solution['robot_route']:
                print(f"  节点 {from_node} -> 节点 {to_node}")
        else:
            print("  (无)")
        
        print(f"\n{'='*60}")
