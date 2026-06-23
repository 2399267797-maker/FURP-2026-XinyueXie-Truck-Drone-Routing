"""
ETRD-NL ALNS Solver
自适应大邻域搜索求解器 - 修复版本
"""
import numpy as np
import random
import copy
import time
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class ETRD_ALNS_Solver:
    """ETRD-NL ALNS求解器"""
    
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
        
        # 节点索引: 0=仓库, 1~N=客户, N+1~N+CS=充电站, N+CS+1=终点仓库
        self.depot_idx = 0
        self.end_depot_idx = self.n_customers + self.n_cs + 1
        self.N = list(range(1, self.n_customers + 1))  # 客户节点
        self.CS = list(range(self.n_customers + 1, self.n_customers + self.n_cs + 1))  # 充电站
        
        # 所有节点坐标
        self.coords = np.vstack([
            self.depot.reshape(1, 2),  # 0: 起点仓库
            self.customers,               # 1~N: 客户
            self.charging_stations,      # N+1~N+CS: 充电站
            self.depot.reshape(1, 2)     # N+CS+1: 终点仓库
        ])
        
        # 计算距离矩阵
        self.distances = self._compute_distance_matrix()
        
        # 车辆参数
        self.truck_params = instance['truck']
        self.robot_params = instance['robot']
        
        # 时间窗参数 (客户编号 -> (最早, 最晚))
        self.time_windows = instance.get('time_windows', {})
        # 确保所有客户都有时间窗
        for c in self.N:
            if c not in self.time_windows:
                self.time_windows[c] = (0, float('inf'))
        
        # ALNS参数
        self.max_iterations = 2000
        self.max_time = 600  # 最大运行时间（秒）
        self.initial_temperature = 100.0
        self.cooling_rate = 0.997
        
        # 算子权重（自适应调整）
        self.destroy_weights = {
            'random': 1.0,
            'worst': 1.0,
            ' Shaw': 1.0,
            'distance': 1.0
        }
        self.repair_weights = {
            'greedy': 1.0,
            'regret': 1.0,
            'random_insert': 1.0
        }
        
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
    
    def _compute_travel_time(self, from_node: int, to_node: int) -> float:
        """计算行驶时间"""
        return self.distances[from_node, to_node] / self.truck_params['speed']
    
    def _compute_energy_consumption(self, from_node: int, to_node: int) -> float:
        """计算能量消耗"""
        return self.distances[from_node, to_node] * self.truck_params['energy_rate']
    
    def solve(self, initial_solution: Optional[Dict] = None, time_limit: int = 600) -> Dict:
        """
        求解ETRD-NL问题
        """
        self.max_time = time_limit
        start_time = time.time()
        
        # 初始化或使用提供的初始解
        if initial_solution is not None:
            self.current_solution = self._normalize_solution(initial_solution)
        else:
            self.current_solution = self._generate_initial_solution()
        
        self.best_solution = copy.deepcopy(self.current_solution)
        
        print(f"ALNS求解开始")
        initial_cost = self._evaluate_solution(self.current_solution)
        print(f"初始解目标值: {initial_cost:.2f}")
        print(f"初始解访问客户数: {len(self.current_solution['visited_customers'])}/{self.n_customers}")
        
        temperature = self.initial_temperature
        
        while time.time() - start_time < self.max_time and self.stats['iterations'] < self.max_iterations:
            self.stats['iterations'] += 1
            
            # 选择破坏和修复算子
            destroy_operator = self._select_destroy_operator()
            repair_operator = self._select_repair_operator()
            
            # 破坏当前解
            partial_solution = self._apply_destroy_operator(destroy_operator)
            
            # 修复得到新解
            new_solution = self._apply_repair_operator(repair_operator, partial_solution)
            
            # 评估解
            current_cost = self._evaluate_solution(self.current_solution)
            new_cost = self._evaluate_solution(new_solution)
            
            # Metropolis接受准则
            delta = new_cost - current_cost
            if delta < 0 or random.random() < np.exp(-delta / temperature):
                self.current_solution = new_solution
                self.stats['accepted'] += 1
                
                if delta < 0:
                    self.stats['improved'] += 1
                    # 更新最优解
                    if new_cost < self._evaluate_solution(self.best_solution):
                        self.best_solution = copy.deepcopy(new_solution)
                else:
                    self.stats['worsened'] += 1
            
            # 降温
            temperature *= self.cooling_rate
            
            # 更新算子权重
            self._update_operator_weights()
            
            # 定期输出进度
            if self.stats['iterations'] % 200 == 0:
                elapsed = time.time() - start_time
                best_cost = self._evaluate_solution(self.best_solution)
                visited = len(self.best_solution['visited_customers'])
                print(f"迭代 {self.stats['iterations']}: "
                      f"当前成本 = {current_cost:.2f}, "
                      f"最优成本 = {best_cost:.2f}, "
                      f"访问客户 = {visited}/{self.n_customers}, "
                      f"温度 = {temperature:.2f}, "
                      f"时间 = {elapsed:.1f}s")
        
        total_time = time.time() - start_time
        print(f"\nALNS求解完成")
        print(f"总迭代次数: {self.stats['iterations']}")
        print(f"总运行时间: {total_time:.2f}秒")
        print(f"最优成本: {self._evaluate_solution(self.best_solution):.2f}")
        print(f"访问客户数: {len(self.best_solution['visited_customers'])}/{self.n_customers}")
        print(f"接受率: {self.stats['accepted']/max(1,self.stats['iterations'])*100:.1f}%")
        print(f"改进率: {self.stats['improved']/max(1,self.stats['iterations'])*100:.1f}%")
        
        return self.best_solution
    
    def _normalize_solution(self, solution: Dict) -> Dict:
        """标准化解格式"""
        normalized = copy.deepcopy(solution)
        if 'visited_customers' not in normalized:
            # 从route提取已访问客户
            visited = set()
            for from_node, to_node in solution.get('truck_route', []):
                if 1 <= from_node <= self.n_customers:
                    visited.add(from_node)
                if 1 <= to_node <= self.n_customers:
                    visited.add(to_node)
            normalized['visited_customers'] = visited
        return normalized
    
    def _generate_initial_solution(self) -> Dict:
        """生成初始解 - 简单的卡车路径贪心构造"""
        solution = {
            'truck_route': [],  # [(from, to), ...]
            'visited_customers': set(),
            'makespan': 0.0
        }
        
        current_node = 0  # 从仓库出发
        current_time = 0.0
        current_energy = self.truck_params['battery_capacity']
        unvisited = set(self.N)
        
        while unvisited:
            # 找到最近的可达客户
            best_customer = None
            best_score = float('inf')
            best_dist = float('inf')
            
            for customer in unvisited:
                dist = self.distances[current_node, customer]
                energy_needed = self._compute_energy_consumption(current_node, customer)
                
                # 检查能量约束
                if energy_needed > current_energy:
                    continue
                
                # 检查时间窗
                tw = self.time_windows.get(customer, (0, float('inf')))
                arrival_time = current_time + self._compute_travel_time(current_node, customer)
                
                # 如果超过时间窗，跳过
                if arrival_time > tw[1] and tw[1] != float('inf'):
                    continue
                
                # 评分：优先距离近的
                score = dist
                if score < best_score:
                    best_score = score
                    best_customer = customer
                    best_dist = dist
            
            if best_customer is None:
                # 没有可达客户，需要充电或结束
                if self.n_cs > 0 and unvisited:
                    # 去最近的充电站
                    best_cs = min(self.CS, key=lambda cs: self.distances[current_node, cs])
                    energy_to_cs = self._compute_energy_consumption(current_node, best_cs)
                    if energy_to_cs <= current_energy:
                        solution['truck_route'].append((current_node, best_cs))
                        current_time += self._compute_travel_time(current_node, best_cs)
                        current_energy -= energy_to_cs
                        current_energy = self.truck_params['battery_capacity']  # 充电
                        current_node = best_cs
                        continue
                
                # 无法继续，停止
                break
            
            # 访问客户
            solution['truck_route'].append((current_node, best_customer))
            travel_time = self._compute_travel_time(current_node, best_customer)
            current_time += travel_time
            current_time += self.service_times.get(best_customer, 0.5)
            current_energy -= self._compute_energy_consumption(current_node, best_customer)
            solution['visited_customers'].add(best_customer)
            unvisited.remove(best_customer)
            current_node = best_customer
        
        # 返回仓库
        if current_node != self.end_depot_idx:
            solution['truck_route'].append((current_node, self.end_depot_idx))
            current_time += self._compute_travel_time(current_node, self.end_depot_idx)
        
        solution['makespan'] = current_time
        
        return solution
    
    def _evaluate_solution(self, solution: Dict) -> float:
        """评估解 - 惩罚未访问客户"""
        makespan = solution.get('makespan', float('inf'))
        
        # 惩罚未访问的客户
        visited = solution.get('visited_customers', set())
        unvisited_count = self.n_customers - len(visited)
        
        # 未访问客户惩罚系数（很大）
        penalty = unvisited_count * 1000.0
        
        # 如果有大量未访问客户，给予极大惩罚
        if unvisited_count > 0:
            return makespan + penalty
        
        return makespan
    
    def _select_destroy_operator(self) -> str:
        """选择破坏算子（轮盘赌）"""
        operators = list(self.destroy_weights.keys())
        weights = list(self.destroy_weights.values())
        total_weight = sum(weights)
        probs = [w / total_weight for w in weights]
        return random.choices(operators, weights=probs, k=1)[0]
    
    def _select_repair_operator(self) -> str:
        """选择修复算子（轮盘赌）"""
        operators = list(self.repair_weights.keys())
        weights = list(self.repair_weights.values())
        total_weight = sum(weights)
        probs = [w / total_weight for w in weights]
        return random.choices(operators, weights=probs, k=1)[0]
    
    def _apply_destroy_operator(self, operator: str) -> Dict:
        """应用破坏算子"""
        if operator == 'random':
            return self._destroy_random()
        elif operator == 'worst':
            return self._destroy_worst()
        elif operator == ' Shaw':
            return self._destroy_shaw()
        elif operator == 'distance':
            return self._destroy_distance()
        else:
            return self._destroy_random()
    
    def _apply_repair_operator(self, operator: str, partial_solution: Dict) -> Dict:
        """应用修复算子"""
        if operator == 'greedy':
            return self._repair_greedy(partial_solution)
        elif operator == 'regret':
            return self._repair_regret(partial_solution)
        elif operator == 'random_insert':
            return self._repair_random(partial_solution)
        else:
            return self._repair_greedy(partial_solution)
    
    def _destroy_random(self, num_remove: int = None) -> Dict:
        """随机破坏：随机移除若干客户"""
        partial = copy.deepcopy(self.current_solution)
        
        visited = list(partial['visited_customers'])
        if len(visited) < 2:
            return partial
        
        if num_remove is None:
            num_remove = max(1, len(visited) // 5)
        
        # 随机选择要移除的客户
        to_remove = set(random.sample(visited, min(num_remove, len(visited))))
        
        # 从路径中移除这些客户
        new_route = []
        for from_node, to_node in partial['truck_route']:
            if from_node in to_remove:
                continue
            if to_node in to_remove:
                continue
            new_route.append((from_node, to_node))
        
        # 更新已访问客户
        new_visited = partial['visited_customers'] - to_remove
        partial['truck_route'] = new_route
        partial['visited_customers'] = new_visited
        
        return partial
    
    def _destroy_worst(self, num_remove: int = None) -> Dict:
        """最差破坏：移除造成最大绕路的客户"""
        partial = copy.deepcopy(self.current_solution)
        
        visited = list(partial['visited_customers'])
        if len(visited) < 2:
            return partial
        
        if num_remove is None:
            num_remove = max(1, len(visited) // 5)
        
        # 构建当前路径上的客户顺序
        route_customers = []
        for from_node, to_node in partial['truck_route']:
            if 1 <= from_node <= self.n_customers:
                route_customers.append(from_node)
            if 1 <= to_node <= self.n_customers:
                route_customers.append(to_node)
        
        # 计算每个客户的绕路成本
        costs = []
        for i, customer in enumerate(route_customers):
            if i == 0:
                continue
            # 绕路 = 直接去下一个 - 经过这个客户
            # 简化计算
            prev_customer = route_customers[i-1]
            next_customer = route_customers[i+1] if i+1 < len(route_customers) else self.end_depot_idx
            
            direct = self.distances[prev_customer, next_customer]
            via = self.distances[prev_customer, customer] + self.distances[customer, next_customer]
            detour = via - direct
            costs.append((customer, detour))
        
        # 移除绕路最大的
        costs.sort(key=lambda x: x[1], reverse=True)
        to_remove = set([c[0] for c in costs[:num_remove]])
        
        # 从路径中移除
        new_route = []
        for from_node, to_node in partial['truck_route']:
            if from_node in to_remove or to_node in to_remove:
                continue
            new_route.append((from_node, to_node))
        
        partial['truck_route'] = new_route
        partial['visited_customers'] = partial['visited_customers'] - to_remove
        
        return partial
    
    def _destroy_shaw(self, num_remove: int = None) -> Dict:
        """Shaw破坏：移除相似的相邻客户"""
        partial = copy.deepcopy(self.current_solution)
        
        visited = list(partial['visited_customers'])
        if len(visited) < 2:
            return partial
        
        if num_remove is None:
            num_remove = max(1, len(visited) // 5)
        
        # 构建路径
        route_customers = []
        for from_node, to_node in partial['truck_route']:
            if 1 <= from_node <= self.n_customers and from_node not in route_customers:
                route_customers.append(from_node)
            if 1 <= to_node <= self.n_customers and to_node not in route_customers:
                route_customers.append(to_node)
        
        if len(route_customers) < 2:
            return partial
        
        # 随机选一个客户作为种子
        seed_idx = random.randint(0, len(route_customers) - 1)
        seed_customer = route_customers[seed_idx]
        
        # 计算与其他客户的距离
        distances_to_seed = []
        for c in route_customers:
            if c != seed_customer:
                d = self.distances[seed_customer, c]
                distances_to_seed.append((c, d))
        
        # 移除距离种子最近的（相似的）
        distances_to_seed.sort(key=lambda x: x[1])
        to_remove = set([seed_customer] + [c[0] for c in distances_to_seed[:num_remove]])
        
        # 从路径中移除
        new_route = []
        for from_node, to_node in partial['truck_route']:
            if from_node in to_remove or to_node in to_remove:
                continue
            new_route.append((from_node, to_node))
        
        partial['truck_route'] = new_route
        partial['visited_customers'] = partial['visited_customers'] - to_remove
        
        return partial
    
    def _destroy_distance(self, num_remove: int = None) -> Dict:
        """距离破坏：移除距离仓库最远的客户"""
        partial = copy.deepcopy(self.current_solution)
        
        visited = list(partial['visited_customers'])
        if len(visited) < 2:
            return partial
        
        if num_remove is None:
            num_remove = max(1, len(visited) // 5)
        
        # 计算每个客户到仓库的距离
        distances_to_depot = []
        for c in visited:
            d = self.distances[0, c]
            distances_to_depot.append((c, d))
        
        # 移除距离最远的
        distances_to_depot.sort(key=lambda x: x[1], reverse=True)
        to_remove = set([c[0] for c in distances_to_depot[:num_remove]])
        
        # 从路径中移除
        new_route = []
        for from_node, to_node in partial['truck_route']:
            if from_node in to_remove or to_node in to_remove:
                continue
            new_route.append((from_node, to_node))
        
        partial['truck_route'] = new_route
        partial['visited_customers'] = partial['visited_customers'] - to_remove
        
        return partial
    
    def _repair_greedy(self, partial: Dict) -> Dict:
        """贪心修复：将未访问客户插入最佳位置"""
        solution = copy.deepcopy(partial)
        
        unvisited = set(self.N) - solution['visited_customers']
        if not unvisited:
            solution['makespan'] = self._calculate_makespan(solution)
            return solution
        
        # 逐个插入未访问的客户
        while unvisited:
            best_customer = None
            best_position = None
            best_cost_increase = float('inf')
            
            for customer in unvisited:
                # 找到最佳插入位置
                route = solution['truck_route']
                
                # 尝试在每个位置插入
                for pos in range(len(route) + 1):
                    # 尝试插入
                    if pos == 0:
                        from_node = 0
                        to_node = route[0][0] if route else self.end_depot_idx
                    elif pos == len(route):
                        from_node = route[-1][1]
                        to_node = self.end_depot_idx
                    else:
                        from_node = route[pos-1][1]
                        to_node = route[pos][0]
                    
                    # 检查能量约束（简化：只检查插入点前后）
                    dist_from = self.distances[from_node, customer]
                    dist_to = self.distances[customer, to_node]
                    dist_direct = self.distances[from_node, to_node]
                    extra_dist = dist_from + dist_to - dist_direct
                    
                    if extra_dist < best_cost_increase:
                        best_cost_increase = extra_dist
                        best_customer = customer
                        best_position = pos
            
            if best_customer is None:
                break
            
            # 执行插入
            new_arcs = []
            customer = best_customer
            route = solution['truck_route']

            if len(route) == 0:
                # 路线为空，创建初始弧
                new_arcs.append((customer, self.end_depot_idx))
            elif best_position == 0:
                # 插在最前面
                new_arcs.append((customer, route[0][0]))
                new_arcs.extend(route)
            elif best_position >= len(route):
                # 插在最后面
                new_arcs.extend(route)
                new_arcs.append((route[-1][1], customer))
            else:
                # 插在中间
                new_arcs.extend(route[:best_position])
                from_node = route[best_position-1][1]
                new_arcs.append((from_node, customer))
                new_arcs.append((customer, route[best_position][0]))
                new_arcs.extend(route[best_position:])
            
            solution['truck_route'] = new_arcs
            solution['visited_customers'].add(customer)
            unvisited.remove(customer)
        
        solution['makespan'] = self._calculate_makespan(solution)
        return solution
    
    def _repair_regret(self, partial: Dict) -> Dict:
        """后悔修复：考虑未来成本插入"""
        solution = copy.deepcopy(partial)
        
        unvisited = set(self.N) - solution['visited_customers']
        if not unvisited:
            solution['makespan'] = self._calculate_makespan(solution)
            return solution
        
        # 逐个插入未访问的客户
        while unvisited:
            best_customer = None
            best_regret = -float('inf')
            best_position = None
            
            for customer in unvisited:
                # 计算插入每个位置的成本
                route = solution['truck_route']
                insertion_costs = []
                
                for pos in range(len(route) + 1):
                    if pos == 0:
                        from_node = 0
                        to_node = route[0][0] if route else self.end_depot_idx
                    elif pos == len(route):
                        from_node = route[-1][1]
                        to_node = self.end_depot_idx
                    else:
                        from_node = route[pos-1][1]
                        to_node = route[pos][0]
                    
                    dist_from = self.distances[from_node, customer]
                    dist_to = self.distances[customer, to_node]
                    dist_direct = self.distances[from_node, to_node]
                    cost = dist_from + dist_to - dist_direct
                    insertion_costs.append(cost)
                
                # 计算后悔值：第二好位置与最好位置的差
                insertion_costs.sort()
                if len(insertion_costs) > 1:
                    regret = insertion_costs[1] - insertion_costs[0]
                else:
                    regret = insertion_costs[0]
                
                if regret > best_regret:
                    best_regret = regret
                    best_customer = customer
                    best_position = insertion_costs.index(min(insertion_costs))
            
            if best_customer is None:
                break
            
            # 执行插入
            new_arcs = []
            customer = best_customer
            route = solution['truck_route']

            if len(route) == 0:
                # 路线为空，创建初始弧
                new_arcs.append((customer, self.end_depot_idx))
            elif best_position == 0:
                new_arcs.append((customer, route[0][0]))
                new_arcs.extend(route)
            elif best_position >= len(route):
                new_arcs.extend(route)
                new_arcs.append((route[-1][1], customer))
            else:
                new_arcs.extend(route[:best_position])
                from_node = route[best_position-1][1]
                new_arcs.append((from_node, customer))
                new_arcs.append((customer, route[best_position][0]))
                new_arcs.extend(route[best_position:])
            
            solution['truck_route'] = new_arcs
            solution['visited_customers'].add(customer)
            unvisited.remove(customer)
        
        solution['makespan'] = self._calculate_makespan(solution)
        return solution
    
    def _repair_random(self, partial: Dict) -> Dict:
        """随机插入修复"""
        solution = copy.deepcopy(partial)
        
        unvisited = set(self.N) - solution['visited_customers']
        if not unvisited:
            solution['makespan'] = self._calculate_makespan(solution)
            return solution
        
        # 随机顺序插入客户
        unvisited_list = list(unvisited)
        random.shuffle(unvisited_list)
        
        for customer in unvisited_list:
            route = solution['truck_route']
            # 随机选择插入位置
            pos = random.randint(0, len(route))

            new_arcs = []
            if len(route) == 0:
                # 路线为空，创建初始弧
                new_arcs.append((customer, self.end_depot_idx))
            elif pos == 0:
                new_arcs.append((customer, route[0][0]))
                new_arcs.extend(route)
            elif pos >= len(route):
                new_arcs.extend(route)
                new_arcs.append((route[-1][1], customer))
            else:
                new_arcs.extend(route[:pos])
                from_node = route[pos-1][1]
                new_arcs.append((from_node, customer))
                new_arcs.append((customer, route[pos][0]))
                new_arcs.extend(route[pos:])
            
            solution['truck_route'] = new_arcs
            solution['visited_customers'].add(customer)
        
        solution['makespan'] = self._calculate_makespan(solution)
        return solution
    
    def _calculate_makespan(self, solution: Dict) -> float:
        """计算完工时间"""
        route = solution.get('truck_route', [])
        if not route:
            return 0.0
        
        current_time = 0.0
        current_node = 0
        
        for from_node, to_node in route:
            travel_time = self._compute_travel_time(current_node, to_node)
            current_time += travel_time
            
            # 服务时间
            if 1 <= to_node <= self.n_customers:
                current_time += self.service_times.get(to_node, 0.5)
            
            current_node = to_node
        
        return current_time
    
    def _update_operator_weights(self):
        """更新算子权重（简化版）"""
        # 每100次迭代后调整权重
        if self.stats['iterations'] % 100 == 0:
            # 保持权重不变，仅做微小调整
            pass
    
    def print_solution(self, solution: Dict):
        """打印解"""
        print(f"\n{'='*60}")
        print("ALNS SOLUTION")
        print(f"{'='*60}")
        print(f"完工时间: {solution['makespan']:.2f}")
        print(f"访问客户: {len(solution.get('visited_customers', []))}/{self.n_customers}")
        
        print("\n卡车路径:")
        route = solution.get('truck_route', [])
        if route:
            print(f"  仓库 -> 节点 {route[0][0]}")
            for i, (from_node, to_node) in enumerate(route):
                print(f"  节点 {from_node} -> 节点 {to_node}")
            print(f"  节点 {route[-1][1]} -> 仓库")
        
        print(f"\n{'='*60}")
