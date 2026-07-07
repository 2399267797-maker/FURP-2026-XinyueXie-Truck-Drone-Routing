import numpy as np
import random
from typing import List, Tuple, Dict
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.vrp_model import VRPTruckDroneModel, Route, DroneMission

class CollaborativePACO:
    def __init__(self, model: VRPTruckDroneModel, n_ants: int = 30, max_iter: int = 100):
        self.model = model
        self.n_ants = n_ants
        self.max_iter = max_iter
        
        # 论文 5) P-ACO 参数调优：完全按照论文给定的绝对参数
        self.alpha = 1.0
        self.beta = 2.0
        self.rho = 0.15
        self.q0 = 0.5
        self.Q_c = 120.0  # 针对成本目标的沉积量基准
        self.Q_t = 60.0   # 针对延迟目标的沉积量基准
        self.TAU_MAX = 20.0
        self.TAU_MIN = 1.0
        self.TAU_INIT = 10.0
        
        self.n_customers = model.get_number_of_customers()
        self.n_trucks = model.get_number_of_trucks()
        self.n_drones = model.get_number_of_drones()
        
        n_nodes = self.n_customers + 1
        
        # ====== 双目标、双车型独立的 2D 和 3D 信息素矩阵 ======
        self.phero_truck_c = np.full((n_nodes, n_nodes), self.TAU_INIT)
        self.phero_truck_t = np.full((n_nodes, n_nodes), self.TAU_INIT)
        self.phero_drone_c = np.full((n_nodes, n_nodes, n_nodes), self.TAU_INIT)
        self.phero_drone_t = np.full((n_nodes, n_nodes, n_nodes), self.TAU_INIT)
        
        self.pareto_front = []
        self.pareto_solutions = []

        # ==================== 【巨幅提速：静态数组与性能预处理】 ====================
        # 1. 预计算高维距离矩阵 (0: Depot, 1..N: 客户点)，将最内层 O(N) 计算降为 O(1) 内存寻址
        self.dist_matrix = np.zeros((n_nodes, n_nodes))
        nodes_list = [model.depot] + model.customers
        for i in range(n_nodes):
            for j in range(n_nodes):
                self.dist_matrix[i, j] = nodes_list[i].distance_to(nodes_list[j])
                
        # 2. 将频繁读取的对象属性扁平化为 NumPy 纯数数组，彻底切断循环内的 Dot Lookup 耗时
        self.demands = np.array([c.demand for c in model.customers])
        self.tw_start = np.array([c.time_window[0] for c in model.customers])
        self.tw_end = np.array([c.time_window[1] for c in model.customers])
        self.service_times = np.array([c.service_time for c in model.customers])
        
        # 3. 提取核心车型参数
        self.truck_speed = model.trucks[0].speed
        self.truck_capacity = model.trucks[0].capacity
        self.drone_speed = model.get_vehicle_speed('drone')
        self.drone_capacity = model.drones[0].capacity
        self.drone_range = model.drone_range
        self.launch_prep_time = getattr(model, 'launch_prep_time', 0.5)
        self.retrieval_time = getattr(model, 'retrieval_time', 0.5)

    def _construct_solution(self) -> List[Route]:
        """终极修复版 Stage 2：修复数学倒置 Bug 与引入空间就近兜底"""
        routes = []
        remaining = set(range(self.n_customers))
        
        # 严格遵守固定卡车数量
        for truck_id in range(self.n_trucks):
            if not remaining: break
            
            route = Route(vehicle_id=truck_id, vehicle_type='truck')
            current_time = 0.0
            current_load = 0.0
            current_node = 0  # 0 代表 Depot
            
            while remaining:
                candidates = list(remaining)
                action_probs = []
                action_list = []
                
                # --------- 选项 A：卡车直接拜访节点 j ---------
                for j in candidates:
                    demand_j = self.demands[j]
                    if current_load + demand_j <= self.truck_capacity:
                        dist_ij = self.dist_matrix[current_node, j + 1]
                        arr_time_j = current_time + (dist_ij / self.truck_speed)
                        
                        eta_c = 1.0 / (dist_ij + 0.001)
                        penalty_j = max(0.0, arr_time_j - self.tw_end[j])
                        eta_t = 1.0 / (1.0 + penalty_j)
                        
                        tau_c = self.phero_truck_c[current_node, j + 1]
                        tau_t = self.phero_truck_t[current_node, j + 1]
                        
                        prob = ((tau_c + tau_t) ** self.alpha) * ((eta_c + eta_t) ** self.beta)
                        action_probs.append(prob)
                        action_list.append(('truck', j))

                # --------- 选项 B：无人机联合子巡航 (i -> j -> k) ---------
                if route.customers:  # 必须已经有卡车节点，避免负索引
                    for j in candidates:
                        if self.demands[j] > self.drone_capacity: continue
                        if j in route.customers: continue
                        
                        d_ij = self.dist_matrix[current_node, j + 1]
                        if d_ij >= self.drone_range: continue 
                        
                        for k in candidates:
                            if j == k: continue 
                            if k in route.customers: continue
                            
                            demand_k = self.demands[k]
                            if current_load + self.demands[j] + demand_k > self.truck_capacity: continue
                            
                            d_jk = self.dist_matrix[j + 1, k + 1]
                            
                            arr_drone_j = current_time + (d_ij / self.drone_speed)
                            arr_drone_k = arr_drone_j + self.service_times[j] + (d_jk / self.drone_speed) 
                            
                            if (d_ij + d_jk) > self.drone_range: continue
                            
                            d_ik = self.dist_matrix[current_node, k + 1]
                            arr_truck_k = current_time + (d_ik / self.truck_speed) + self.launch_prep_time + self.retrieval_time
                            
                            if arr_drone_k > arr_truck_k: continue
                            
                            sync_time_k = arr_truck_k
                            penalty_j = max(0.0, arr_drone_j - self.tw_end[j])
                            penalty_k = max(0.0, sync_time_k - self.tw_end[k])
                            
                            # 【核心修复】取消倒置的 Savings，直接基于卡车的物理移动意图，引导卡车走向最近的点
                            eta_drone_c = 1.0 / (d_ik + 0.001) 
                            eta_drone_t = 1.0 / (1.0 + penalty_j + penalty_k)
                            
                            tau_dc = self.phero_drone_c[current_node, j + 1, k + 1]
                            tau_dt = self.phero_drone_t[current_node, j + 1, k + 1]
                            
                            prob = ((tau_dc + tau_dt) ** self.alpha) * ((eta_drone_c + eta_drone_t) ** self.beta)
                            action_probs.append(prob)
                            action_list.append(('drone', j, k, sync_time_k))

                # 轮盘赌决策
                total_prob = sum(action_probs)
                if total_prob == 0: break
                
                action_probs = [p / total_prob for p in action_probs]
                if random.random() < self.q0:
                    chosen_idx = np.argmax(action_probs)
                else:
                    chosen_idx = random.choices(range(len(action_list)), weights=action_probs)[0]
                
                chosen_action = action_list[chosen_idx]
                
                if chosen_action[0] == 'truck':
                    _, next_node = chosen_action
                    route.customers.append(next_node)
                    remaining.remove(next_node)
                    current_load += self.demands[next_node]
                    
                    dist = self.dist_matrix[current_node, next_node + 1]
                    current_time += (dist / self.truck_speed)
                    current_time = max(current_time, self.tw_start[next_node]) + self.service_times[next_node]
                    current_node = next_node + 1
                    
                elif chosen_action[0] == 'drone':
                    _, drone_target, truck_target, sync_time = chosen_action
                    
                    launch_idx = len(route.customers) - 1
                    return_idx = launch_idx + 1
                    
                    route.customers.append(truck_target)
                    mission = DroneMission(
                        drone_id=self.n_trucks + (truck_id % self.n_drones), 
                        customer_ids=[drone_target],
                        launch_point=launch_idx,
                        return_point=return_idx
                    )
                    route.drone_missions.append(mission)
                    
                    remaining.remove(drone_target)
                    remaining.remove(truck_target)
                    current_load += self.demands[drone_target] + self.demands[truck_target]
                    current_time = max(sync_time, self.tw_start[truck_target]) + self.service_times[truck_target]
                    current_node = truck_target + 1
            
            if route.customers:
                routes.append(route)
                
        # ==================== 【终极修复：就近贪心兜底 (Nearest-Neighbor Fallback)】 ====================
        # 如果卡车数量用尽，但仍有客户未服务。
        # 绝对不随机硬塞！而是计算剩余客户离哪辆卡车的终点最近，顺滑地追加在尾部。
        while remaining and routes:
            best_dist = float('inf')
            best_route = None
            best_cust = None
            
            for r in routes:
                last_node = r.customers[-1] if r.customers else 0
                for cust_id in remaining:
                    dist = self.dist_matrix[last_node + 1 if r.customers else 0, cust_id + 1]
                    if dist < best_dist:
                        best_dist = dist
                        best_route = r
                        best_cust = cust_id
            
            if best_route is not None:
                best_route.customers.append(best_cust)
                remaining.remove(best_cust)
            else:
                break # 极小概率防御

        # ==================== 后置防御性冲突拦截器 ====================
        for r in routes:
            truck_visited_set = set(r.customers)
            valid_missions = []
            
            for m in r.drone_missions:
                drone_target_cust = m.customer_ids[0]
                if (drone_target_cust not in truck_visited_set) and \
                   (m.launch_point < len(r.customers)) and \
                   (m.return_point < len(r.customers)):
                    valid_missions.append(m)
            
            r.drone_missions = valid_missions
            
        return routes
    def _update_pheromones(self, solutions):
        """优化版 Stage 4：自适应极值标定更新与 MMAS 后置矩阵截断"""
        # 1. 精英解全局同步蒸发
        self.phero_truck_c *= (1 - self.rho)
        self.phero_truck_t *= (1 - self.rho)
        self.phero_drone_c *= (1 - self.rho)
        self.phero_drone_t *= (1 - self.rho)
        
        solutions = list(solutions)
        if not solutions: return
        
        # 获取当前前沿的两端极值，用于动态归一化不同量纲的目标函数
        min_cost = max(0.001, min(o[0] for _, o in solutions))
        min_tard = max(0.001, min(o[1] for _, o in solutions))
        
        # 2. 密集贡献无损累加（拒绝中途截断造成的增量抹杀）
        for route_sol, objectives in solutions:
            cost, tardiness = objectives
            delta_c = self.Q_c * (min_cost / max(0.001, cost))
            delta_t = self.Q_t * (min_tard / max(0.001, tardiness))
            
            for route in route_sol:
                nodes = [0] + [c + 1 for c in route.customers] + [0]
                for i in range(len(nodes) - 1):
                    self.phero_truck_c[nodes[i], nodes[i+1]] += delta_c
                    self.phero_truck_t[nodes[i], nodes[i+1]] += delta_t
                
                for mission in route.drone_missions:
                    i_node = 0 if mission.launch_point == -1 else route.customers[mission.launch_point] + 1
                    j_node = mission.customer_ids[0] + 1
                    k_node = route.customers[mission.return_point] + 1 if mission.return_point < len(route.customers) else 0
                    
                    self.phero_drone_c[i_node, j_node, k_node] += delta_c
                    self.phero_drone_t[i_node, j_node, k_node] += delta_t

        # 3. 统一后置全局 MMAS 阈值硬边界拦截
        self.phero_truck_c = np.clip(self.phero_truck_c, self.TAU_MIN, self.TAU_MAX)
        self.phero_truck_t = np.clip(self.phero_truck_t, self.TAU_MIN, self.TAU_MAX)
        self.phero_drone_c = np.clip(self.phero_drone_c, self.TAU_MIN, self.TAU_MAX)
        self.phero_drone_t = np.clip(self.phero_drone_t, self.TAU_MIN, self.TAU_MAX)

    def _dominates(self, obj1: Tuple[float, float], obj2: Tuple[float, float]) -> bool:
        return (obj1[0] <= obj2[0] and obj1[1] <= obj2[1]) and (obj1[0] < obj2[0] or obj1[1] < obj2[1])

    def solve(self) -> Tuple[List[List[Route]], List[Tuple[float, float]]]:
        for iteration in range(self.max_iter):
            current_sols = []
            
            for _ in range(self.n_ants):
                route = self._construct_solution()
                cost, _ = self.model.evaluate_solution(route)
                tardiness = self.model.calculate_pure_tardiness(route)
                current_sols.append((route, (cost, tardiness)))
            
            # 精英帕累托档案动态更新
            for solution, objectives in current_sols:
                dominated = False
                new_pareto, new_solutions = [], []
                
                for ex_sol, ex_obj in zip(self.pareto_solutions, self.pareto_front):
                    if self._dominates(ex_obj, objectives):
                        dominated = True
                        break
                    elif not self._dominates(objectives, ex_obj):
                        new_pareto.append(ex_obj)
                        new_solutions.append(ex_sol)
                        
                if not dominated:
                    if not any(abs(eo[0]-objectives[0]) < 1e-4 and abs(eo[1]-objectives[1]) < 1e-4 for eo in new_pareto):
                        new_pareto.append(objectives)
                        new_solutions.append(solution)
                        self.pareto_front = new_pareto
                        self.pareto_solutions = new_solutions
                        
            # 【Bug 修复：帕累托前沿多样性均匀保持机制】
            # 解决切片引发的历史陈旧解“占着坑不走”导致前沿停滞的问题。按主目标排序并实施等间距均匀网格抽样
            if len(self.pareto_front) > 50:
                combined = list(zip(self.pareto_front, self.pareto_solutions))
                combined.sort(key=lambda x: x[0][0])  # 按成本目标线性排序
                
                # 均匀提取 50 个分散在全前沿的精英解（保留边界极端解与中间过渡解）
                indices = np.linspace(0, len(combined) - 1, 50, dtype=int)
                self.pareto_front = [combined[idx][0] for idx in indices]
                self.pareto_solutions = [combined[idx][1] for idx in indices]
            
            if self.pareto_solutions:
                self._update_pheromones(zip(self.pareto_solutions, self.pareto_front))
                
            if (iteration + 1) % 10 == 0:
                print(f"[P-ACO] Iteration {iteration + 1}/{self.max_iter} - Elite Archive Size: {len(self.pareto_front)}")
                
        return self.pareto_solutions, self.pareto_front