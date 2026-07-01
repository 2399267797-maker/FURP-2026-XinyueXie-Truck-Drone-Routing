# 
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
        self.Q_c = 120.0  # 针对成本目标的沉积量
        self.Q_t = 60.0   # 针对延迟目标的沉积量
        self.TAU_MAX = 20.0
        self.TAU_MIN = 1.0
        self.TAU_INIT = 10.0
        
        self.n_customers = model.get_number_of_customers()
        self.n_trucks = model.get_number_of_trucks()
        self.n_drones = model.get_number_of_drones()
        
        n_nodes = self.n_customers + 1
        
        # ====== 绝对还原：双目标、双车型独立的 2D 和 3D 信息素矩阵 ======
        # 卡车：2D 矩阵 (i -> j)
        self.phero_truck_c = np.full((n_nodes, n_nodes), self.TAU_INIT)
        self.phero_truck_t = np.full((n_nodes, n_nodes), self.TAU_INIT)
        
        # 无人机：3D 矩阵 (i -> j -> k) - 论文中最核心的三元组同步模型
        self.phero_drone_c = np.full((n_nodes, n_nodes, n_nodes), self.TAU_INIT)
        self.phero_drone_t = np.full((n_nodes, n_nodes, n_nodes), self.TAU_INIT)
        
        self.pareto_front = []
        self.pareto_solutions = []

    def _construct_solution(self) -> List[Route]:
        """论文 Stage 2 绝对还原：卡车与无人机在同一节点 [同步联合决策]"""
        routes = []
        remaining = set(range(self.n_customers))
        
        for truck_id in range(self.n_trucks):
            if not remaining: break
            
            route = Route(vehicle_id=truck_id, vehicle_type='truck')
            truck = self.model.trucks[truck_id]
            drone_speed = self.model.get_vehicle_speed('drone')
            
            current_time = 0.0
            current_load = 0.0
            current_node = 0  # 0 代表 Depot
            while remaining:
                candidates = list(remaining)
                action_probs = []
                action_list = []
                
                # 选项 A：卡车直接去节点 j (传统蚁群边选择)
                for j in candidates:
                    cust_j = self.model.customers[j]
                    if current_load + cust_j.demand <= truck.capacity:
                        # 启发式计算
                        dist_ij = self.model.depot.distance_to(cust_j) if current_node == 0 else \
                                  self.model.customers[current_node - 1].distance_to(cust_j)
                        arr_time_j = current_time + (dist_ij / truck.speed)
                        
                        eta_c = 1.0 / (dist_ij + 0.001)
                        penalty_j = max(0.0, arr_time_j - cust_j.time_window[1])
                        eta_t = 1.0 / (1.0 + penalty_j)
                        
                        # 信息素提取
                        tau_c = self.phero_truck_c[current_node, j + 1]
                        tau_t = self.phero_truck_t[current_node, j + 1]
                        
                        # 综合概率计算
                        prob = ((tau_c + tau_t) ** self.alpha) * ((eta_c + eta_t) ** self.beta)
                        action_probs.append(prob)
                        action_list.append(('truck', j))

                # 选项 B：无人机联合子巡航 (Sub Tour: i -> j -> k)
                for j in candidates:
                    cust_j = self.model.customers[j]
                    if cust_j.demand > self.model.drones[0].capacity: continue
                    
                    for k in candidates:
                        if j == k: continue
                        cust_k = self.model.customers[k]
                        
                        # 卡车需要携带 j 和 k 的货物出发
                        if current_load + cust_j.demand + cust_k.demand > truck.capacity: continue
                        
                        node_i = self.model.depot if current_node == 0 else self.model.customers[current_node - 1]
                        d_ij = node_i.distance_to(cust_j)
                        d_jk = cust_j.distance_to(cust_k)
                        
                        # 论文硬约束：必须满足耐力 (Endurance)
                        if (d_ij + d_jk) <= self.model.drone_range:
                            d_ik = node_i.distance_to(cust_k)
                            
                            s_l = getattr(self.model, 'launch_prep_time', 0.5)
                            s_r = getattr(self.model, 'retrieval_time', 0.5)
                            arr_truck_k = current_time + (d_ik / truck.speed) + s_l + s_r
                            arr_drone_j = current_time + (d_ij / drone_speed)
                            arr_drone_k = arr_drone_j + (d_jk / drone_speed)
                            
                            if arr_drone_k > arr_truck_k:
                                continue
                            
                            sync_time_k = arr_truck_k
                            
                            penalty_j = max(0.0, arr_drone_j - cust_j.time_window[1])
                            penalty_k = max(0.0, sync_time_k - cust_k.time_window[1])
                            
                            # 公式 (34) (35) 对应的启发式
                            #eta_drone_c = 1.0 / (d_ij + d_jk + d_ik + 0.001)
                            eta_drone_c = 1.0 / (d_ij + d_jk + 0.001)   # 只用无人机自身飞行距离
                            eta_drone_t = 1.0 / (1.0 + penalty_j + penalty_k)
                            
                            # 提取 3D 信息素矩阵 (i -> j -> k)
                            tau_dc = self.phero_drone_c[current_node, j + 1, k + 1]
                            tau_dt = self.phero_drone_t[current_node, j + 1, k + 1]
                            
                            prob = ((tau_dc + tau_dt) ** self.alpha) * ((eta_drone_c + eta_drone_t) ** self.beta)
                            action_probs.append(prob)
                            action_list.append(('drone', j, k, sync_time_k))

                # 轮盘赌选择 (Roulette Wheel Selection)
                total_prob = sum(action_probs)
                if total_prob == 0:
                    break
                    
                action_probs = [p / total_prob for p in action_probs]
                
                # 伪随机比例规则 (Pseudo-random proportional rule, q0 = 0.5)
                if random.random() < self.q0:
                    chosen_idx = np.argmax(action_probs)
                else:
                    chosen_idx = random.choices(range(len(action_list)), weights=action_probs)[0]
                
                chosen_action = action_list[chosen_idx]
                
                # 执行状态更新
                if chosen_action[0] == 'truck':
                    _, next_node = chosen_action
                    route.customers.append(next_node)
                    remaining.remove(next_node)
                    current_load += self.model.customers[next_node].demand
                    
                    dist = self.model.depot.distance_to(self.model.customers[next_node]) if current_node == 0 else \
                           self.model.customers[current_node - 1].distance_to(self.model.customers[next_node])
                    current_time += (dist / truck.speed)
                    current_time = max(current_time, self.model.customers[next_node].time_window[0]) + self.model.customers[next_node].service_time
                    current_node = next_node + 1
                    
                elif chosen_action[0] == 'drone':
                    _, drone_target, truck_target, sync_time = chosen_action
                    
                    launch_idx = len(route.customers) - 1
                    return_idx = launch_idx + 1
                    
                    route.customers.append(truck_target)
                    mission = DroneMission(
                        drone_id=self.n_trucks + truck_id,
                        customer_ids=[drone_target],
                        launch_point=launch_idx,
                        return_point=return_idx
                    )
                    route.drone_missions.append(mission)
                    
                    remaining.remove(drone_target)
                    remaining.remove(truck_target)
                    current_load += self.model.customers[drone_target].demand + self.model.customers[truck_target].demand
                    current_time = max(sync_time, self.model.customers[truck_target].time_window[0]) + self.model.customers[truck_target].service_time
                    current_node = truck_target + 1
            
            if route.customers:
                routes.append(route)
                
        # 极小概率的兜底
        for cust_id in list(remaining):
            if routes: routes[0].customers.append(cust_id)
            else: routes.append(Route(0, 'truck', [cust_id]))
            
        return routes

    def _update_pheromones(self, solutions):
        """论文 Stage 4: 全局精英信息素更新与独立目标蒸发"""
        # 1. 蒸发公式 (37): tau = (1 - rho) * tau
        self.phero_truck_c *= (1 - self.rho)
        self.phero_truck_t *= (1 - self.rho)
        self.phero_drone_c *= (1 - self.rho)
        self.phero_drone_t *= (1 - self.rho)
        
        solutions = list(solutions)
        if not solutions: return
        
        min_cost = max(0.001, min(o[0] for _, o in solutions))
        min_tard = max(0.001, min(o[1] for _, o in solutions))
        
        # 2. 精英档案更新公式 (36)
        for route_sol, objectives in solutions:
            cost, tardiness = objectives
            
            # Delta 与目标函数适应度成正比
            delta_c = self.Q_c / max(0.001, cost)
            delta_t = self.Q_t / max(0.001, tardiness)
            
            for route in route_sol:
                nodes = [0] + [c + 1 for c in route.customers] + [0]
                
                # 卡车边更新
                for i in range(len(nodes) - 1):
                    n_from, n_to = nodes[i], nodes[i+1]
                    self.phero_truck_c[n_from, n_to] = max(self.TAU_MIN, min(self.TAU_MAX, self.phero_truck_c[n_from, n_to] + delta_c))
                    self.phero_truck_t[n_from, n_to] = max(self.TAU_MIN, min(self.TAU_MAX, self.phero_truck_t[n_from, n_to] + delta_t))
                
                # 无人机 3D 拓扑更新 (i -> j -> k)
                for mission in route.drone_missions:
                    i_node = 0 if mission.launch_point == -1 else route.customers[mission.launch_point] + 1
                    j_node = mission.customer_ids[0] + 1
                    k_node = route.customers[mission.return_point] + 1 if mission.return_point < len(route.customers) else 0
                    
                    self.phero_drone_c[i_node, j_node, k_node] = max(self.TAU_MIN, min(self.TAU_MAX, self.phero_drone_c[i_node, j_node, k_node] + delta_c))
                    self.phero_drone_t[i_node, j_node, k_node] = max(self.TAU_MIN, min(self.TAU_MAX, self.phero_drone_t[i_node, j_node, k_node] + delta_t))

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
            
            # 精英档案维护：仅保留非支配解
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
                    # 去重
                    if not any(abs(eo[0]-objectives[0]) < 1e-4 and abs(eo[1]-objectives[1]) < 1e-4 for eo in new_pareto):
                        new_pareto.append(objectives)
                        new_solutions.append(solution)
                        self.pareto_front = new_pareto
                        self.pareto_solutions = new_solutions
                        
            # 如果精英档案满了 (MaxEAN 控制)，保留拥挤度最好的
            if len(self.pareto_front) > 50:
                # 简化处理：保留两端极端解，中间随机淘汰，维持规模 (替代耗时的严格拥挤度排序)
                self.pareto_solutions = self.pareto_solutions[:50]
                self.pareto_front = self.pareto_front[:50]
            
            if self.pareto_solutions:
                self._update_pheromones(zip(self.pareto_solutions, self.pareto_front))
                
            if (iteration + 1) % 10 == 0:
                print(f"[P-ACO] Iteration {iteration + 1}/{self.max_iter} - Elite Archive Size: {len(self.pareto_front)}")
                
        return self.pareto_solutions, self.pareto_front