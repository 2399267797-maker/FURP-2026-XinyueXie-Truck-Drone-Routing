import numpy as np
import random
from typing import List, Tuple, Dict
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.vrp_model import VRPTruckDroneModel, Route, DroneMission

class CollaborativePACOALNS:
    def __init__(self, model: VRPTruckDroneModel, n_ants: int = 30, max_iter: int = 100, alns_iter: int = 15):
        self.model = model
        self.n_ants = n_ants
        self.max_iter = max_iter
        self.alns_iter = alns_iter  # 每只蚂蚁构建完解后，进行 ALNS 微观迭代的次数
        
        # P-ACO 核心参数
        self.alpha = 1.0
        self.beta = 2.0
        self.rho = 0.15
        self.q0 = 0.5
        self.Q_c = 120.0  
        self.Q_t = 60.0   
        self.TAU_MAX = 20.0
        self.TAU_MIN = 1.0
        self.TAU_INIT = 10.0
        
        self.n_customers = model.get_number_of_customers()
        self.n_trucks = model.get_number_of_trucks()
        self.n_drones = model.get_number_of_drones()
        
        n_nodes = self.n_customers + 1
        
        # 信息素矩阵
        self.phero_truck_c = np.full((n_nodes, n_nodes), self.TAU_INIT)
        self.phero_truck_t = np.full((n_nodes, n_nodes), self.TAU_INIT)
        self.phero_drone_c = np.full((n_nodes, n_nodes, n_nodes), self.TAU_INIT)
        self.phero_drone_t = np.full((n_nodes, n_nodes, n_nodes), self.TAU_INIT)
        
        self.pareto_front = []
        self.pareto_solutions = []

        # ==================== 静态数据预处理（高能提速核心） ====================
        self.dist_matrix = np.zeros((n_nodes, n_nodes))
        nodes_list = [model.depot] + model.customers
        for i in range(n_nodes):
            for j in range(n_nodes):
                self.dist_matrix[i, j] = nodes_list[i].distance_to(nodes_list[j])
                
        self.demands = np.array([c.demand for c in model.customers])
        self.tw_start = np.array([c.time_window[0] for c in model.customers])
        self.tw_end = np.array([c.time_window[1] for c in model.customers])
        self.service_times = np.array([c.service_time for c in model.customers])
        
        self.truck_speed = model.trucks[0].speed
        self.truck_capacity = model.trucks[0].capacity
        self.drone_speed = model.get_vehicle_speed('drone')
        self.drone_capacity = model.drones[0].capacity
        self.drone_range = model.drone_range
        self.launch_prep_time = getattr(model, 'launch_prep_time', 0.5)
        self.retrieval_time = getattr(model, 'retrieval_time', 0.5)

    def _clone_routes(self, routes: List[Route]) -> List[Route]:
        """轻量化高效对象复制，彻底干掉 copy.deepcopy 的巨大耗时"""
        cloned = []
        for r in routes:
            new_r = Route(vehicle_id=r.vehicle_id, vehicle_type='truck', customers=r.customers.copy())
            for m in r.drone_missions:
                new_m = DroneMission(
                    drone_id=m.drone_id,
                    customer_ids=m.customer_ids.copy(),
                    launch_point=m.launch_point,
                    return_point=m.return_point
                )
                new_r.drone_missions.append(new_m)
            cloned.append(new_r)
        return cloned

    def _surgical_destroy(self, routes: List[Route], target_custs: set) -> List[Route]:
        """ALNS 破坏算子：精确切除指定客户并平滑重构无人机索引映射"""
        destroyed = []
        for r in routes:
            new_r = Route(vehicle_id=r.vehicle_id, vehicle_type='truck')
            idx_map = {}
            new_idx = 0
            
            # 1. 过滤并重构卡车路径
            for old_idx, c in enumerate(r.customers):
                if c not in target_custs:
                    new_r.customers.append(c)
                    idx_map[old_idx] = new_idx
                    new_idx += 1
            
            # 2. 过滤并重配无人机任务
            for m in r.drone_missions:
                d_target = m.customer_ids[0]
                if d_target not in target_custs:
                    # 只有当发射点和回收点都未被破坏时，保留该协同任务并更新时空索引
                    if m.launch_point in idx_map and m.return_point in idx_map:
                        new_m = DroneMission(
                            drone_id=m.drone_id,
                            customer_ids=[d_target],
                            launch_point=idx_map[m.launch_point],
                            return_point=idx_map[m.return_point]
                        )
                        new_r.drone_missions.append(new_m)
            destroyed.append(new_r)
        return destroyed

    def _greedy_repair(self, routes: List[Route], removed_custs: List[int]) -> List[Route]:
        """ALNS 修复算子：增量代价最小化插入，带有时空索引联动位移"""
        for c in removed_custs:
            best_dist_increase = float('inf')
            best_route_idx = 0
            best_insert_pos = 0
            
            # 遍历所有活跃卡车路线及所有可插入间隙
            for r_idx, r in enumerate(routes):
                for pos in range(len(r.customers) + 1):
                    prev_node = 0 if pos == 0 else r.customers[pos - 1] + 1
                    next_node = 0 if pos == len(r.customers) else r.customers[pos] + 1
                    
                    increase = (self.dist_matrix[prev_node, c + 1] + 
                                self.dist_matrix[c + 1, next_node] - 
                                self.dist_matrix[prev_node, next_node])
                    
                    if increase < best_dist_increase:
                        best_dist_increase = increase
                        best_route_idx = r_idx
                        best_insert_pos = pos
            
            # 联动位移更新：插入位置之后的无人机 launch/return 索引全部后移一位
            target_route = routes[best_route_idx]
            target_route.customers.insert(best_insert_pos, c)
            for m in target_route.drone_missions:
                if m.launch_point >= best_insert_pos:
                    m.launch_point += 1
                if m.return_point >= best_insert_pos:
                    m.return_point += 1
                    
        return routes

    def _alns_local_search(self, routes: List[Route]) -> List[Route]:
        """ALNS 局部搜索核心外壳：控制微观毁灭与重生迭代"""
        best_routes = self._clone_routes(routes)
        best_cost, _ = self.model.evaluate_solution(best_routes)
        best_tard = self.model.calculate_pure_tardiness(best_routes)
        
        # 统计当前解中服务的总客户数，确定毁灭规模（15%）
        total_custs = sum(len(r.customers) + sum(len(m.customer_ids) for m in r.drone_missions) for r in routes)
        if total_custs < 4: return routes
        k = max(1, int(total_custs * 0.15))
        
        current_routes = self._clone_routes(routes)
        
        for _ in range(self.alns_iter):
            # 收集当前路线中所有的客户名单
            all_served = []
            for r in current_routes:
                all_served.extend(r.customers)
                for m in r.drone_missions:
                    all_served.extend(m.customer_ids)
            
            if len(all_served) < k: break
            
            # 随机选择破坏目标
            target_custs = set(random.sample(all_served, k))
            
            # 毁灭与重生
            working_routes = self._surgical_destroy(current_routes, target_custs)
            working_routes = self._greedy_repair(working_routes, list(target_custs))
            
            # 计算多目标适应度
            rep_cost, _ = self.model.evaluate_solution(working_routes)
            rep_tard = self.model.calculate_pure_tardiness(working_routes)
            
            # 多目标接受准则：如果新解支配了当前解，或者与当前解互不支配（帕累托等价），则接受用于继续扰动
            is_dominated_by_current = (best_cost <= rep_cost and best_tard <= rep_tard) and (best_cost < rep_cost or best_tard < rep_tard)
            
            if not is_dominated_by_current:
                current_routes = self._clone_routes(working_routes)
                # 如果打破了历史最优边界，直接替换最优解
                if (rep_cost <= best_cost and rep_tard <= best_tard) and (rep_cost < best_cost or rep_tard < best_tard):
                    best_routes = self._clone_routes(working_routes)
                    best_cost, best_tard = rep_cost, rep_tard
                    
        return best_routes

    def _construct_solution(self) -> List[Route]:
        """Stage 2：由信息素矩阵和启发式信息指导的蚂蚁宏观探路逻辑"""
        routes = []
        remaining = set(range(self.n_customers))
        
        for truck_id in range(self.n_trucks):
            if not remaining: break
            
            route = Route(vehicle_id=truck_id, vehicle_type='truck')
            current_time = 0.0
            current_load = 0.0
            current_node = 0  
            
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
                if route.customers:  
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
                            
                            eta_drone_c = 1.0 / (d_ik + 0.001) 
                            eta_drone_t = 1.0 / (1.0 + penalty_j + penalty_k)
                            
                            tau_dc = self.phero_drone_c[current_node, j + 1, k + 1]
                            tau_dt = self.phero_drone_t[current_node, j + 1, k + 1]
                            
                            prob = ((tau_dc + tau_dt) ** self.alpha) * ((eta_drone_c + eta_drone_t) ** self.beta)
                            action_probs.append(prob)
                            action_list.append(('drone', j, k, sync_time_k))

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
                
        # 贪心就近兜底
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
                break

        # 后置防御性冲突拦截器
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
        """Stage 4：由 ALNS 深度优化后的高质量解集体同步更新信息素"""
        self.phero_truck_c *= (1 - self.rho)
        self.phero_truck_t *= (1 - self.rho)
        self.phero_drone_c *= (1 - self.rho)
        self.phero_drone_t *= (1 - self.rho)
        
        solutions = list(solutions)
        if not solutions: return
        
        min_cost = max(0.001, min(o[0] for _, o in solutions))
        min_tard = max(0.001, min(o[1] for _, o in solutions))
        
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
                # 1. 蚂蚁进行宏观概率路径构建
                route = self._construct_solution()
                
                # 2. 【核心注入】调用 ALNS 算子对单蚁生成的路径进行微观大邻域搜索微调
                route = self._alns_local_search(route)
                
                cost, _ = self.model.evaluate_solution(route)
                tardiness = self.model.calculate_pure_tardiness(route)
                current_sols.append((route, (cost, tardiness)))
            
            # 精英帕累托档案维护
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
                        
            # 帕累托前沿多样性网格均匀抽样保持机制
            if len(self.pareto_front) > 50:
                combined = list(zip(self.pareto_front, self.pareto_solutions))
                combined.sort(key=lambda x: x[0][0])  
                
                indices = np.linspace(0, len(combined) - 1, 50, dtype=int)
                self.pareto_front = [combined[idx][0] for idx in indices]
                self.pareto_solutions = [combined[idx][1] for idx in indices]
            
            if self.pareto_solutions:
                self._update_pheromones(zip(self.pareto_solutions, self.pareto_front))
                
            if (iteration + 1) % 10 == 0:
                print(f"[P-ACO+ALNS] Iteration {iteration + 1}/{self.max_iter} - Elite Archive Size: {len(self.pareto_front)}")
                
        return self.pareto_solutions, self.pareto_front