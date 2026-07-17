import numpy as np
import random
import math
from typing import List, Tuple, Dict, Optional
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.vrp_model import VRPTruckDroneModel, Route, DroneMission

# =============================================================================
# PACO+ALNS v2.0 — 2026-07-15 (updated 2026-07-16)
# Adaptive parameter scaling based on problem size (n_customers)
#   - n_ants, alns_iter, tau_init, destroy_ratio scale automatically
#   - Construction-phase drone embedding with pheromone learning
#   - Soft wait constraint: truck can wait up to 3 min for drone
#   - Savings-based heuristic with wait penalty in eta_drone_t
# =============================================================================

class ScaleAdaptiveParams:
    """参数自适应缩放，适用于任意规模客户数。

    缩放原则：
    - n_ants: 随 sqrt(n) 亚线性增长，大规模问题收益递减
    - Q_c/Q_t: 随 n^0.6 增长，因为 cost/tardiness 数值随规模增大，
               Q 必须增大才能维持信息素更新信号强度
    - alns_iter/destroy_ratio: 常数，与问题规模无关
    - TAU 边界: 常数，信息素绝对值与问题规模无关
    """
    @staticmethod
    def compute(n_customers: int) -> dict:
        # 蚂蚁数：sqrt(n) 缩放，25c→30, 100c→50, 200c→70
        n_ants = max(30, min(100, int(5 * n_customers ** 0.5)))

        # ALNS 微迭代：常数，不随规模变化
        alns_iter = 15

        # 破坏比例：常数，15% 客户被破坏重组
        destroy_ratio = 0.15

        # 信息素边界：常数
        tau_max = 20.0
        tau_min = 1.0
        tau_init = 10.0

        # Q 因子：随 n^0.6 增长，维持信号强度
        # 25c→120/60, 100c→276/138, 200c→418/209
        scale = (n_customers / 25.0) ** 0.6
        q_c = max(120.0, int(120.0 * scale))
        q_t = max(60.0, int(60.0 * scale))

        # 帕累托档案容量：随规模增长，有上限
        archive_capacity = max(30, min(100, int(n_customers * 0.5)))

        return {
            'n_ants': n_ants,
            'alns_iter': alns_iter,
            'destroy_ratio': destroy_ratio,
            'TAU_MAX': tau_max,
            'TAU_MIN': tau_min,
            'TAU_INIT': tau_init,
            'Q_c': q_c,
            'Q_t': q_t,
            'archive_capacity': archive_capacity,
        }

class CollaborativePACOALNS:
    def __init__(self, model: VRPTruckDroneModel, n_ants: Optional[int] = None,
                 max_iter: int = 100, alns_iter: Optional[int] = None):
        self.model = model
        self.n_customers = model.get_number_of_customers()
        self.n_trucks = model.get_number_of_trucks()
        self.n_drones = model.get_number_of_drones()
        
        # --- Adaptive parameter scaling ---
        p = ScaleAdaptiveParams.compute(self.n_customers)
        self.n_ants = n_ants if n_ants is not None else p['n_ants']
        self.max_iter = max_iter
        self.alns_iter = alns_iter if alns_iter is not None else p['alns_iter']
        self.destroy_ratio = p['destroy_ratio']
        
        # P-ACO 核心参数
        self.alpha = 1.0
        self.beta = 2.0
        self.rho = 0.15
        self.q0 = 0.5
        self.Q_c = p['Q_c']
        self.Q_t = p['Q_t']
        self.TAU_MAX = p['TAU_MAX']
        self.TAU_MIN = p['TAU_MIN']
        self.TAU_INIT = p['TAU_INIT']
        self.archive_capacity = p['archive_capacity']
        
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
        
        # ==================== 初始日志 ====================
        print(f"[PACO+ALNS v2.0] {self.n_customers}c | ants={self.n_ants} "
              f"alns_iter={self.alns_iter} destroy_ratio={self.destroy_ratio:.2f} "
              f"tau_init={self.TAU_INIT:.1f} Q_c={self.Q_c:.0f} "
              f"archive={self.archive_capacity}")

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

    def _drone_aware_repair(self, routes: List[Route], removed_custs: List[int]) -> List[Route]:
        """ALNS 修复算子：无人机感知修复，同时考虑卡车插入和无人机插入选项。

        对每个待插入客户，评估两种选项：
        1. 卡车插入：在当前路线中找距离增量最小的位置（原逻辑）
        2. 无人机插入：在相邻客户 (i, k) 之间，将客户 j 作为无人机任务插入，
           卡车直接从 i 到 k 不走 j，节省距离 d_ij + d_jk - d_ik

        评分使用复合指标：距离增量 + 延迟惩罚，选择得分最低的选项。
        """
        for c in removed_custs:
            best_score = float('inf')
            best_action = None  # ('truck', route_idx, pos) or ('drone', route_idx, launch_pos, return_pos)

            for r_idx, r in enumerate(routes):
                # ── 选项 A：卡车插入 ──
                for pos in range(len(r.customers) + 1):
                    prev_node = 0 if pos == 0 else r.customers[pos - 1] + 1
                    next_node = 0 if pos == len(r.customers) else r.customers[pos] + 1

                    increase = (self.dist_matrix[prev_node, c + 1] +
                                self.dist_matrix[c + 1, next_node] -
                                self.dist_matrix[prev_node, next_node])

                    # 卡车插入评分 = 距离增量（纯距离）
                    if increase < best_score:
                        best_score = increase
                        best_action = ('truck', r_idx, pos)

                # ── 选项 B：无人机插入 ──
                # 在相邻客户 (i, k) 之间插入无人机任务 j=c
                if len(r.customers) >= 2:
                    for pos in range(len(r.customers) - 1):
                        # 跳过容量检查
                        if self.demands[c] > self.drone_capacity:
                            continue

                        i_node = r.customers[pos] + 1
                        j_node = c + 1
                        k_node = r.customers[pos + 1] + 1

                        d_ij = self.dist_matrix[i_node, j_node]
                        d_jk = self.dist_matrix[j_node, k_node]
                        d_ik = self.dist_matrix[i_node, k_node]

                        # 航程检查
                        if d_ij >= self.drone_range or (d_ij + d_jk) > self.drone_range:
                            continue

                        # 有时间约束检查（no-wait-for-drone 软约束）
                        arr_drone_k = (d_ij + d_jk) / self.drone_speed + self.service_times[c]
                        arr_truck_k = d_ik / self.truck_speed + self.launch_prep_time + self.retrieval_time
                        wait_time = max(0.0, arr_drone_k - arr_truck_k)
                        if wait_time > 3.0:
                            continue

                        # 距离节省 = d_ij + d_jk - d_ik（正数 = 节省）
                        savings = d_ij + d_jk - d_ik
                        if savings < 0.1:
                            continue

                        # 无人机插入评分 = -(节省距离) + 延迟惩罚
                        # 延迟惩罚：估计无人机到达 j 的时间窗口违反
                        drone_arrival_j = d_ij / self.drone_speed
                        drone_tardiness_j = max(0.0, drone_arrival_j - self.tw_end[c])
                        penalty = 5.0 * drone_tardiness_j

                        drone_score = -savings + penalty
                        if drone_score < best_score:
                            best_score = drone_score
                            best_action = ('drone', r_idx, pos, pos + 1)

            # ── 执行最佳动作 ──
            if best_action is None:
                continue  # 异常情况：跳过此客户

            if best_action[0] == 'truck':
                _, r_idx, pos = best_action
                target_route = routes[r_idx]
                target_route.customers.insert(pos, c)
                for m in target_route.drone_missions:
                    if m.launch_point >= pos:
                        m.launch_point += 1
                    if m.return_point >= pos:
                        m.return_point += 1
            else:
                _, r_idx, launch_pos, return_pos = best_action
                target_route = routes[r_idx]
                # 添加无人机任务（不修改卡车路线客户列表）
                mission = DroneMission(
                    drone_id=self.n_trucks + (r_idx % self.n_drones),
                    customer_ids=[c],
                    launch_point=launch_pos,
                    return_point=return_pos
                )
                target_route.drone_missions.append(mission)

        return routes

    def _post_repair_drone_optimization(self, routes: List[Route]) -> List[Route]:
        """After greedy repair, convert truck visits to drone missions where beneficial.

        Scans each route for consecutive pairs (j, k) where j can be served by a drone
        launched from the previous customer (i) and recovered at k. This re-introduces
        drone missions that the greedy repair naturally destroys.
        """
        for truck_id, route in enumerate(routes):
            if not route.customers or len(route.customers) < 3:
                continue

            idx = 0
            while idx < len(route.customers) - 1:
                j = route.customers[idx]       # potential drone target
                k = route.customers[idx + 1]   # potential recovery point

                # Skip the first customer (no launch point before it)
                if idx == 0:
                    idx += 1
                    continue

                # Drone capacity check
                if self.demands[j] > self.drone_capacity:
                    idx += 1
                    continue

                # Launch point: customer at position idx-1
                i_node = route.customers[idx - 1] + 1
                j_node = j + 1
                k_node = k + 1

                d_ij = self.dist_matrix[i_node, j_node]
                d_jk = self.dist_matrix[j_node, k_node]
                d_ik = self.dist_matrix[i_node, k_node]

                # Range check
                if d_ij >= self.drone_range or (d_ij + d_jk) > self.drone_range:
                    idx += 1
                    continue

                # Savings check (same threshold as construction phase)
                savings = d_ij + d_jk - d_ik
                if savings < 0.1:
                    idx += 1
                    continue

                # Time constraint check (same logic as construction phase)
                arr_drone_k = ((d_ij + d_jk) / self.drone_speed
                               + self.service_times[j])
                arr_truck_k = (d_ik / self.truck_speed
                               + self.launch_prep_time + self.retrieval_time)
                wait_time = max(0.0, arr_drone_k - arr_truck_k)
                if wait_time > 3.0:
                    idx += 1
                    continue

                # --- Convert to drone mission ---
                # Remove j from truck route first (k shifts to position idx)
                route.customers.pop(idx)

                # 修复：pop 后所有已有无人机任务的索引需同步位移
                # 注意：用 > 不用 >=，因为 == idx 的引用客户已被移除，该任务无效应删除
                valid_missions = []
                for m in route.drone_missions:
                    if m.launch_point > idx:
                        m.launch_point -= 1
                    if m.return_point > idx:
                        m.return_point -= 1
                    # 如果 launch_point==idx 或 return_point==idx，引用客户已不存在，丢弃该任务
                    if m.launch_point != idx and m.return_point != idx:
                        valid_missions.append(m)
                route.drone_missions = valid_missions

                # Create drone mission referencing the modified route indices
                mission = DroneMission(
                    drone_id=self.n_trucks + (truck_id % self.n_drones),
                    customer_ids=[j],
                    launch_point=idx - 1,  # i is now at idx-1
                    return_point=idx       # k is now at idx
                )
                route.drone_missions.append(mission)

                # 关键修复：跳过 k（回收点），确保无人机先回收再发射
                # 上一任务：发射点=i(idx-1), 无人机服务=j, 回收点=k(idx)
                # 下一轮从 k 之后开始检查，发射点变为 k(idx-1)
                idx += 1

        return routes

    def _alns_local_search(self, routes: List[Route]) -> List[Route]:
        """ALNS 局部搜索核心外壳：控制微观毁灭与重生迭代"""
        best_routes = self._clone_routes(routes)
        best_cost, _ = self.model.evaluate_solution(best_routes)
        best_tard = self.model.calculate_pure_tardiness(best_routes)
        
        # 统计当前解中服务的总客户数，确定毁灭规模（自适应比例）
        total_custs = sum(len(r.customers) + sum(len(m.customer_ids) for m in r.drone_missions) for r in routes)
        if total_custs < 4: return routes
        k = max(1, int(total_custs * self.destroy_ratio))
        
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
            # 无人机感知修复：同时考虑卡车插入和无人机插入选项
            working_routes = self._drone_aware_repair(working_routes, list(target_custs))
            # 后处理：尝试将连续的卡车访问转为无人机任务（补充修复可能遗漏的无人机机会）
            working_routes = self._post_repair_drone_optimization(working_routes)
            # 跨路线 relocate：消除路径交叉，形成花瓣形路线
            working_routes = self._inter_route_relocate(working_routes)
            
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
        """Stage 2：信息素引导的构造阶段，嵌入无人机选项供蚂蚁选择学习"""
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
                            
                            # Soft constraint: truck can wait up to 3 min, wait time added to penalty
                            if arr_drone_k > arr_truck_k:
                                wait_time = arr_drone_k - arr_truck_k
                                if wait_time > 3.0:
                                    continue
                                sync_time_k = arr_drone_k
                            else:
                                wait_time = 0.0
                                sync_time_k = arr_truck_k
                            
                            # Savings: truck distance saved by not going to j
                            savings = (d_ij + d_jk - d_ik)
                            if savings < 0.1: continue  # low threshold, pheromones learn what's good
                            
                            penalty_j = max(0.0, arr_drone_j - self.tw_end[j])
                            penalty_k = max(0.0, sync_time_k - self.tw_end[k])
                            # Add wait time penalty so the algorithm learns to avoid long waits
                            wait_penalty = wait_time * 2.0
                            
                            # Heuristic based on actual savings (not just truck distance)
                            eta_drone_c = savings / (d_ik + 0.001) 
                            eta_drone_t = 1.0 / (1.0 + penalty_j + penalty_k + wait_penalty)
                            
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

    def _two_opt_route(self, route):
        """2-opt local search on a single route.
        - Skip routes with drone_missions to avoid index misalignment
        - Use distance delta for fast evaluation (no full cost recalculation)
        - Check time window feasibility after accepting swap
        - First-improvement strategy for efficiency
        """
        if not route.customers or len(route.customers) < 4 or route.drone_missions:
            return route
        
        customers = route.customers.copy()
        improved = True
        
        while improved:
            improved = False
            best_delta = 0.0
            best_i = best_j = -1
            
            n = len(customers)
            
            for i in range(n - 2):
                for j in range(i + 2, n):
                    # Current edges: i->i+1, j->j+1 (with depot)
                    # After 2-opt: i->j, i+1->j+1
                    
                    # Convert to node indices (0=depot, 1..n=customers)
                    node_i = customers[i] + 1 if i > 0 else 0
                    node_i_plus_1 = customers[i + 1] + 1
                    node_j = customers[j] + 1
                    node_j_plus_1 = customers[j + 1] + 1 if j < n - 1 else 0
                    
                    # Distance delta: new - old
                    old_dist = self.dist_matrix[node_i, node_i_plus_1] + self.dist_matrix[node_j, node_j_plus_1]
                    new_dist = self.dist_matrix[node_i, node_j] + self.dist_matrix[node_i_plus_1, node_j_plus_1]
                    delta = new_dist - old_dist
                    
                    if delta < best_delta - 1e-6:
                        best_delta = delta
                        best_i, best_j = i, j
            
            if best_i >= 0 and best_delta < -1e-6:
                # Perform 2-opt swap
                customers[best_i + 1:best_j + 1] = customers[best_i + 1:best_j + 1][::-1]
                
                # Check time window feasibility
                if self._route_time_feasible(customers):
                    improved = True
                else:
                    # Revert if infeasible
                    customers[best_i + 1:best_j + 1] = customers[best_i + 1:best_j + 1][::-1]
        
        route.customers = customers
        return route
    
    def _inter_route_relocate(self, routes):
        """跨路线 relocate 算子：消除路径交叉，形成花瓣形路线。

        对每条路线中的每个客户，尝试将其移动到其他路线的更优位置。
        移动后删除被破坏的无人机任务（ALNS 后续会重新引入）。
        """
        if len(routes) < 2:
            return routes

        improved = True
        max_iter = len(routes) * 3

        for _ in range(max_iter):
            if not improved:
                break
            improved = False
            best_delta = 0.0
            best_move = None  # (from_r, from_pos, to_r, to_pos, cust)

            for r_idx in range(len(routes)):
                route = routes[r_idx]
                if not route.customers:
                    continue

                # 标记该路线中无人机服务的客户
                drone_targets = set()
                for m in route.drone_missions:
                    drone_targets.update(m.customer_ids)

                for pos in range(len(route.customers)):
                    cust = route.customers[pos]

                    # 跳过无人机服务的客户（不移动）
                    if cust in drone_targets:
                        continue

                    # 计算移除节省
                    prev_node = 0 if pos == 0 else route.customers[pos - 1] + 1
                    next_node = 0 if pos == len(route.customers) - 1 else route.customers[pos + 1] + 1
                    remove_saving = (self.dist_matrix[prev_node, cust + 1] +
                                     self.dist_matrix[cust + 1, next_node] -
                                     self.dist_matrix[prev_node, next_node])

                    for t_idx in range(len(routes)):
                        if t_idx == r_idx:
                            continue

                        target = routes[t_idx]
                        current_load = sum(self.demands[c] for c in target.customers)
                        if current_load + self.demands[cust] > self.truck_capacity:
                            continue

                        for ipos in range(len(target.customers) + 1):
                            t_prev = 0 if ipos == 0 else target.customers[ipos - 1] + 1
                            t_next = 0 if ipos == len(target.customers) else target.customers[ipos] + 1
                            insert_cost = (self.dist_matrix[t_prev, cust + 1] +
                                           self.dist_matrix[cust + 1, t_next] -
                                           self.dist_matrix[t_prev, t_next])

                            delta = insert_cost - remove_saving
                            if delta < best_delta - 1e-6:
                                best_delta = delta
                                best_move = (r_idx, pos, t_idx, ipos, cust)

            if best_move and best_delta < -1e-6:
                r_idx, pos, t_idx, ipos, cust = best_move
                from_route = routes[r_idx]
                to_route = routes[t_idx]

                # 移除前检查并删除引用该客户的无人机任务
                from_route.customers.pop(pos)
                valid_missions = []
                for m in from_route.drone_missions:
                    if m.launch_point == pos or m.return_point == pos:
                        continue  # 索引失效，删除
                    # 位移
                    if m.launch_point > pos:
                        m.launch_point -= 1
                    if m.return_point > pos:
                        m.return_point -= 1
                    valid_missions.append(m)
                from_route.drone_missions = valid_missions

                # 插入目标路线
                to_route.customers.insert(ipos, cust)
                for m in to_route.drone_missions:
                    if m.launch_point >= ipos:
                        m.launch_point += 1
                    if m.return_point >= ipos:
                        m.return_point += 1

                improved = True

        return routes

    def _route_time_feasible(self, customers):
        """Check if a route satisfies time window constraints."""
        current_time = 0.0
        current_node = 0  # depot
        
        for cust_id in customers:
            dist = self.dist_matrix[current_node, cust_id + 1]
            current_time += dist / self.truck_speed
            current_time = max(current_time, self.tw_start[cust_id])
            current_time += self.service_times[cust_id]
            
            if current_time > self.tw_end[cust_id] + 1e-6:
                return False
            
            current_node = cust_id + 1
        
        return True
    
    def _two_opt_solutions(self, routes):
        """Apply 2-opt to all routes in solution."""
        for route in routes:
            self._two_opt_route(route)
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
                
                # 2. 2-opt local search on each route (improves pure truck routes)
                route = self._two_opt_solutions(route)
                
                # 2.5 跨路线 relocate：消除路径交叉，形成花瓣形路线
                route = self._inter_route_relocate(route)
                
                # 3. 【核心注入】调用 ALNS 算子对单蚁生成的路径进行微观大邻域搜索微调
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
            if len(self.pareto_front) > self.archive_capacity:
                combined = list(zip(self.pareto_front, self.pareto_solutions))
                combined.sort(key=lambda x: x[0][0])  
                
                indices = np.linspace(0, len(combined) - 1, self.archive_capacity, dtype=int)
                self.pareto_front = [combined[idx][0] for idx in indices]
                self.pareto_solutions = [combined[idx][1] for idx in indices]
            
            if self.pareto_solutions:
                self._update_pheromones(zip(self.pareto_solutions, self.pareto_front))
                
            if (iteration + 1) % 10 == 0:
                print(f"[P-ACO+ALNS] Iteration {iteration + 1}/{self.max_iter} - Elite Archive Size: {len(self.pareto_front)}")
                
        return self.pareto_solutions, self.pareto_front