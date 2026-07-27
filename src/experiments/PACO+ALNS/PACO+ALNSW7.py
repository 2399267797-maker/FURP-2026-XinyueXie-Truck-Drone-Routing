import numpy as np
import random
import math
from typing import List, Tuple, Dict, Optional, Set
from collections import defaultdict
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'PACO_vs_NSGA2'))
from models.vrp_model import VRPTruckDroneModel, Route, DroneMission

# =============================================================================
# PACO+ALNS 终极膜拜版 (Memetic Algorithm) + imp2 稳健构造 (Fix Missing Bugs)
# [修复日志]:
#   1. [Fix] 彻底解决 ALNS 评分漏洞：将 missing 和 overload 惩罚同步到 SA 退火判断中。
#   2. [Fix] 构造解兜底修复：消除 greedy 阶段 break 导致的直接抛弃。
#   3. [Fix] 废弃无人机清理修复：失效任务中的客户将强制退回卡车，杜绝凭空消失。
# =============================================================================

class ScaleAdaptiveParams:
    @staticmethod
    def compute(n_customers: int) -> dict:
        scale = n_customers / 25.0
        return {
            'n_ants': max(20, min(40, int(20 + 10 * math.log(scale + 1)))),
            'alns_iter': max(10, min(25, int(15 + 5 * math.log(scale + 1)))),
            'destroy_ratio': 0.08,  
            'alns_max_remove': 5,   
            'elite_alns_ratio': 0.20, 
            'TAU_MAX': 20.0, 'TAU_MIN': 1.0, 'TAU_INIT': 10.0,
            'Q_c': max(120.0, int(120.0 * (scale ** 0.6))),
            'Q_t': max(60.0, int(60.0 * (scale ** 0.6))),
            'archive_capacity': max(10, min(100, int(n_customers * 0.25))),
            'DRONE_WAIT_THRESHOLD': 3.0,
            'ALNS_DECAY': 0.85,
            'SCORE_BEST': 5.0, 'SCORE_BETTER': 3.0, 'SCORE_ACCEPT': 1.0,
            'TARD_PENALTY_TRUCK': 10.0, 'TARD_PENALTY_DRONE': 5.0,
            'KNN_RATIO': 0.3, 'MAX_DRONE_GAP': 4,
            'WARM_START_MAX_RATIO': 0.15  
        }

class CollaborativePACOALNS:
    def __init__(self, model: VRPTruckDroneModel, n_ants: Optional[int] = None, max_iter: int = 100, alns_iter: Optional[int] = None):
        self.model = model
        self.n_customers = model.get_number_of_customers()
        self.n_trucks = model.get_number_of_trucks()
        self.n_drones = model.get_number_of_drones()

        p = ScaleAdaptiveParams.compute(self.n_customers)
        self.n_ants = n_ants if n_ants is not None else p['n_ants']
        self.max_iter = max_iter
        self.alns_iter = alns_iter if alns_iter is not None else p['alns_iter']
        self.destroy_ratio = p['destroy_ratio']
        self.alns_max_remove = p['alns_max_remove']
        self.elite_alns_ratio = p['elite_alns_ratio']

        self.alpha, self.beta, self.rho, self.q0 = 1.0, 2.0, 0.15, 0.5
        self.Q_c, self.Q_t, self.TAU_MAX, self.TAU_MIN, self.TAU_INIT = p['Q_c'], p['Q_t'], p['TAU_MAX'], p['TAU_MIN'], p['TAU_INIT']
        self.archive_capacity = p['archive_capacity']
        self.drone_wait_threshold = p['DRONE_WAIT_THRESHOLD']
        self.tard_penalty_truck, self.tard_penalty_drone = p['TARD_PENALTY_TRUCK'], p['TARD_PENALTY_DRONE']
        self.knn_ratio, self.max_drone_gap = p['KNN_RATIO'], p['MAX_DRONE_GAP']
        self.warm_start_max_ratio = p['WARM_START_MAX_RATIO']
        self.max_two_opt_iter = 5

        n_nodes = self.n_customers + 1
        self.phero_truck_c = np.full((n_nodes, n_nodes), self.TAU_INIT, dtype=np.float64)
        self.phero_truck_t = np.full((n_nodes, n_nodes), self.TAU_INIT, dtype=np.float64)
        self.phero_drone_c: Dict[Tuple[int, int, int], float] = {}
        self.phero_drone_t: Dict[Tuple[int, int, int], float] = {}
        self.pareto_front: List[Tuple[float, float]] = []
        self.pareto_solutions: List[List[Route]] = []

        self.dist_matrix = np.zeros((n_nodes, n_nodes), dtype=np.float64)
        nodes_list = [model.depot] + model.customers
        for i in range(n_nodes):
            for j in range(n_nodes):
                self.dist_matrix[i, j] = nodes_list[i].distance_to(nodes_list[j])

        self.demands = np.array([c.demand for c in model.customers], dtype=np.float64)
        self.tw_start = np.array([c.time_window[0] for c in model.customers], dtype=np.float64)
        self.tw_end = np.array([c.time_window[1] for c in model.customers], dtype=np.float64)
        self.service_times = np.array([c.service_time for c in model.customers], dtype=np.float64)

        self.truck_speed, self.truck_capacity = model.trucks[0].speed, model.trucks[0].capacity
        self.drone_speed, self.drone_capacity = model.get_vehicle_speed('drone'), model.drones[0].capacity
        self.drone_range = model.drone_range
        self.launch_prep_time, self.retrieval_time = getattr(model, 'launch_prep_time', 0.5), getattr(model, 'retrieval_time', 0.5)

        self.drone_knn = {}
        for j in range(self.n_customers):
            dists = self.dist_matrix[j + 1, 1:]
            self.drone_knn[j] = [idx for idx in np.argsort(dists) if idx != j][:self.max_drone_gap]

        self.destroy_ops = ['surgical', 'random', 'worst', 'related', 'route']
        self.repair_ops = ['drone_aware', 'greedy', 'regret']
        self.d_weights = {op: 1.0 for op in self.destroy_ops}
        self.r_weights = {op: 1.0 for op in self.repair_ops}
        self.score_best, self.score_better, self.score_accept = p['SCORE_BEST'], p['SCORE_BETTER'], p['SCORE_ACCEPT']
        self.alns_decay = p['ALNS_DECAY']

    # -------------------------- 工具与时间线 --------------------------
    def _alloc_drone_id(self, truck_id: int = 0, proposed_launch_time: float = 0.0, current_route: Optional[Route] = None) -> int:
        did = truck_id
        if current_route and current_route.drone_missions:
            last_m = max((m for m in current_route.drone_missions if m.drone_id == did), key=lambda x: x.return_point, default=None)
            if last_m and last_m.return_point < len(current_route.customers):
                dc = last_m.customer_ids[0]
                dr_end, _, feas = self._compute_drone_mission_timeline(self._compute_truck_time_at(current_route, last_m.launch_point), self._get_node_index(current_route, last_m.launch_point), dc, self._get_node_index(current_route, last_m.return_point))
                if dr_end and proposed_launch_time < dr_end + 1e-4: return -1
        return did

    def _compute_drone_mission_timeline(self, current_time, i_node, j_cust, k_node):
        d_ij, d_jk = self.dist_matrix[i_node, j_cust + 1], self.dist_matrix[j_cust + 1, k_node]
        if (d_ij + d_jk) > self.drone_range + 1e-6: return None, None, False
        lt = current_time + self.launch_prep_time
        arr_dr_k = max(lt + (d_ij / self.drone_speed), self.tw_start[j_cust]) + self.service_times[j_cust] + (d_jk / self.drone_speed)
        arr_tr_k = lt + (self.dist_matrix[i_node, k_node] / self.truck_speed)
        wait = max(0.0, arr_dr_k - arr_tr_k)
        if wait > self.drone_wait_threshold: return None, None, False
        return max(arr_tr_k, arr_dr_k) + self.retrieval_time, wait, True

    def _get_node_index(self, route, point_idx: int) -> int:
        return 0 if point_idx < 0 or point_idx >= len(route.customers) else route.customers[point_idx] + 1

    def _compute_truck_time_at(self, route, idx):
        if idx < 0 or not route.customers: return 0.0
        tl = self._compute_truck_timeline(route)
        return tl[idx] if idx < len(tl) else 0.0

    def _compute_truck_timeline(self, route):
        if hasattr(route, '_timeline_dirty') and not route._timeline_dirty: return route._timeline
        custs = route.customers
        if not custs:
            route._timeline, route._timeline_dirty = [], False
            return []
        
        times = [0.0] * len(custs)
        cur_t, cur_n = 0.0, 0
        for ci, cc in enumerate(custs):
            cur_t = max(cur_t + self.dist_matrix[cur_n, cc + 1] / self.truck_speed, self.tw_start[cc]) + self.service_times[cc]
            times[ci], cur_n = cur_t, cc + 1

        changed, iters = True, 0
        missions = sorted(route.drone_missions, key=lambda x: x.return_point)
        while changed and iters < 5:
            changed, iters = False, iters + 1
            for m in missions:
                i_idx, k_idx = m.launch_point, m.return_point
                if not (0 <= i_idx < len(custs) and 0 <= k_idx < len(custs)): continue
                arr_k, _, feas = self._compute_drone_mission_timeline(times[i_idx], custs[i_idx] + 1, m.customer_ids[0], custs[k_idx] + 1)
                if feas:
                    actual_k = max(arr_k, self.tw_start[custs[k_idx]]) + self.service_times[custs[k_idx]]
                    diff = actual_k - times[k_idx]
                    if diff > 1e-4:
                        for jj in range(k_idx, len(custs)): times[jj] += diff
                        changed = True

        route._timeline, route._timeline_dirty = times, False
        return times

    def _clone_routes(self, routes: List[Route]) -> List[Route]:
        cloned = []
        for r in routes:
            new_r = Route(r.vehicle_id, 'truck', r.customers.copy(), [DroneMission(m.drone_id, m.customer_ids.copy(), m.launch_point, m.return_point) for m in r.drone_missions])
            new_r._timeline_dirty = True
            cloned.append(new_r)
        return cloned

    # -------------------------- ALNS 破坏算子库 --------------------------
    def _destroy_surgical(self, routes: List[Route], k: int) -> Tuple[List[Route], Set[int]]:
        if not routes: return self._clone_routes(routes), set()
        valid_routes = [r for r in routes if r.customers]
        if not valid_routes: return self._clone_routes(routes), set()
        r = random.choice(valid_routes)
        
        start_idx = random.randint(0, len(r.customers) - 1)
        chunk_custs = set(r.customers[start_idx:min(start_idx + k, len(r.customers))])
        
        for m in r.drone_missions:
            if m.launch_point >= start_idx and m.launch_point < start_idx + k:
                chunk_custs.add(m.customer_ids[0])
            elif m.return_point >= start_idx and m.return_point < start_idx + k:
                chunk_custs.add(m.customer_ids[0])

        while len(chunk_custs) < k:
            all_s = [c for rr in routes for c in rr.customers] + [m.customer_ids[0] for rr in routes for m in rr.drone_missions]
            rem = list(set(all_s) - chunk_custs)
            if not rem: break
            chunk_custs.add(random.choice(rem))
            
        return self._apply_removal(routes, chunk_custs)

    def _destroy_random(self, routes: List[Route], k: int) -> Tuple[List[Route], Set[int]]:
        all_served = [c for r in routes for c in r.customers] + [m.customer_ids[0] for r in routes for m in r.drone_missions]
        if not all_served: return self._clone_routes(routes), set()
        to_remove = set(random.sample(all_served, min(k, len(all_served))))
        return self._apply_removal(routes, to_remove)

    def _destroy_worst(self, routes: List[Route], k: int) -> Tuple[List[Route], Set[int]]:
        costs = []
        for r in routes:
            for i, c in enumerate(r.customers):
                prev_n = 0 if i == 0 else r.customers[i-1]+1
                next_n = 0 if i == len(r.customers)-1 else r.customers[i+1]+1
                cost = self.dist_matrix[prev_n, c+1] + self.dist_matrix[c+1, next_n] - self.dist_matrix[prev_n, next_n]
                costs.append((cost, c))
            for m in r.drone_missions:
                i_n = self._get_node_index(r, m.launch_point)
                j_n = m.customer_ids[0] + 1
                k_n = self._get_node_index(r, m.return_point)
                cost = self.dist_matrix[i_n, j_n] + self.dist_matrix[j_n, k_n] - self.dist_matrix[i_n, k_n]
                costs.append((cost, m.customer_ids[0]))
        costs.sort(key=lambda x: -x[0])
        to_remove = set(c for _, c in costs[:min(k, len(costs))])
        return self._apply_removal(routes, to_remove)

    def _destroy_related(self, routes: List[Route], k: int) -> Tuple[List[Route], Set[int]]:
        all_served = [c for r in routes for c in r.customers] + [m.customer_ids[0] for r in routes for m in r.drone_missions]
        if not all_served: return self._clone_routes(routes), set()
        seed = random.choice(all_served)
        dists = [(self.dist_matrix[seed+1, c+1], c) for c in all_served]
        dists.sort(key=lambda x: x[0])
        to_remove = set(c for _, c in dists[:min(k, len(dists))])
        return self._apply_removal(routes, to_remove)

    def _destroy_route(self, routes: List[Route], k: int) -> Tuple[List[Route], Set[int]]:
        if not routes: return [], set()
        valid = [r for r in routes if r.customers]
        if not valid: return self._clone_routes(routes), set()
        r = random.choice(valid)
        to_remove = set(r.customers + [m.customer_ids[0] for m in r.drone_missions])
        return self._apply_removal(routes, to_remove)

    def _apply_removal(self, routes: List[Route], to_remove: Set[int]) -> Tuple[List[Route], Set[int]]:
        destroyed = []
        for r in routes:
            new_r = Route(r.vehicle_id, 'truck')
            new_r._timeline_dirty = True
            idx_map = {}
            for old_idx, c in enumerate(r.customers):
                if c not in to_remove:
                    idx_map[old_idx] = len(new_r.customers)
                    new_r.customers.append(c)
            for m in r.drone_missions:
                dt = m.customer_ids[0]
                nla, nret = idx_map.get(m.launch_point, -1), idx_map.get(m.return_point, -1)
                if dt not in to_remove and nla >= 0 and nret >= 0 and nla < nret:
                    new_r.drone_missions.append(DroneMission(m.drone_id, [dt], nla, nret))
                else: to_remove.add(dt)
            destroyed.append(new_r)
        return destroyed, to_remove

    # -------------------------- ALNS 统一底层与修复算子库 --------------------------
    def _get_all_insertions(self, routes: List[Route], c: int) -> List[Tuple[float, tuple]]:
        insertions = []
        for rid, r in enumerate(routes):
            load = sum(self.demands[cc] for cc in r.customers)
            for pos in range(len(r.customers) + 1):
                if load + self.demands[c] > self.truck_capacity + 1e-6: continue
                pn = 0 if pos == 0 else r.customers[pos-1]+1
                nn = 0 if pos == len(r.customers) else r.customers[pos]+1
                inc = self.dist_matrix[pn, c+1] + self.dist_matrix[c+1, nn] - self.dist_matrix[pn, nn]
                insertions.append((inc, ('truck', rid, pos)))
            
            if len(r.customers) >= 2 and self.demands[c] <= self.drone_capacity + 1e-6:
                for pos in range(len(r.customers) - 1):
                    i_n, k_n = r.customers[pos]+1, r.customers[pos+1]+1
                    if self.dist_matrix[i_n, c+1] + self.dist_matrix[c+1, k_n] > self.drone_range + 1e-6: continue
                    lt = self._compute_truck_time_at(r, pos) + self.launch_prep_time
                    arr_k, wait, feas = self._compute_drone_mission_timeline(lt - self.launch_prep_time, i_n, c, k_n)
                    if not feas: continue
                    save = self.dist_matrix[i_n, c+1] + self.dist_matrix[c+1, k_n] - self.dist_matrix[i_n, k_n]
                    if save > 0:
                        tard = max(0.0, (lt + self.dist_matrix[i_n, c+1]/self.drone_speed) - self.tw_end[c])
                        sc = -save + self.tard_penalty_drone * tard
                        insertions.append((sc, ('drone', rid, pos, pos+1)))
        
        insertions.sort(key=lambda x: x[0])
        return insertions

    def _apply_insertion(self, routes: List[Route], action: tuple, c: int):
        if not action:
            min(routes, key=lambda x: sum(self.demands[cc] for cc in x.customers)).customers.append(c)
            return
            
        if action[0] == 'truck':
            tr = routes[action[1]]
            tr.customers.insert(action[2], c); tr._timeline_dirty = True
            for m in tr.drone_missions:
                if m.launch_point >= action[2]: m.launch_point += 1
                if m.return_point >= action[2]: m.return_point += 1
        else:
            tr = routes[action[1]]
            lt = self._compute_truck_time_at(tr, action[2]) + self.launch_prep_time
            did = self._alloc_drone_id(action[1], lt, current_route=tr)
            if did < 0:
                tr.customers.insert(action[3], c)
            else:
                tr.drone_missions.append(DroneMission(did, [c], action[2], action[3]))
            tr._timeline_dirty = True

    def _repair_greedy(self, routes: List[Route], removed: List[int]) -> List[Route]:
        unassigned = list(removed)
        while unassigned:
            best_sc, best_act, best_c = float('inf'), None, None
            for c in unassigned:
                ins = self._get_all_insertions(routes, c)
                if ins and ins[0][0] < best_sc:
                    best_sc, best_act, best_c = ins[0][0], ins[0][1], c
            if best_c is None: best_c = unassigned[0]
            self._apply_insertion(routes, best_act, best_c)
            unassigned.remove(best_c)
        return routes

    def _repair_regret(self, routes: List[Route], removed: List[int]) -> List[Route]:
        unassigned = list(removed)
        while unassigned:
            regrets = []
            for c in unassigned:
                ins = self._get_all_insertions(routes, c)
                if not ins: regrets.append((0, c, None))
                elif len(ins) == 1: regrets.append((999, c, ins[0][1]))
                else: regrets.append((ins[1][0] - ins[0][0], c, ins[0][1]))
            regrets.sort(key=lambda x: -x[0])
            _, best_c, best_act = regrets[0]
            self._apply_insertion(routes, best_act, best_c)
            unassigned.remove(best_c)
        return routes

    def _repair_drone_aware(self, routes: List[Route], removed: List[int]) -> List[Route]:
        for c in removed:
            ins = self._get_all_insertions(routes, c)
            best_act = ins[0][1] if ins else None
            self._apply_insertion(routes, best_act, c)
        return routes

    def _select_operator(self, op_type: str) -> str:
        weights = self.d_weights if op_type == 'destroy' else self.r_weights
        ops, ws = list(weights.keys()), list(weights.values())
        return random.choices(ops, weights=ws)[0]

    def _update_weights(self, d_op: str, r_op: str, score: float):
        self.d_weights[d_op] = self.d_weights[d_op] * self.alns_decay + score * (1 - self.alns_decay)
        self.r_weights[r_op] = self.r_weights[r_op] * self.alns_decay + score * (1 - self.alns_decay)

    # -------------------------- 失效任务清理与跨路线优化 --------------------------
    def _cleanup_invalid_drone_missions(self, routes: List[Route]) -> List[Route]:
        all_lost = []
        for r in routes:
            custs = r.customers
            if not custs:
                r.drone_missions = []
                continue

            changed = True
            while changed:
                changed = False
                valid = []

                for m in r.drone_missions:
                    dc = m.customer_ids[0]
                    if m.launch_point >= len(custs) or m.return_point >= len(custs) or m.launch_point >= m.return_point:
                        all_lost.append((dc, r)); changed = True; continue
                    i_n = self._get_node_index(r, m.launch_point)
                    j_n = dc + 1
                    k_n = self._get_node_index(r, m.return_point)
                    if self.dist_matrix[i_n, j_n] + self.dist_matrix[j_n, k_n] > self.drone_range + 1e-6:
                        all_lost.append((dc, r)); changed = True; continue
                    
                    cur_t = self._compute_truck_time_at(r, m.launch_point)
                    arr_k, wait, feas = self._compute_drone_mission_timeline(cur_t, i_n, dc, k_n)
                    if not feas:
                        all_lost.append((dc, r)); changed = True; continue
                    valid.append(m)

                if not valid:
                    r.drone_missions = []; r._timeline_dirty = True; break

                truck_times = self._compute_truck_timeline(r)
                drone_mission_map = defaultdict(list)
                for m in valid: drone_mission_map[m.drone_id].append(m)

                new_valid = []
                for did, missions in drone_mission_map.items():
                    if len(missions) < 2:
                        new_valid.extend(missions); continue
                    intervals = []
                    for m in missions:
                        dc = m.customer_ids[0]
                        i_n = self._get_node_index(r, m.launch_point)
                        j_n = dc + 1
                        k_n = self._get_node_index(r, m.return_point)
                        lt = truck_times[m.launch_point] + self.launch_prep_time
                        arr_drone_k = max(lt + self.dist_matrix[i_n, j_n] / self.drone_speed, self.tw_start[dc])
                        arr_drone_k += self.service_times[dc] + self.dist_matrix[j_n, k_n] / self.drone_speed
                        truck_arr_k = lt + self.dist_matrix[i_n, k_n] / self.truck_speed
                        dr_end = max(truck_arr_k, arr_drone_k) + self.retrieval_time
                        intervals.append((lt, dr_end, m))
                    
                    intervals.sort(key=lambda x: x[0])
                    kept = []
                    for i in range(len(intervals)):
                        conflict = False
                        for j in range(i):
                            if intervals[j][1] > intervals[i][0] + 1e-4:
                                conflict = True; break
                        if not conflict: kept.append(intervals[i][2])
                        else: all_lost.append((intervals[i][2].customer_ids[0], r)); changed = True
                    new_valid.extend(kept)

                if len(new_valid) < len(valid):
                    r.drone_missions = new_valid; r._timeline_dirty = True; changed = True
                else:
                    r.drone_missions = valid; r._timeline_dirty = True; break

        for dc, _ in all_lost:
            if any(dc in rr.customers or any(dc in m.customer_ids for m in rr.drone_missions) for rr in routes): continue
            placed = False
            for rr in routes:
                if sum(self.demands[c] for c in rr.customers) + self.demands[dc] <= self.truck_capacity + 1e-6:
                    rr.customers.append(dc); placed = True; break
            if not placed:
                min(routes, key=lambda rr: sum(self.demands[c] for c in rr.customers)).customers.append(dc)
        return routes

    def _inter_route_relocate(self, routes: List[Route]) -> List[Route]:
        if len(routes) < 2: return routes
        improved = True
        max_iter = len(routes) * 3

        for _ in range(max_iter):
            if not improved: break
            improved, best_delta, best_move = False, 0.0, None

            for r_idx in range(len(routes)):
                route = routes[r_idx]
                if not route.customers: continue
                drone_targets = set()
                for m in route.drone_missions: drone_targets.update(m.customer_ids)

                for pos in range(len(route.customers)):
                    cust = route.customers[pos]
                    if cust in drone_targets: continue
                    prev_node = 0 if pos == 0 else route.customers[pos - 1] + 1
                    next_node = 0 if pos == len(route.customers) - 1 else route.customers[pos + 1] + 1
                    remove_saving = self.dist_matrix[prev_node, cust + 1] + self.dist_matrix[cust + 1, next_node] - self.dist_matrix[prev_node, next_node]

                    for t_idx in range(len(routes)):
                        if t_idx == r_idx: continue
                        target = routes[t_idx]
                        if sum(self.demands[c] for c in target.customers) + self.demands[cust] > self.truck_capacity + 1e-6: continue

                        for ipos in range(len(target.customers) + 1):
                            t_prev = 0 if ipos == 0 else target.customers[ipos - 1] + 1
                            t_next = 0 if ipos == len(target.customers) else target.customers[ipos] + 1
                            delta = (self.dist_matrix[t_prev, cust + 1] + self.dist_matrix[cust + 1, t_next] - self.dist_matrix[t_prev, t_next]) - remove_saving
                            if delta < best_delta - 1e-6:
                                best_delta, best_move = delta, (r_idx, pos, t_idx, ipos, cust)

            if best_move and best_delta < -1e-6:
                r_idx, pos, t_idx, ipos, cust = best_move
                from_route, to_route = routes[r_idx], routes[t_idx]

                from_route.customers.pop(pos)
                lost_drone_custs, valid_missions = [], []
                for m in from_route.drone_missions:
                    if m.launch_point == pos or m.return_point == pos: lost_drone_custs.extend(m.customer_ids); continue
                    if m.launch_point > pos: m.launch_point -= 1
                    if m.return_point > pos: m.return_point -= 1
                    valid_missions.append(m)
                from_route.drone_missions = valid_missions

                for dc in lost_drone_custs:
                    if sum(self.demands[c] for c in from_route.customers) + self.demands[dc] <= self.truck_capacity + 1e-6:
                        from_route.customers.append(dc)
                    else:
                        placed = False
                        for rr in routes:
                            if rr == from_route: continue
                            if sum(self.demands[c] for c in rr.customers) + self.demands[dc] <= self.truck_capacity + 1e-6 and dc not in rr.customers:
                                rr.customers.append(dc); placed = True; break
                        if not placed: from_route.customers.append(dc)

                to_route.customers.insert(ipos, cust)
                for m in to_route.drone_missions:
                    if m.launch_point >= ipos: m.launch_point += 1
                    if m.return_point >= ipos: m.return_point += 1
                
                from_route._timeline_dirty = to_route._timeline_dirty = True
                improved = True

        return routes

    def _alns_local_search(self, routes: List[Route], global_best_score: float) -> List[Route]:
        best_r = self._clone_routes(routes)
        best_cost, best_tard = self._eval_solution(best_r)[0], self._calc_tardiness(best_r)
        
        # [Fix 1]: 确保基准 score 包含所有可能的遗漏和超载惩罚
        served_init = set([c for r in best_r for c in r.customers] + [m.customer_ids[0] for r in best_r for m in r.drone_missions])
        missing_init = self.n_customers - len(served_init)
        overload_init = sum(max(0.0, sum(self.demands[c] for c in r.customers) - self.truck_capacity) for r in best_r)
        
        if missing_init > 0:
            best_cost += missing_init * self.tard_penalty_truck * 1000.0
            best_tard += missing_init * self.tard_penalty_drone * 1000.0
        if overload_init > 1e-6:
            best_cost += overload_init * 1000.0
            
        best_score = best_cost + best_tard * self.tard_penalty_truck
        
        cur_r = self._clone_routes(routes)
        cur_score = best_score
        
        total_cust = sum(len(r.customers) + len(r.drone_missions) for r in routes)
        if total_cust < 4: return routes
        
        k = max(1, min(int(total_cust * self.destroy_ratio), self.alns_max_remove))
        start_temp = 10.0
        cooling_rate = 0.93

        for it in range(self.alns_iter):
            d_op = self._select_operator('destroy')
            r_op = self._select_operator('repair')

            if d_op == 'surgical': work_r, removed = self._destroy_surgical(cur_r, k)
            elif d_op == 'worst': work_r, removed = self._destroy_worst(cur_r, k)
            elif d_op == 'related': work_r, removed = self._destroy_related(cur_r, k)
            elif d_op == 'route': work_r, removed = self._destroy_route(cur_r, k)
            else: work_r, removed = self._destroy_random(cur_r, k)

            if r_op == 'greedy': work_r = self._repair_greedy(work_r, list(removed))
            elif r_op == 'regret': work_r = self._repair_regret(work_r, list(removed))
            else: work_r = self._repair_drone_aware(work_r, list(removed))

            work_r = self._post_repair_drone_optimization(work_r)
            work_r = self._cleanup_invalid_drone_missions(work_r)
            work_r = self._repair_capacity(work_r)

            rep_c, _ = self._eval_solution(work_r)
            rep_t = self._calc_tardiness(work_r)
            
            # [Fix 1] 计算 SA 退火用的 rep_score 时，彻底封死惩罚漏洞！
            served_alns = set([c for r in work_r for c in r.customers] + [m.customer_ids[0] for r in work_r for m in r.drone_missions])
            missing_alns = self.n_customers - len(served_alns)
            overload_alns = sum(max(0.0, sum(self.demands[c] for c in r.customers) - self.truck_capacity) for r in work_r)
            
            if missing_alns > 0:
                rep_c += missing_alns * self.tard_penalty_truck * 1000.0
                rep_t += missing_alns * self.tard_penalty_drone * 1000.0
            if overload_alns > 1e-6:
                rep_c += overload_alns * 1000.0
                
            rep_score = rep_c + rep_t * self.tard_penalty_truck

            delta = rep_score - cur_score
            current_temp = max(0.1, start_temp * (cooling_rate ** it))
            
            accept = delta <= 0 or random.random() < math.exp(-delta / current_temp)

            score_val = 0.0
            if accept:
                if rep_score < global_best_score:
                    score_val = self.score_best
                    global_best_score = rep_score
                elif rep_score < cur_score:
                    score_val = self.score_better
                else:
                    score_val = self.score_accept
                self._update_weights(d_op, r_op, score_val)

                cur_r, cur_score = work_r, rep_score
                
                if rep_score < best_score:
                    best_r = self._clone_routes(work_r)
                    best_score = rep_score

        return best_r

    # -------------------------- [核心融合] imp2 构造解与极速 2-opt --------------------------
    def _construct_solution(self) -> List[Route]:
        routes, remaining = [], set(range(self.n_customers))
        for truck_id in range(self.n_trucks):
            if not remaining: break
            route = Route(vehicle_id=truck_id, vehicle_type='truck')
            ct, cl, cn = 0.0, 0.0, 0
            while remaining:
                cand = list(remaining)
                probs, acts = [], []
                for j in cand:
                    if cl + self.demands[j] > self.truck_capacity + 1e-6: continue
                    dij = self.dist_matrix[cn, j + 1]
                    arj = ct + (dij / self.truck_speed)
                    p = (self.phero_truck_c[cn, j + 1] ** self.alpha) * (self.phero_truck_t[cn, j + 1] ** self.alpha) * \
                        ((1.0 / (dij + 0.001)) ** self.beta) * ((1.0 / (1.0 + max(0.0, arj - self.tw_end[j]))) ** self.beta)
                    probs.append(p); acts.append(('truck', j))
                
                if route.customers:
                    for j in cand:
                        if self.demands[j] > self.drone_capacity + 1e-6 or j in route.customers: continue
                        dij = self.dist_matrix[cn, j + 1]
                        if dij >= self.drone_range: continue
                        for k in cand:
                            if j == k or k in route.customers: continue
                            if cl + self.demands[k] > self.truck_capacity + 1e-6: continue
                            djk = self.dist_matrix[j + 1, k + 1]
                            if (dij + djk) > self.drone_range + 1e-6: continue
                            sk, wt, feas = self._compute_drone_mission_timeline(ct, cn, j, k + 1)
                            if not feas: continue
                            dik = self.dist_matrix[cn, k + 1]
                            save = dij + djk - dik
                            if save <= 0: continue
                            
                            pj = max(0.0, (ct + dij / self.drone_speed) - self.tw_end[j])
                            pk = max(0.0, sk - self.tw_end[k])
                            key = (cn, j + 1, k + 1)
                            tc = self.phero_drone_c.get(key, self.TAU_INIT)
                            tt = self.phero_drone_t.get(key, self.TAU_INIT)
                            edc = (1.0 + max(0.0, save)) / (dik + 0.001)
                            edt = 1.0 / (1.0 + pj + pk + wt * 2.0)
                            p = (tc ** self.alpha) * (tt ** self.alpha) * (edc ** self.beta) * (edt ** self.beta)
                            probs.append(p); acts.append(('drone', j, k, sk))
                            
                tp = sum(probs)
                if tp == 0: break
                
                if random.random() < self.q0: ci = np.argmax([p / tp for p in probs])
                else: ci = random.choices(range(len(acts)), weights=[p / tp for p in probs])[0]
                
                a = acts[ci]
                if a[0] == 'truck':
                    _, j = a
                    route.customers.append(j); remaining.remove(j)
                    cl += self.demands[j]
                    ct = max(ct + self.dist_matrix[cn, j + 1] / self.truck_speed, self.tw_start[j]) + self.service_times[j]
                    cn = j + 1
                else:
                    _, dj, tk, sk = a
                    li = len(route.customers) - 1
                    ri = li + 1
                    route.customers.append(tk)
                    route.drone_missions.append(DroneMission(self.n_trucks + (truck_id % max(1, self.n_drones)), [dj], li, ri))
                    remaining.remove(dj); remaining.remove(tk)
                    cl += self.demands[tk]
                    ct = max(sk, self.tw_start[tk]) + self.service_times[tk]
                    cn = tk + 1
                    
            if route.customers: 
                route._timeline_dirty = True
                routes.append(route)
                
        while remaining and routes:
            bd, br, bc, bp = float('inf'), None, None, None
            for r in routes:
                ld = sum(self.demands[c] for c in r.customers)
                for pos in range(len(r.customers) + 1):
                    for cid in remaining:
                        if ld + self.demands[cid] > self.truck_capacity + 1e-6: continue
                        pn = 0 if pos == 0 else r.customers[pos - 1] + 1
                        nn = 0 if pos == len(r.customers) else r.customers[pos] + 1
                        d = self.dist_matrix[pn, cid + 1] + self.dist_matrix[cid + 1, nn] - self.dist_matrix[pn, nn]
                        if d < bd: bd, br, bc, bp = d, r, cid, pos
            if br:
                br.customers.insert(bp, bc)
                for m in br.drone_missions:
                    if m.launch_point >= bp: m.launch_point += 1
                    if m.return_point >= bp: m.return_point += 1
                br._timeline_dirty = True
                remaining.remove(bc)
            else: 
                # [Fix 2] 兜底强制插入，绝不允许 break 导致客户直接凭空消失！
                bc = list(remaining)[0]
                min_r = min(routes, key=lambda r: sum(self.demands[c] for c in r.customers))
                min_r.customers.append(bc)
                min_r._timeline_dirty = True
                remaining.remove(bc)
                
        for r in routes:
            tvs = set(r.customers)
            valid_m = []
            for m in r.drone_missions:
                dc = m.customer_ids[0]
                # [Fix 3] 收紧判断：如果不满足限制，不仅要剔除 mission，必须将失去的客户强塞回卡车！
                if dc not in tvs and 0 <= m.launch_point < len(r.customers) and 0 <= m.return_point < len(r.customers) and m.launch_point < m.return_point:
                    valid_m.append(m)
                else:
                    if dc not in tvs:
                        r.customers.append(dc)
                        tvs.add(dc)
                        r._timeline_dirty = True
            r.drone_missions = valid_m
            r._timeline_dirty = True
            
        return routes

    def _calc_route_tardiness(self, customers: List[int]) -> float:
        cur_t, tardiness, cur_n = 0.0, 0.0, 0
        for cc in customers:
            cur_t += self.dist_matrix[cur_n, cc + 1] / self.truck_speed
            tardiness += max(0.0, cur_t - self.tw_end[cc])
            cur_t = max(cur_t, self.tw_start[cc]) + self.service_times[cc]
            cur_n = cc + 1
        return tardiness

    def _two_opt_route(self, route):
        if not route.customers or len(route.customers) < 4: return route
        customers = route.customers.copy()
        improved = True
        _opt_iter = 0

        while improved and _opt_iter < self.max_two_opt_iter:
            _opt_iter += 1
            improved = False
            best_score_delta = 0.0
            best_i = best_j = -1
            n = len(customers)
            
            old_tardiness = self._calc_route_tardiness(customers)

            for i in range(n - 2):
                for j in range(i + 2, n):
                    n_i, n_ip, n_j, n_jp = customers[i] + 1 if i > 0 else 0, customers[i + 1] + 1, customers[j] + 1, customers[j + 1] + 1 if j < n - 1 else 0
                    delta_dist = self.dist_matrix[n_i, n_j] + self.dist_matrix[n_ip, n_jp] - (self.dist_matrix[n_i, n_ip] + self.dist_matrix[n_j, n_jp])
                    
                    if delta_dist < -1e-6:
                        temp_cust = customers[:i+1] + customers[i+1:j+1][::-1] + customers[j+1:]
                        score_delta = delta_dist + self.tard_penalty_truck * (self._calc_route_tardiness(temp_cust) - old_tardiness)
                        if score_delta < best_score_delta - 1e-6:
                            best_score_delta, best_i, best_j = score_delta, i, j

            if best_i >= 0 and best_score_delta < -1e-6:
                customers[best_i + 1:best_j + 1] = customers[best_i + 1:best_j + 1][::-1]
                route.customers = customers.copy()
                route._timeline_dirty = True
                valid_missions, lost = [], []
                truck_times_at = self._compute_truck_timeline(route)

                for m in route.drone_missions:
                    la, ret, dc = m.launch_point, m.return_point, m.customer_ids[0]
                    la_in = best_i < la <= best_j
                    ret_in = best_i < ret <= best_j
                    
                    if la_in != ret_in:
                        lost.append(dc); continue 
                    
                    if la_in and ret_in:
                        la, ret = best_i + 1 + (best_j - ret), best_i + 1 + (best_j - la)

                    if la >= len(customers) or ret >= len(customers) or la >= ret: lost.append(dc); continue
                    arr_k, _, feas = self._compute_drone_mission_timeline(truck_times_at[la], customers[la]+1, dc, customers[ret]+1)
                    if not feas: lost.append(dc); continue
                    
                    m.launch_point, m.return_point = la, ret
                    valid_missions.append(m)

                route.drone_missions = valid_missions
                for dc in lost:
                    route.customers.append(dc); customers.append(dc)
                route._timeline_dirty = True
                improved = True

        route.customers = customers
        return route

    # -------------------------- 事务隔离防污染 --------------------------
    def _post_repair_drone_optimization(self, routes: List[Route]) -> List[Route]:
        for tid, r in enumerate(routes):
            custs = r.customers
            if len(custs) < 2: continue
            idx = 0
            while idx < len(custs) - 1:
                j = custs[idx]
                if self.demands[j] > self.drone_capacity + 1e-6:
                    idx += 1; continue
                i_n = 0 if idx == 0 else custs[idx-1] + 1
                j_n = j + 1
                best_na, best_save = None, -float('inf')
                
                for gap in range(1, self.max_drone_gap + 1):
                    kidx = idx + gap
                    if kidx >= len(custs): break
                    k_n = custs[kidx] + 1
                    dij, djk = self.dist_matrix[i_n, j_n], self.dist_matrix[j_n, k_n]
                    if dij + djk > self.drone_range + 1e-6: continue
                    save = dij + djk - self.dist_matrix[i_n, k_n]
                    if save <= 0: continue
                    
                    cur_t = self._compute_truck_time_at(r, idx - 1)
                    launch_t = cur_t + self.launch_prep_time
                    drone_arr_j = max(launch_t + dij / self.drone_speed, self.tw_start[j])
                    wait = max(0.0, (drone_arr_j + self.service_times[j] + djk / self.drone_speed) - (launch_t + self.dist_matrix[i_n, k_n] / self.truck_speed))
                    if wait > self.drone_wait_threshold: continue
                    
                    if save > best_save:
                        best_save, best_na = save, kidx

                if best_na is None: idx += 1; continue
                kidx = best_na

                lt = self._compute_truck_time_at(r, idx - 1) + self.launch_prep_time
                did = self._alloc_drone_id(tid, lt, current_route=r)
                if did < 0:
                    idx += 1; continue 

                lost_dc, adjusted_missions = set(), []
                for m in r.drone_missions:
                    la, ret, dc = m.launch_point, m.return_point, m.customer_ids[0]
                    if la == idx: la = idx - 1 if idx > 0 else -1
                    if ret == idx: ret = idx - 1 if idx > 0 else -1
                    if la < idx < ret: ret -= 1
                    if la > idx: la -= 1
                    if ret > idx: ret -= 1
                    if la >= 0 and ret >= 0 and la < ret: m.launch_point, m.return_point = la, ret; adjusted_missions.append(m)
                    else: lost_dc.add(dc)
                
                r.drone_missions = adjusted_missions
                for lc in lost_dc:
                    placed = False
                    for rr in routes:
                        if sum(self.demands[cc] for cc in rr.customers) + self.demands[lc] <= self.truck_capacity + 1e-6 and lc not in rr.customers:
                            rr.customers.append(lc); rr._timeline_dirty = True; placed = True; break
                    if not placed:
                        min(routes, key=lambda rr: sum(self.demands[cc] for cc in rr.customers)).customers.append(lc)
                
                r.customers.pop(idx)
                adj_kidx = kidx - 1
                new_m = DroneMission(did, [j], idx-1, adj_kidx)
                r.drone_missions.append(new_m)
                r._timeline_dirty = True
                idx = adj_kidx + 1
        return routes

    def _repair_capacity(self, routes: List[Route]) -> List[Route]:
        if len(routes) < 2: return routes
        for _ in range(len(routes) * 3):
            overloaded = [r for r in routes if sum(self.demands[c] for c in r.customers) > self.truck_capacity + 1e-6]
            if not overloaded: break
            src = overloaded[0]
            target = min(routes, key=lambda r: sum(self.demands[c] for c in r.customers))
            if src is target and len(overloaded) > 1: src = overloaded[1]

            if src is not target:
                dst_load = sum(self.demands[c] for c in target.customers)
                if dst_load <= self.truck_capacity + 1e-6:
                    moved = False
                    for pos, cust in enumerate(src.customers):
                        if dst_load + self.demands[cust] > self.truck_capacity + 1e-6 or any(cust in m.customer_ids for m in src.drone_missions) or any(m.launch_point == pos or m.return_point == pos for m in src.drone_missions): continue
                        src.customers.pop(pos); target.customers.append(cust)
                        for m in src.drone_missions:
                            if m.launch_point > pos: m.launch_point -= 1
                            if m.return_point > pos: m.return_point -= 1
                        moved = True; src._timeline_dirty = target._timeline_dirty = True; break
                    if moved: continue

            for r in overloaded:
                pos = 0
                while pos < len(r.customers) - 1:
                    if sum(self.demands[c] for c in r.customers) <= self.truck_capacity + 1e-6: break
                    j = r.customers[pos]
                    if self.demands[j] > self.drone_capacity + 1e-6: 
                        pos += 1
                        continue
                    
                    i_n = 0 if pos == 0 else r.customers[pos-1] + 1
                    converted = False
                    for gap in range(1, min(self.max_drone_gap + 1, len(r.customers) - pos)):
                        kidx = pos + gap
                        k_n = r.customers[kidx] + 1
                        if self.dist_matrix[i_n, j+1] + self.dist_matrix[j+1, k_n] > self.drone_range + 1e-6: continue
                        launch_t = self._compute_truck_time_at(r, pos - 1) + self.launch_prep_time
                        _, _, feas = self._compute_drone_mission_timeline(launch_t, i_n, j, k_n)
                        if not feas: continue
                        did = self._alloc_drone_id(r.vehicle_id, launch_t, current_route=r)
                        if did < 0: continue
                        
                        for m2 in r.drone_missions:
                            if m2.launch_point > pos: m2.launch_point -= 1
                            if m2.return_point > pos: m2.return_point -= 1
                        r.customers.pop(pos)
                        r.drone_missions.append(DroneMission(did, [j], -1 if pos == 0 else pos - 1, kidx - 1))
                        r._timeline_dirty = True
                        converted = True
                        break
                    
                    if not converted:
                        pos += 1

            overloaded = [r for r in routes if sum(self.demands[c] for c in r.customers) > self.truck_capacity + 1e-6]
            if not overloaded: break

            src = overloaded[0]
            target = min(routes, key=lambda r: sum(self.demands[c] for c in r.customers))
            if src is not target:
                swapped = False
                for tpos, tcust in enumerate(target.customers):
                    if any(tcust in m.customer_ids for m in target.drone_missions) or any(m.launch_point == tpos or m.return_point == tpos for m in target.drone_missions): continue
                    target.customers.pop(tpos)
                    for m in target.drone_missions:
                        if m.launch_point > tpos: m.launch_point -= 1
                        if m.return_point > tpos: m.return_point -= 1
                    for spos, scust in enumerate(src.customers):
                        if scust == tcust or any(scust in m.customer_ids for m in src.drone_missions) or any(m.launch_point == spos or m.return_point == spos for m in src.drone_missions): continue
                        if sum(self.demands[c] for c in target.customers) + self.demands[scust] <= self.truck_capacity + 1e-6:
                            src.customers.pop(spos); target.customers.append(scust)
                            for m in src.drone_missions:
                                if m.launch_point > spos: m.launch_point -= 1
                                if m.return_point > spos: m.return_point -= 1
                            src.customers.append(tcust)
                            target._timeline_dirty = src._timeline_dirty = True
                            swapped = True; break
                    if swapped: break
                    
                    target.customers.insert(tpos, tcust)
                    for m in target.drone_missions:
                        if m.launch_point > tpos: m.launch_point += 1
                        if m.return_point > tpos: m.return_point += 1
        return routes

    # -------------------------- 信息素与主流程 --------------------------
    def _update_pheromones(self, solutions):
        self.phero_truck_c *= (1 - self.rho)
        self.phero_truck_t *= (1 - self.rho)

        for key in list(self.phero_drone_c.keys()):
            self.phero_drone_c[key] *= (1 - self.rho)
            if self.phero_drone_c[key] < self.TAU_MIN: del self.phero_drone_c[key]
        for key in list(self.phero_drone_t.keys()):
            self.phero_drone_t[key] *= (1 - self.rho)
            if self.phero_drone_t[key] < self.TAU_MIN: del self.phero_drone_t[key]

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
                    key = (i_node, j_node, k_node)
                    self.phero_drone_c[key] = self.phero_drone_c.get(key, self.TAU_INIT) + delta_c
                    self.phero_drone_t[key] = self.phero_drone_t.get(key, self.TAU_INIT) + delta_t

        self.phero_truck_c = np.clip(self.phero_truck_c, self.TAU_MIN, self.TAU_MAX)
        self.phero_truck_t = np.clip(self.phero_truck_t, self.TAU_MIN, self.TAU_MAX)

    def _eval_solution(self, routes): return self.model.evaluate_solution(routes)
    def _calc_tardiness(self, routes): return self.model.calculate_pure_tardiness(routes)
    def _dominates(self, obj1, obj2): return (obj1[0] <= obj2[0] and obj1[1] <= obj2[1]) and (obj1[0] < obj2[0] or obj1[1] < obj2[1])

    def solve(self) -> Tuple[List[List[Route]], List[Tuple[float, float]]]:
        global_best_score = float('inf')
        for iteration in range(self.max_iter):
            ant_solutions_pool = []
            progress = iteration / max(1, self.max_iter - 1)
            current_warm_ratio = self.warm_start_max_ratio * progress
            n_warm_starts = min(len(self.pareto_solutions), int(self.n_ants * current_warm_ratio)) if self.pareto_solutions else 0
            
            for ant_idx in range(self.n_ants):
                if ant_idx < n_warm_starts:
                    routes = self._clone_routes(random.choice(self.pareto_solutions))
                else:
                    routes = self._construct_solution()
                
                for r in routes: self._two_opt_route(r)
                routes = self._inter_route_relocate(routes)
                
                served = set([c for r in routes for c in r.customers] + [m.customer_ids[0] for r in routes for m in r.drone_missions])
                missing = self.n_customers - len(served)
                cost, _ = self._eval_solution(routes)
                tardiness = self._calc_tardiness(routes)
                
                if missing > 0:
                    cost += missing * self.tard_penalty_truck * 1000.0
                    tardiness += missing * self.tard_penalty_drone * 1000.0
                overload_penalty = sum(max(0.0, sum(self.demands[c] for c in r.customers) - self.truck_capacity) for r in routes)
                if overload_penalty > 1e-6:
                    cost += overload_penalty * 1000.0
                    
                score = cost + tardiness * self.tard_penalty_truck
                ant_solutions_pool.append({
                    'routes': routes, 'cost': cost, 'tardiness': tardiness, 'score': score
                })

            ant_solutions_pool.sort(key=lambda x: x['score'])
            n_elite_ants = max(1, int(self.n_ants * self.elite_alns_ratio))
            
            iteration_solutions = []
            for rank_idx, sol_data in enumerate(ant_solutions_pool):
                routes = sol_data['routes']
                
                if rank_idx < n_elite_ants:
                    routes = self._alns_local_search(routes, global_best_score)
                    
                    served = set([c for r in routes for c in r.customers] + [m.customer_ids[0] for r in routes for m in r.drone_missions])
                    missing = self.n_customers - len(served)
                    cost, _ = self._eval_solution(routes)
                    tardiness = self._calc_tardiness(routes)
                    if missing > 0:
                        cost += missing * self.tard_penalty_truck * 1000.0
                        tardiness += missing * self.tard_penalty_drone * 1000.0
                    overload_penalty = sum(max(0.0, sum(self.demands[c] for c in r.customers) - self.truck_capacity) for r in routes)
                    if overload_penalty > 1e-6:
                        cost += overload_penalty * 1000.0
                    score = cost + tardiness * self.tard_penalty_truck
                else:
                    cost, tardiness, score = sol_data['cost'], sol_data['tardiness'], sol_data['score']

                if score < global_best_score: global_best_score = score
                iteration_solutions.append((routes, (cost, tardiness)))

            all_candidates = list(zip(self.pareto_solutions, self.pareto_front))
            for sol, obj in iteration_solutions:
                if not any(self._dominates(ex_obj, obj) for ex_obj in self.pareto_front): all_candidates.append((sol, obj))

            new_front, new_solutions = [], []
            for i, (sol_i, obj_i) in enumerate(all_candidates):
                if not any(self._dominates(obj_j, obj_i) for j, (sol_j, obj_j) in enumerate(all_candidates) if i != j):
                    if not any(abs(obj_i[0] - eo[0]) < 1e-4 and abs(obj_i[1] - eo[1]) < 1e-4 for eo in new_front):
                        new_front.append(obj_i); new_solutions.append(sol_i)
            self.pareto_front, self.pareto_solutions = new_front, new_solutions

            if len(self.pareto_front) > self.archive_capacity:
                combined = list(zip(self.pareto_front, self.pareto_solutions))
                n, crowd, front_objs = len(combined), [0.0] * len(combined), [x[0] for x in combined]
                for d in [0, 1]:
                    idx_sort = sorted(range(n), key=lambda i: front_objs[i][d])
                    crowd[idx_sort[0]] = crowd[idx_sort[-1]] = float('inf')
                    span = front_objs[idx_sort[-1]][d] - front_objs[idx_sort[0]][d]
                    if span > 1e-6:
                        for i in range(1, n - 1): crowd[idx_sort[i]] += (front_objs[idx_sort[i+1]][d] - front_objs[idx_sort[i-1]][d]) / span
                sorted_idx = sorted(range(n), key=lambda i: -crowd[i])[:self.archive_capacity]
                self.pareto_front = [combined[i][0] for i in sorted(sorted_idx)]
                self.pareto_solutions = [combined[i][1] for i in sorted(sorted_idx)]

            if self.pareto_solutions: 
                self._update_pheromones(zip(self.pareto_solutions, self.pareto_front))

            if (iteration + 1) % 10 == 0:
                print(f"[PACO+ALNS Fixed+imp2] Iteration {iteration + 1}/{self.max_iter} - Pareto Archive: {len(self.pareto_front)}")
                print(f"  ALNS weights: Destroy={['%.2f'%w for w in self.d_weights.values()]} Repair={['%.2f'%w for w in self.r_weights.values()]}")

        return self.pareto_solutions, self.pareto_front