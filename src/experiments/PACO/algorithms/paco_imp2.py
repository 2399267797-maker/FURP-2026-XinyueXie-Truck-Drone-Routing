import numpy as np
import random
import math
from typing import List, Tuple, Dict, Optional, Set
from collections import defaultdict
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.vrp_model import VRPTruckDroneModel, Route, DroneMission


class CollaborativePACO:
    """PACO-imp2: 修复版纯 PACO（无 ALNS）。
    
    修复清单:
      1. 帕累托更新顺序依赖 — 两阶段一次性支配排序
      2. 无人机负载不累加至卡车 — 只检查卡车客户 demand_k
      3. 无人机时间线对齐时间窗等待
      4. 信息素加权乘积融合
      5. 兜底插入检查容量
      6. 稀疏字典替代三维矩阵
      7. 拥挤度距离裁剪替代等距采样
    """
    def __init__(self, model: VRPTruckDroneModel, n_ants: int = 30, max_iter: int = 100):
        self.model = model
        self.n_ants = n_ants
        self.max_iter = max_iter
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
        self.archive_capacity = max(10, min(50, int(self.n_customers * 0.5)))
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
        self.truck_speed = model.trucks[0].speed
        self.truck_capacity = model.trucks[0].capacity
        self.drone_speed = model.get_vehicle_speed('drone')
        self.drone_capacity = model.drones[0].capacity
        self.drone_range = model.drone_range
        self.launch_prep_time = getattr(model, 'launch_prep_time', 0.5)
        self.retrieval_time = getattr(model, 'retrieval_time', 0.5)
        self.drone_wait_threshold = 3.0

    def _compute_drone_mission_timeline(self, current_time, i_node, j_cust, k_node):
        d_ij = self.dist_matrix[i_node, j_cust + 1]
        d_jk = self.dist_matrix[j_cust + 1, k_node]
        if (d_ij + d_jk) > self.drone_range + 1e-6: return None, None, False
        lt = current_time + self.launch_prep_time
        arr_dj = max(lt + (d_ij / self.drone_speed), self.tw_start[j_cust])
        arr_dk = arr_dj + self.service_times[j_cust] + (d_jk / self.drone_speed)
        arr_tk = lt + (self.dist_matrix[i_node, k_node] / self.truck_speed)
        wt = max(0.0, arr_dk - arr_tk)
        if wt > self.drone_wait_threshold: return None, None, False
        return max(arr_tk, arr_dk) + self.retrieval_time, wt, True

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
                    _, j = a; route.customers.append(j); remaining.remove(j)
                    cl += self.demands[j]
                    ct = max(ct + self.dist_matrix[cn, j + 1] / self.truck_speed, self.tw_start[j]) + self.service_times[j]
                    cn = j + 1
                else:
                    _, dj, tk, sk = a
                    li = len(route.customers) - 1
                    ri = li + 1
                    route.customers.append(tk)
                    route.drone_missions.append(DroneMission(self.n_trucks + (truck_id % self.n_drones), [dj], li, ri))
                    remaining.remove(dj); remaining.remove(tk)
                    cl += self.demands[tk]
                    ct = max(sk, self.tw_start[tk]) + self.service_times[tk]
                    cn = tk + 1
            if route.customers: routes.append(route)
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
                remaining.remove(bc)
            else: break
        for r in routes:
            tvs = set(r.customers)
            r.drone_missions = [m for m in r.drone_missions if m.customer_ids[0] not in tvs and m.launch_point < len(r.customers) and m.return_point < len(r.customers)]
        return routes

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
        mc = max(0.001, min(o[0] for _, o in solutions))
        mt = max(0.001, min(o[1] for _, o in solutions))
        for rs, ob in solutions:
            dc = self.Q_c * (mc / max(0.001, ob[0]))
            dt = self.Q_t * (mt / max(0.001, ob[1]))
            for route in rs:
                nds = [0] + [c + 1 for c in route.customers] + [0]
                for i in range(len(nds) - 1):
                    self.phero_truck_c[nds[i], nds[i+1]] += dc
                    self.phero_truck_t[nds[i], nds[i+1]] += dt
                for m in route.drone_missions:
                    i_n = 0 if m.launch_point == -1 else route.customers[m.launch_point] + 1
                    j_n = m.customer_ids[0] + 1
                    k_n = route.customers[m.return_point] + 1 if m.return_point < len(route.customers) else 0
                    key = (i_n, j_n, k_n)
                    self.phero_drone_c[key] = self.phero_drone_c.get(key, self.TAU_INIT) + dc
                    self.phero_drone_t[key] = self.phero_drone_t.get(key, self.TAU_INIT) + dt
        self.phero_truck_c = np.clip(self.phero_truck_c, self.TAU_MIN, self.TAU_MAX)
        self.phero_truck_t = np.clip(self.phero_truck_t, self.TAU_MIN, self.TAU_MAX)

    def _dominates(self, o1, o2):
        return (o1[0] <= o2[0] and o1[1] <= o2[1]) and (o1[0] < o2[0] or o1[1] < o2[1])

    def solve(self) -> Tuple[List[List[Route]], List[Tuple[float, float]]]:
        for it in range(self.max_iter):
            cur = []
            for _ in range(self.n_ants):
                rs = self._construct_solution()
                sd = set()
                for r in rs:
                    sd.update(r.customers)
                    for m in r.drone_missions: sd.update(m.customer_ids)
                ms = self.n_customers - len(sd)
                co, _ = self.model.evaluate_solution(rs)
                ta = self.model.calculate_pure_tardiness(rs)
                if ms > 0:
                    co += ms * 10000.0
                    ta += ms * 10000.0
                cur.append((rs, (co, ta)))
            ac = list(zip(self.pareto_solutions, self.pareto_front))
            for s, o in cur:
                if not any(self._dominates(eo, o) for eo in self.pareto_front):
                    ac.append((s, o))
            nf, ns = [], []
            for i, (si, oi) in enumerate(ac):
                if not any(self._dominates(oj, oi) for j, (sj, oj) in enumerate(ac) if i != j):
                    if not any(abs(oi[0] - eo[0]) < 1e-4 and abs(oi[1] - eo[1]) < 1e-4 for eo in nf):
                        nf.append(oi); ns.append(si)
            self.pareto_front, self.pareto_solutions = nf, ns
            if len(self.pareto_front) > self.archive_capacity:
                cb = list(zip(self.pareto_front, self.pareto_solutions))
                n, cr, fo = len(cb), [0.0] * len(cb), [x[0] for x in cb]
                for d in [0, 1]:
                    ix = sorted(range(n), key=lambda i: fo[i][d])
                    cr[ix[0]] = cr[ix[-1]] = float('inf')
                    sp = fo[ix[-1]][d] - fo[ix[0]][d]
                    if sp > 1e-6:
                        for i in range(1, n - 1):
                            cr[ix[i]] += (fo[ix[i+1]][d] - fo[ix[i-1]][d]) / sp
                si = sorted(range(n), key=lambda i: -cr[i])[:self.archive_capacity]
                self.pareto_front = [cb[i][0] for i in sorted(si)]
                self.pareto_solutions = [cb[i][1] for i in sorted(si)]
            if self.pareto_solutions:
                self._update_pheromones(zip(self.pareto_solutions, self.pareto_front))
            if (it + 1) % 10 == 0:
                print(f"[P-ACO-imp2] Iteration {it + 1}/{self.max_iter} - Archive: {len(self.pareto_front)}")
        return self.pareto_solutions, self.pareto_front
