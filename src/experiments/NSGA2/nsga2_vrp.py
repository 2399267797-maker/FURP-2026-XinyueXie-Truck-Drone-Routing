# 
import random
import numpy as np
from typing import List, Tuple
from deap import base, creator, tools, algorithms
from deap.tools import emo  
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.vrp_model import VRPTruckDroneModel, Route, DroneMission

OVERLOAD_PENALTY = 1000.0

class NSGA2VRP:
    def __init__(self, model: VRPTruckDroneModel, pop_size: int = 100, max_gen: int = 120,
                 cxpb: float = 0.8, mutpb: float = 0.1):
        self.model = model
        
        # ====== DEAP 的 DCD 竞赛要求种群大小必须是 4 的倍数 ======
        if pop_size % 4 != 0:
            self.pop_size = pop_size + (4 - pop_size % 4)
        else:
            self.pop_size = pop_size
        # ==================================================================
            
        self.max_gen = max_gen
        self.cxpb = cxpb
        self.mutpb = mutpb
        
        self.n_customers = model.get_number_of_customers()
        self.n_trucks = model.get_number_of_trucks()
        self.n_drones = model.get_number_of_drones()
        
        self._setup_deap()

    def _setup_deap(self):
        if hasattr(creator, 'FitnessMulti'):
            del creator.FitnessMulti
        if hasattr(creator, 'Individual'):
            del creator.Individual
        
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
        creator.create("Individual", list, fitness=creator.FitnessMulti)
        
        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", self._init_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self._evaluate)
        
        # 严格遵守论文 4)：使用 PMX (部分映射交叉) 和 Scramble 突变
        self.toolbox.register("mate", self._cx_pmx)
        self.toolbox.register("mutate", self._mut_scramble)
        
        # 严格遵守论文 2) 3)：基于非支配等级和拥挤距离的二元竞赛选择（用于父母交配）
        self.toolbox.register("select_mating", tools.selTournamentDCD)

    # def _cx_pmx(self, ind1, ind2):
    #     """论文指定的 PMX 交叉算子 (适配双层编码)"""
    #     n = self.n_customers
    #     # 对客户顺序序列执行 PMX
    #     tools.cxPartialyMatched(ind1[:n], ind2[:n])
    #     # 对模式序列 (无人机/卡车) 执行单点交叉
    #     cxpoint = random.randint(1, n - 1)
    #     ind1[n+cxpoint:], ind2[n+cxpoint:] = ind2[n+cxpoint:], ind1[n+cxpoint:]
    #     return ind1, ind2
    #bug fix
    def _cx_pmx(self, ind1, ind2):
        n = self.n_customers
        # 提取副本进行交叉
        sub1, sub2 = ind1[:n], ind2[:n]
        tools.cxPartialyMatched(sub1, sub2)
        # 必须显式写回原染色体！
        ind1[:n], ind2[:n] = sub1, sub2
        
        # 模式序列交叉
        cxpoint = random.randint(1, n - 1)
        ind1[n+cxpoint:], ind2[n+cxpoint:] = ind2[n+cxpoint:], ind1[n+cxpoint:]
        return ind1, ind2

    def _mut_scramble(self, individual):
        """论文指定的 Scramble 扰乱突变"""
        n = self.n_customers
        # 对路径序列进行片段扰乱
        if random.random() < 0.8:
            idx1, idx2 = sorted(random.sample(range(n + 1), 2))
            sub_list = individual[idx1:idx2]
            random.shuffle(sub_list)
            individual[idx1:idx2] = sub_list
            
        # 模式反转突变 (维持 0,1 多样性)
        if random.random() < 0.4:
            idx = random.randint(n, 2*n - 1)
            individual[idx] = 1 - individual[idx]
        return individual,
    def _select_controlled_elitism(self, population, k, r=0.6):
        """
        论文第 5) 点：受控精英主义 (Controlled Elitism)
        按几何比例 r 分配各前沿的晋级名额，保证侧向多样性。
        """
        fronts = tools.sortNondominated(population, len(population))
        
        selected = []
        remaining = k
        num_fronts = len(fronts)
        
        if num_fronts > 1:
            factor = (1 - r) / (1 - r ** num_fronts)
        else:
            factor = 1.0

        for i, front in enumerate(fronts):
            if remaining <= 0:
                break
                
            target_size = int(round(k * factor * (r ** i)))
            
            if target_size == 0 and len(front) > 0:
                target_size = 1
                
            actual_size = min(len(front), target_size, remaining)
            
            emo.assignCrowdingDist(front)
            front.sort(key=lambda ind: ind.fitness.crowding_dist, reverse=True)
            
            selected.extend(front[:actual_size])
            remaining -= actual_size
            
        return selected
    
    def _cx_ox(self, ind1, ind2):
        n = self.n_customers
        order1, order2 = ind1[:n].copy(), ind2[:n].copy()
        mode1, mode2 = ind1[n:2*n].copy(), ind2[n:2*n].copy()
        
        size = len(order1)
        cxpoint1, cxpoint2 = sorted(random.sample(range(size), 2))
        
        temp1, temp2 = order1[cxpoint1:cxpoint2], order2[cxpoint1:cxpoint2]
        child1, child2 = [None] * size, [None] * size
        child1[cxpoint1:cxpoint2] = temp1
        child2[cxpoint1:cxpoint2] = temp2
        
        ptr1, ptr2 = cxpoint2 % size, cxpoint2 % size
        for i in range(size):
            pos = (cxpoint2 + i) % size
            if order2[pos] not in temp1:
                child1[ptr1] = order2[pos]
                ptr1 = (ptr1 + 1) % size
            if order1[pos] not in temp2:
                child2[ptr2] = order1[pos]
                ptr2 = (ptr2 + 1) % size
        
        mode_cxpoint = random.randint(0, n)
        for i in range(mode_cxpoint, n):
            mode1[i], mode2[i] = mode2[i], mode1[i]
        
        for i in range(n):
            ind1[i], ind2[i] = child1[i], child2[i]
            ind1[n+i], ind2[n+i] = mode1[i], mode2[i]
        
        return ind1, ind2
    
    def _mut_swap_and_flip(self, individual):
        n = self.n_customers
        # 变异一：客户顺序交换
        if random.random() < 0.6:
            i, j = random.sample(range(n), 2)
            individual[i], individual[j] = individual[j], individual[i]
        
        # 变异二：模式反转 (卡车 <-> 无人机)
        if random.random() < 0.4:
            i = random.randint(0, n - 1)
            individual[n + i] = 1 - individual[n + i]
        return individual,

    def _init_individual(self):
        # 编码：N位客户顺序 + N位服务模式(0:卡车, 1:无人机)
        customer_order = list(range(self.n_customers))
        random.shuffle(customer_order)
        mode_assignments = [0 if random.random() < 0.6 else 1 for _ in range(self.n_customers)]
                
        return creator.Individual(customer_order + mode_assignments)

    def _route_load(self, route) -> float:
        return sum(self.model.customers[c].demand for c in route.customers)

    def _total_overload(self, routes) -> float:
        capacity = self.model.trucks[0].capacity
        return sum(max(0.0, self._route_load(r) - capacity) for r in routes)

    def _repair_capacity(self, routes):
        capacity = self.model.trucks[0].capacity
        max_passes = max(40, self.n_customers * 4)
        for _ in range(max_passes):
            overloaded = [r for r in routes if self._route_load(r) > capacity + 1e-6]
            if not overloaded:
                return

            moved = False
            for src in sorted(overloaded, key=lambda r: -self._route_load(r)):
                for pos in sorted(
                        range(len(src.customers)),
                        key=lambda p: -self.model.customers[src.customers[p]].demand):
                    customer = src.customers[pos]
                    demand = self.model.customers[customer].demand
                    targets = [
                        r for r in routes
                        if r is not src and self._route_load(r) + demand <= capacity + 1e-6
                    ]
                    if not targets:
                        continue
                    target = max(
                        targets,
                        key=lambda r: (capacity - self._route_load(r), -len(r.customers)),
                    )
                    src.customers.pop(pos)
                    target.customers.append(customer)
                    moved = True
                    break
                if moved:
                    break
            if moved:
                continue

            # 单点搬不动时尝试交换；仍无法修复说明车队总容量本身不足。
            swapped = False
            for src in sorted(overloaded, key=lambda r: -self._route_load(r)):
                src_load = self._route_load(src)
                for pos in range(len(src.customers)):
                    customer = src.customers[pos]
                    demand = self.model.customers[customer].demand
                    for target in routes:
                        if target is src:
                            continue
                        target_load = self._route_load(target)
                        for tpos in range(len(target.customers)):
                            other = target.customers[tpos]
                            other_demand = self.model.customers[other].demand
                            if (src_load - demand + other_demand <= capacity + 1e-6 and
                                    target_load - other_demand + demand <= capacity + 1e-6):
                                src.customers[pos], target.customers[tpos] = other, customer
                                swapped = True
                                break
                        if swapped:
                            break
                    if swapped:
                        break
                if swapped:
                    break
            if not swapped:
                return

    def _decode_solution(self, individual):
        n = self.n_customers
        order = individual[:n]
        mode = individual[n:2*n]  # 0: 卡车, 1: 无人机
        
        # =====================================================================
        # 论文原生逻辑 (Subtractive Decoding):
        # 1. Truck First: 先把所有节点按顺序全部分配给卡车，形成天然合法的卡车主干
        # =====================================================================
        routes = []
        chunk_size = max(1, len(order) // self.n_trucks)
        for tid in range(self.n_trucks):
            start = tid * chunk_size
            end = len(order) if tid == self.n_trucks - 1 else (tid + 1) * chunk_size
            
            # 初始状态：卡车包揽一切
            route = Route(vehicle_id=tid, vehicle_type='truck', customers=order[start:end].copy())
            routes.append(route)

        # 容量修复：按需求而不是只按客户数量切分路线
        self._repair_capacity(routes)

        # =====================================================================
        # 2. Drone Extraction: 遍历卡车主干，将合法节点“剥离”给无人机
        # =====================================================================
        drone_fail_count = 0
        
        for r_idx, route in enumerate(routes):
            if len(route.customers) < 3: 
                continue
                
            new_truck_customers = []
            missions = []
            
            i = 0
            while i < len(route.customers):
                curr_cust = route.customers[i]
                
                # 尝试构建协同三元组: i(卡车发射) -> i+1(无人机服务) -> i+2(卡车回收)
                if i + 2 < len(route.customers):
                    target_cust = route.customers[i+1]
                    
                    # 查找目标节点在染色体中的 mode 基因，看它是否被算法指定为无人机
                    target_idx_in_gene = order.index(target_cust)
                    is_drone_mode = (mode[target_idx_in_gene] == 1)
                    
                    if is_drone_mode and self.n_drones > 0:
                        l_node = self.model.customers[curr_cust]
                        d_model = self.model.customers[target_cust]
                        r_node = self.model.customers[route.customers[i+2]]
                        
                        d1 = l_node.distance_to(d_model)
                        d2 = d_model.distance_to(r_node)
                        drone_time = (d1 / self.model.get_vehicle_speed('drone')) + (d2 / self.model.get_vehicle_speed('drone'))
                        d_truck = l_node.distance_to(r_node)
                        s_l = getattr(self.model, 'launch_prep_time', 0.5)
                        s_r = getattr(self.model, 'retrieval_time', 0.5)
                        truck_time = d_truck / self.model.get_vehicle_speed('truck') + s_l + s_r
                        
                        # 【核心】物理约束检查
                        if (d1 + d2) <= self.model.drone_range and d_model.demand <= self.model.drones[0].capacity and drone_time <= truck_time:
                            # 满足所有条件！将其从卡车剥离，转化为无人机任务
                            new_truck_customers.append(curr_cust)
                            launch_idx = len(new_truck_customers) - 1
                            return_idx = launch_idx + 1
                            
                            mission = DroneMission(
                                drone_id=self.n_trucks + r_idx,
                                customer_ids=[target_cust],
                                launch_point=launch_idx,
                                return_point=return_idx
                            )
                            missions.append(mission)
                            
                            i += 2  # 跳过被无人机带走的 target_cust(i+1)，下一步直接处理回收点 i+2
                            continue
                        else:
                            # 标记为无人机但不满足约束 → 不降级，直接计失败
                            drone_fail_count += 1
                            
                new_truck_customers.append(curr_cust)
                i += 1
                
            # 更新剥离后的真实卡车节点和无人机任务
            route.customers = new_truck_customers
            route.drone_missions = missions
            
        return routes, drone_fail_count
    def _evaluate(self, individual):
        routes, drone_fail_count = self._decode_solution(individual)
        cost, _ = self.model.evaluate_solution(routes)
        cost += self._total_overload(routes) * OVERLOAD_PENALTY
        # 统一使用 evaluator 级别的延迟计算 (解决问题4：评估指标断层)
        tardiness_penalty = self.model.calculate_pure_tardiness(routes)
        # 无人机失败惩罚：每个标记为无人机但无法执行的客户，加 50 延误惩罚
        DRONE_FAIL_PENALTY = 50.0
        tardiness_penalty += drone_fail_count * DRONE_FAIL_PENALTY
        return (cost, tardiness_penalty)

    def solve(self) -> Tuple[List[List[Route]], List[Tuple[float, float]]]:
        pop = self.toolbox.population(n=self.pop_size)
        
        # 初始评估
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            
        # 赋予初代拥挤距离
        pop = tools.selNSGA2(pop, self.pop_size)
        
        for gen in range(1, self.max_gen + 1):
            # 论文第 3) 点：使用二元竞赛 (TournamentDCD) 选择父母
            k = len(pop) - (len(pop) % 4) if len(pop) >= 4 else len(pop)
            mating_pool = self.toolbox.select_mating(pop, k)
            mating_pool = list(map(self.toolbox.clone, mating_pool))
            
            # 论文第 4) 点：PMX交叉与Scramble突变生成后代
            offspring = algorithms.varAnd(mating_pool, self.toolbox, self.cxpb, self.mutpb)
            
            # 评估后代 (包含亚组的重新初始化)
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # 论文第 5) 点：使用受控精英主义 (Controlled Elitism) 筛选存活者
            combined_pop = pop + offspring
            pop = self._select_controlled_elitism(combined_pop, self.pop_size, r=0.6)
            
            if gen % 20 == 0:
                print(f"Generation {gen}/{self.max_gen}, Population size: {len(pop)}")
        
        pareto_solutions = []
        pareto_front = []
        all_solutions = []
        all_objectives = []
        
        for ind in pop:
            routes, drone_fail_count = self._decode_solution(ind)
            cost, _ = self.model.evaluate_solution(routes)
            cost += self._total_overload(routes) * OVERLOAD_PENALTY
            tardiness_penalty = self.model.calculate_pure_tardiness(routes)
            DRONE_FAIL_PENALTY = 50.0
            tardiness_penalty += drone_fail_count * DRONE_FAIL_PENALTY
            
            # 公平对比口径：返回前沿只保留可行解（无缺客户、无超载），
            # 与 PACO / PACO+ALNS / Pure ALNS 的档案语义一致；搜索阶段仍使用惩罚适应度。
            served = set()
            for r in routes:
                served.update(r.customers)
                for m in r.drone_missions:
                    served.update(m.customer_ids)
            missing = self.n_customers - len(served)
            overload = self._total_overload(routes)
            if missing > 0 or overload > 1e-6:
                continue
            
            all_solutions.append(routes)
            all_objectives.append((cost, tardiness_penalty))
        
        # 提取最终全局帕累托前沿
        for i in range(len(all_objectives)):
            dominated = False
            for j in range(len(all_objectives)):
                if i == j: continue
                if (all_objectives[j][0] <= all_objectives[i][0] and 
                    all_objectives[j][1] <= all_objectives[i][1] and
                    (all_objectives[j][0] < all_objectives[i][0] or 
                     all_objectives[j][1] < all_objectives[i][1])):
                    dominated = True
                    break
            if not dominated:
                pareto_solutions.append(all_solutions[i])
                pareto_front.append(all_objectives[i])
        
        return pareto_solutions, pareto_front
