import random, numpy as np
from typing import List, Tuple
from deap import base, creator, tools, algorithms
# emo imported via inline function
import sys, os as osmod
sys.path.insert(0, osmod.path.dirname(osmod.path.dirname(osmod.path.abspath(__file__))))
from models.vrp_model import VRPTruckDroneModel, Route, DroneMission

class NSGA2VRP:
    def __init__(self, model, pop_size=100, max_gen=120, cxpb=0.8, mutpb=0.1):
        self.model = model
        self.pop_size = pop_size if pop_size % 4 == 0 else pop_size + (4 - pop_size % 4)
        self.max_gen, self.cxpb, self.mutpb = max_gen, cxpb, mutpb
        self.n = model.get_number_of_customers()
        self.n_trucks, self.n_drones = model.get_number_of_trucks(), model.get_number_of_drones()
        self.max_drone_gap = 4
        self.drone_wait_threshold = 3.0
        self.ol_penalty = 50.0
        self.df_penalty = 50.0
        self._setup_deap()

    def _setup_deap(self):
        for a in ['FitnessMulti', 'Individual']:
            if hasattr(creator, a): delattr(creator, a)
        creator.create('FitnessMulti', base.Fitness, weights=(-1.0, -1.0))
        creator.create('Individual', list, fitness=creator.FitnessMulti)
        tb = base.Toolbox()
        tb.register('individual', self._init_ind)
        tb.register('population', tools.initRepeat, list, tb.individual)
        tb.register('evaluate', self._eval)
        tb.register('mate', self._cx)
        tb.register('mutate', self._mut)
        tb.register('select_mating', tools.selTournamentDCD)
        self.toolbox = tb

    def _init_ind(self):
        n = self.n
        o = list(range(n)); random.shuffle(o)
        m = [0 if random.random() < 0.6 else 1 for _ in range(n)]
        r = [random.randint(0, self.n_trucks - 1) for _ in range(n)]
        return creator.Individual(o + m + r)

    def _decode(self, ind):
        n = self.n
        order, mode, assign = ind[:n], ind[n:2*n], ind[2*n:3*n]
        rc = {t: [] for t in range(self.n_trucks)}
        for pos, c in enumerate(order):
            rc[assign[pos] % self.n_trucks].append(c)
        routes = [Route(t, 'truck', rc[t]) for t in range(self.n_trucks)]
        ol_pen = 0.0
        cap = self.model.trucks[0].capacity
        for r in routes:
            ld = sum(self.model.customers[c].demand for c in r.customers)
            if ld > cap + 1e-6:
                ol_pen += (ld - cap) * self.ol_penalty
        c2m = {}
        for pos, c in enumerate(order):
            c2m[c] = mode[pos]
        dfc = 0
        for ri, rt in enumerate(routes):
            if len(rt.customers) < 3: continue
            new_t, miss = [], []
            i = 0
            while i < len(rt.customers):
                cur = rt.customers[i]
                conv = False
                for gap in range(1, self.max_drone_gap + 1):
                    k = i + gap + 1
                    if k >= len(rt.customers): break
                    tgt = rt.customers[i + gap]
                    if c2m.get(tgt, 0) != 1: continue
                    ln = self.model.customers[cur]
                    dn = self.model.customers[tgt]
                    rn = self.model.customers[rt.customers[k]]
                    d1, d2 = ln.distance_to(dn), dn.distance_to(rn)
                    if d1 + d2 > self.model.drone_range + 1e-6: continue
                    if dn.demand > self.model.drones[0].capacity + 1e-6: continue
                    dspeed = self.model.get_vehicle_speed('drone')
                    dt = (d1 / dspeed) + dn.service_time + (d2 / dspeed)
                    tt = ln.distance_to(rn) / self.model.get_vehicle_speed('truck')
                    wait = max(0.0, dt - tt)
                    if wait > self.drone_wait_threshold: continue
                    new_t.append(cur)
                    li = len(new_t) - 1
                    miss.append(DroneMission(self.n_trucks + (ri % max(1, self.n_drones)), [tgt], li, li + 1))
                    i = k; conv = True; break
                if not conv:
                    if c2m.get(cur, 0) == 1: dfc += 1
                    new_t.append(cur)
                    i += 1
            rt.customers, rt.drone_missions = new_t, miss
        return routes, dfc, ol_pen

    def _eval(self, ind):
        r, dfc, olp = self._decode(ind)
        c, _ = self.model.evaluate_solution(r)
        t = self.model.calculate_pure_tardiness(r) + dfc * self.df_penalty
        return (c + olp, t)

    def _cx(self, i1, i2):
        n = self.n
        o1, o2 = i1[:n].copy(), i2[:n].copy()
        tools.cxPartialyMatched(o1, o2)
        i1[:n], i2[:n] = o1, o2
        cp = random.randint(1, n - 1)
        i1[n+cp:2*n], i2[n+cp:2*n] = i2[n+cp:2*n], i1[n+cp:2*n]
        cp2 = random.randint(1, n - 1)
        i1[2*n+cp2:3*n], i2[2*n+cp2:3*n] = i2[2*n+cp2:3*n], i1[2*n+cp2:3*n]
        return i1, i2

    def _mut(self, ind):
        n = self.n
        if random.random() < 0.8:
            a, b = sorted(random.sample(range(n), 2))
            seg = ind[a:b]; random.shuffle(seg); ind[a:b] = seg
        if random.random() < 0.6:
            for _ in range(random.randint(1, max(1, n // 10))):
                ind[n + random.randint(0, n - 1)] = 1 - ind[n + random.randint(0, n - 1)]
        if random.random() < 0.4:
            for _ in range(random.randint(1, max(1, n // 8))):
                ind[2*n + random.randint(0, n - 1)] = random.randint(0, self.n_trucks - 1)
        return ind,

    def _select_elite(self, pop, k, r=0.6):
        fronts = tools.sortNondominated(pop, len(pop))
        sel, rem = [], k
        nf = len(fronts)
        fct = (1 - r) / (1 - r ** nf) if nf > 1 else 1.0
        for i, f in enumerate(fronts):
            if rem <= 0: break
            tg = max(1, int(round(k * fct * (r ** i))))
            ac = min(len(f), tg, rem)
            # inline assignCrowdingDist (PyPy compatible)
            if f:
                for ind in f:
                    ind.fitness.crowding_dist = 0.0
                nobj = len(f[0].fitness.values)
                for oi in range(nobj):
                    f.sort(key=lambda x, idx=oi: x.fitness.values[idx])
                    f[0].fitness.crowding_dist = float('inf')
                    f[-1].fitness.crowding_dist = float('inf')
                    sp = f[-1].fitness.values[oi] - f[0].fitness.values[oi]
                    if sp > 1e-10:
                        for j_ in range(1, len(f)-1):
                            f[j_].fitness.crowding_dist += (f[j_+1].fitness.values[oi] - f[j_-1].fitness.values[oi]) / sp
            f.sort(key=lambda x: x.fitness.crowding_dist, reverse=True)
            sel.extend(f[:ac]); rem -= ac
        return sel

    def solve(self):
        pop = self.toolbox.population(n=self.pop_size)
        inv = [ind for ind in pop if not ind.fitness.valid]
        for ind, ft in zip(inv, self.toolbox.map(self.toolbox.evaluate, inv)):
            ind.fitness.values = ft
        pop = tools.selNSGA2(pop, self.pop_size)
        for g in range(1, self.max_gen + 1):
            k = max(4, len(pop) - (len(pop) % 4))
            pool = list(map(self.toolbox.clone, self.toolbox.select_mating(pop, k)))
            off = algorithms.varAnd(pool, self.toolbox, self.cxpb, self.mutpb)
            inv = [ind for ind in off if not ind.fitness.valid]
            for ind, ft in zip(inv, self.toolbox.map(self.toolbox.evaluate, inv)):
                ind.fitness.values = ft
            pop = self._select_elite(pop + off, self.pop_size)
            if g % 20 == 0:
                print(f"[NSGA2-imp] Gen {g}/{self.max_gen}, Pop: {len(pop)}")
        all_s, all_o = [], []
        for ind in pop:
            r, dfc, olp = self._decode(ind)
            c, _ = self.model.evaluate_solution(r)
            t = self.model.calculate_pure_tardiness(r) + dfc * self.df_penalty
            all_s.append(r); all_o.append((c + olp, t))
        pf_s, pf_o = [], []
        for i in range(len(all_o)):
            if not any(all_o[j][0] <= all_o[i][0] and all_o[j][1] <= all_o[i][1] and (all_o[j][0] < all_o[i][0] or all_o[j][1] < all_o[i][1]) for j in range(len(all_o)) if i != j):
                pf_s.append(all_s[i]); pf_o.append(all_o[i])
        return pf_s, pf_o
