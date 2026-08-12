"""Pure ALNS solver for the truck-drone routing model.

The solver reuses the destroy/repair operator library and the adaptive-weight
simulated-annealing loop implemented in PACO+ALNS W8, but removes every
PACO-specific component: no ant construction, no pheromone matrix, and no warm
starts from the Pareto archive.  Each restart builds an initial solution with a
randomized greedy insertion heuristic (the same insertion machinery used by
the ALNS repair operators) and then runs the ALNS local search.

The configuration is intentionally the same as W8's ALNS component: the number
of restarts defaults to max_iter (the same outer-iteration budget as
CollaborativePACOALNS.solve), alns_iter uses the same scale-adaptive default,
and the scalarized acceptance score uses the same fixed tard_penalty_truck.
Only the PACO construction and pheromone update are removed.

The solve() interface matches CollaborativePACOALNS so the same evaluation
harness can be used for comparison experiments.
"""

import os
import random
import sys
import importlib.util
from typing import List, Tuple


BASE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(os.path.dirname(BASE)))
sys.path.insert(0, os.path.join(PROJ, 'src', 'experiments', 'PACO_vs_NSGA2'))

from models.vrp_model import Route  # noqa: E402


def _load_w8_module():
    path = os.path.join(PROJ, 'src', 'experiments', 'PACO+ALNS', 'PACO+ALNSW8.py')
    spec = importlib.util.spec_from_file_location('paco_alns_w8_pure', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_W8 = _load_w8_module()


class PureALNS(_W8.CollaborativePACOALNS):
    """ALNS-only solver: greedy insertion construction + ALNS improvement."""

    def __init__(self, model, max_iter: int = 100, n_starts: int = None,
                 alns_iter: int = None):
        super().__init__(model, max_iter=max_iter, alns_iter=alns_iter)
        self.max_iter = max_iter  # kept for interface compatibility
        self.n_starts = n_starts if n_starts is not None else max(1, max_iter)
        self.pareto_front = []
        self.pareto_solutions = []

    def _construct_greedy(self) -> List[Route]:
        """Randomized greedy insertion starting from empty truck routes."""
        routes = [Route(truck_id, 'truck') for truck_id in range(self.n_trucks)]
        unassigned = list(range(self.n_customers))
        random.shuffle(unassigned)
        for c in unassigned:
            insertions = self._get_all_insertions(routes, c)
            best_action = insertions[0][1] if insertions else None
            self._apply_insertion(routes, best_action, c)
        return routes

    def _archive_objectives(self, routes: List[Route]) -> Tuple[float, float, int, float]:
        served = set()
        for r in routes:
            served.update(r.customers)
            for m in r.drone_missions:
                served.update(m.customer_ids)
        missing = self.n_customers - len(served)
        overload = sum(
            max(0.0, sum(self.demands[c] for c in r.customers) - self.truck_capacity)
            for r in routes
        )
        cost, _ = self._eval_solution(routes)
        tard = self._calc_tardiness(routes)
        cost += missing * 10000.0 + overload * 1000.0
        tard += missing * 10000.0
        return float(cost), float(tard), int(missing), float(overload)

    def _add_to_archive(self, routes: List[Route], cost: float, tard: float):
        if any(self._dominates(ex_obj, (cost, tard)) for ex_obj in self.pareto_front):
            return
        all_candidates = list(zip(self.pareto_solutions, self.pareto_front))
        all_candidates.append((self._clone_routes(routes), (cost, tard)))

        new_front, new_solutions = [], []
        for i, (sol_i, obj_i) in enumerate(all_candidates):
            if any(self._dominates(obj_j, obj_i)
                   for j, (_, obj_j) in enumerate(all_candidates) if j != i):
                continue
            if any(abs(obj_i[0] - eo[0]) < 1e-4 and abs(obj_i[1] - eo[1]) < 1e-4
                   for eo in new_front):
                continue
            new_front.append(obj_i)
            new_solutions.append(sol_i)
        self.pareto_front, self.pareto_solutions = new_front, new_solutions

        if len(self.pareto_front) <= self.archive_capacity:
            return
        combined = list(zip(self.pareto_front, self.pareto_solutions))
        n, crowd = len(combined), [0.0] * len(combined)
        for d in (0, 1):
            idx_sort = sorted(range(n), key=lambda i: combined[i][0][d])
            crowd[idx_sort[0]] = crowd[idx_sort[-1]] = float('inf')
            span = combined[idx_sort[-1]][0][d] - combined[idx_sort[0]][0][d]
            if span > 1e-6:
                for i in range(1, n - 1):
                    crowd[idx_sort[i]] += (
                        combined[idx_sort[i + 1]][0][d]
                        - combined[idx_sort[i - 1]][0][d]
                    ) / span
        kept = sorted(range(n), key=lambda i: -crowd[i])[:self.archive_capacity]
        kept = sorted(kept)
        self.pareto_front = [combined[i][0] for i in kept]
        self.pareto_solutions = [combined[i][1] for i in kept]

    def solve(self) -> Tuple[List[List[Route]], List[Tuple[float, float]]]:
        global_best_score = float('inf')
        n_starts = max(1, self.n_starts)

        for start in range(n_starts):
            routes = self._construct_greedy()
            for r in routes:
                self._two_opt_route(r)
            routes = self._inter_route_relocate(routes)
            routes = self._alns_local_search(routes, global_best_score)

            cost, tard, missing, overload = self._archive_objectives(routes)
            score = cost + tard * self.tard_penalty_truck
            if score < global_best_score:
                global_best_score = score
            if missing == 0 and overload <= 1e-6:
                self._add_to_archive(routes, cost, tard)

        return self.pareto_solutions, self.pareto_front
