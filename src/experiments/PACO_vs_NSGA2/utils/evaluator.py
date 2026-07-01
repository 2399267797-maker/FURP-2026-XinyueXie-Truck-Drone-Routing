import numpy as np
from typing import List, Tuple
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.vrp_model import VRPTruckDroneModel, Route


class Evaluator:
    """
    Unified evaluation module for multi-objective truck-drone routing problem.
    
    Objectives (both minimization):
        obj[0] = total travel cost (Cost of Travel)
        obj[1] = total tardiness penalty (Penalty due to tardiness)
    
    Provides Pareto front extraction and standard multi-objective performance metrics.
    All calculations are aligned with the formulation in Das et al. (2021).
    """
    def __init__(self, model: VRPTruckDroneModel):
        self.model = model

    # ---------- Basic solution evaluation ----------
    def evaluate_solution(self, routes: List[Route]) -> Tuple[float, float]:
        """
        Evaluate a single solution (list of routes) with full penalty rules.
        
        Returns:
            (total_cost, total_tardiness_penalty)
        """
        return self.model.evaluate_multi_objective(routes)

    def calculate_pure_tardiness(self, routes: List[Route]) -> float:
        """
        Calculate raw priority-weighted tardiness penalty.
        Delegates to the unified implementation in the model.
        """
        return self.model.calculate_pure_tardiness(routes)

    def is_feasible(self, routes: List[Route]) -> bool:
        """Check whether a solution satisfies all constraints (capacity, sync, range, coverage)."""
        return self.model.is_solution_feasible(routes)

    # ---------- Pareto front extraction ----------
    def calculate_pareto_front(self, solutions: List[List[Route]]) -> List[Tuple[float, float]]:
        """
        Extract non-dominated Pareto front from a set of solutions.
        
        Args:
            solutions: list of solutions, each solution is a list of Route objects
        
        Returns:
            List of objective vectors on the Pareto front
        """
        objectives_list = [self.evaluate_solution(sol) for sol in solutions]
        pareto_front = []
        
        for i, obj_i in enumerate(objectives_list):
            dominated = False
            for j, obj_j in enumerate(objectives_list):
                if i != j and self._dominates(obj_j, obj_i):
                    dominated = True
                    break
            if not dominated:
                pareto_front.append(obj_i)
        
        return pareto_front

    def _dominates(self, obj1: Tuple[float, float], obj2: Tuple[float, float]) -> bool:
        """
        Pareto dominance check for minimization problem.
        
        Returns True if obj1 dominates obj2
        (obj1 is better or equal on all objectives and strictly better on at least one).
        """
        all_better = obj1[0] <= obj2[0] and obj1[1] <= obj2[1]
        strictly_better = obj1[0] < obj2[0] or obj1[1] < obj2[1]
        return all_better and strictly_better

    # ---------- Multi-objective performance metrics ----------
    def calculate_hypervolume(self, pareto_front: List[Tuple[float, float]],
                              reference_point: Tuple[float, float]) -> float:
        """
        Calculate Hypervolume (HV) indicator for 2D minimization problem.
        
        Args:
            pareto_front: list of (cost, tardiness) objective vectors
            reference_point: (max_cost, max_tardiness), must be worse than all front points
        
        Returns:
            Hypervolume value (larger = better)
        """
        if not pareto_front:
            return 0.0

        # Sort by cost ascending
        points = sorted(pareto_front, key=lambda x: x[0])
        ref_x, ref_y = reference_point
        hv = 0.0
        prev_x = ref_x

        # Accumulate rectangle areas from right to left
        for x, y in reversed(points):
            if x >= ref_x or y >= ref_y:
                continue
            width = prev_x - x
            height = ref_y - y
            if width > 0 and height > 0:
                hv += width * height
            prev_x = x

        return hv

    def calculate_spacing(self, pareto_front: List[Tuple[float, float]]) -> float:
        """
        Calculate Spacing metric: uniformity of solution distribution on Pareto front.
        
        Returns:
            Spacing value (smaller = more evenly distributed)
        """
        if len(pareto_front) <= 1:
            return 0.0

        distances = []
        for i, obj_i in enumerate(pareto_front):
            min_dist = float('inf')
            for j, obj_j in enumerate(pareto_front):
                if i != j:
                    dist = np.sqrt((obj_i[0] - obj_j[0]) ** 2 + (obj_i[1] - obj_j[1]) ** 2)
                    if dist < min_dist:
                        min_dist = dist
            distances.append(min_dist)

        mean_dist = np.mean(distances)
        spacing = np.sqrt(np.sum((d - mean_dist) ** 2 for d in distances) / len(distances))
        return spacing

    def calculate_generational_distance(self, pareto_front: List[Tuple[float, float]],
                                       true_front: List[Tuple[float, float]]) -> float:
        """
        Calculate Generational Distance (GD): how close the obtained front is to the true Pareto front.
        
        Args:
            pareto_front: obtained approximate Pareto front
            true_front: reference / true Pareto front
        
        Returns:
            GD value (smaller = closer to true front)
        """
        if not pareto_front or not true_front:
            return float('inf')

        total_dist = 0.0
        for obj in pareto_front:
            min_dist = min(np.sqrt((obj[0] - t[0]) ** 2 + (obj[1] - t[1]) ** 2) for t in true_front)
            total_dist += min_dist
        return total_dist / len(pareto_front)

    def calculate_coverage(self, front_a: List[Tuple[float, float]],
                           front_b: List[Tuple[float, float]]) -> float:
        """
        Coverage metric C(A, B): fraction of solutions in B that are dominated by A.
        
        Returns:
            Ratio in [0, 1]; larger = A is stronger relative to B
        """
        if not front_b:
            return 0.0

        covered = 0
        for obj_b in front_b:
            for obj_a in front_a:
                if self._dominates(obj_a, obj_b):
                    covered += 1
                    break
        return covered / len(front_b)

    # ---------- Solution summary statistics ----------
    def summarize_solution(self, routes: List[Route]) -> dict:
        """
        Generate a structured summary dictionary for a single solution.
        
        Includes: cost, tardiness, distance, route count, customer coverage, feasibility.
        """
        summary = {
            'total_cost': 0.0,
            'total_tardiness': 0.0,
            'total_distance': 0.0,
            'number_of_routes': len(routes),
            'truck_routes': 0,
            'drone_mission_count': 0,
            'truck_served_customers': 0,
            'drone_served_customers': 0,
            'total_served': 0,
            'unserved_customers': self.model.get_number_of_customers(),
            'feasible': False
        }

        served = set()

        for route in routes:
            if route.vehicle_type == 'truck':
                summary['truck_routes'] += 1
                summary['drone_mission_count'] += len(route.drone_missions)

            summary['total_distance'] += route.total_distance(self.model)

            # Count customers served directly by the vehicle
            served.update(route.customers)
            summary['truck_served_customers'] += len(route.customers)

            # Count customers served by attached drone missions
            for mission in route.drone_missions:
                served.add(mission.customer_id)
                summary['drone_served_customers'] += 1

        total_cost, total_tardiness = self.evaluate_solution(routes)
        summary['total_cost'] = total_cost
        summary['total_tardiness'] = total_tardiness

        summary['total_served'] = len(served)
        summary['unserved_customers'] = self.model.get_number_of_customers() - len(served)
        summary['feasible'] = self.is_feasible(routes)

        return summary