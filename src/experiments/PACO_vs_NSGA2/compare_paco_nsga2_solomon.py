"""
P-ACO vs NSGA-II Comparison using Solomon RC Benchmark Dataset.

Uses Solomon RC series benchmark dataset with linear scaling for urban logistics.
Customer scales: 25, 50 customers.
Pareto front: Cost of Travel vs Penalty due to tardiness.
Multi-run experiments (10 repetitions) for statistical significance.
Includes No-Drone baseline (pure truck).
"""
import os
import sys
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from models.vrp_model import VRPTruckDroneModel, Customer, Vehicle, Route
from utils.visualizer import Visualizer
from utils.evaluator import Evaluator
# Add algorithm paths
paco_path = r'C:\Users\23992\FURP-2026-XinyueXie-Truck-Drone-Routing\src\experiments\PACO'
nsga2_path = r'C:\Users\23992\FURP-2026-XinyueXie-Truck-Drone-Routing\src\experiments\NSGA2'
sys.path.insert(0, paco_path)
sys.path.insert(0, nsga2_path)

from algorithms.paco import CollaborativePACO
from data.solomon_loader import SolomonLoader

# Import NSGA2
import importlib.util
nsga2_spec = importlib.util.spec_from_file_location("nsga2_vrp", os.path.join(nsga2_path, "nsga2_vrp.py"))
nsga2_module = importlib.util.module_from_spec(nsga2_spec)
nsga2_spec.loader.exec_module(nsga2_module)
NSGA2VRP = nsga2_module.NSGA2VRP


def load_solomon_instance(n_customers: int, rc_type: str = 'RC1', instance_id: int = 1,
                          n_vehicles: int = 2, endurance_type: str = 'medium',
                          use_drones: bool = True) -> VRPTruckDroneModel:
    """Load Solomon RC benchmark instance.
    
    Args:
        n_customers: Number of customers (25 or 50)
        rc_type: 'RC1' (short horizon) or 'RC2' (long horizon)
        instance_id: Instance number (1-8)
        n_vehicles: Number of trucks
        endurance_type: 'medium' (4km) or 'high' (6km) drone endurance
        use_drones: If False, creates pure truck model with 0 drones
    """
    loader = SolomonLoader()
    
    if rc_type == 'RC1':
        return loader.load_rc1_instance(n_customers, instance_id, n_vehicles, endurance_type, use_drones)
    else:
        return loader.load_rc2_instance(n_customers, instance_id, n_vehicles, endurance_type, use_drones)


def calculate_tardiness_penalty(model: VRPTruckDroneModel, routes: List[Route]) -> float:
    """
    Calculate total tardiness penalty for routes via unified Evaluator.
    Tardiness = max(0, arrival_time - time_window_end) * priority
    """
    evaluator = Evaluator(model)
    return evaluator.calculate_pure_tardiness(routes)


def run_paco_single(model: VRPTruckDroneModel, n_ants: int = 50, max_iter: int = 200) -> Dict:
    """Run single P-ACO trial and calculate metrics."""
    start_time = time.time()
    paco = CollaborativePACO(model, n_ants=n_ants, max_iter=max_iter)
    solutions, _ = paco.solve()
    solve_time = time.time() - start_time
    
    costs = []
    tardiness_penalties = []
    pareto_front = []
    for sol in solutions:
        cost, _ = model.evaluate_solution(sol)
        tardiness = calculate_tardiness_penalty(model, sol)
        costs.append(cost)
        tardiness_penalties.append(tardiness)
        pareto_front.append((cost, tardiness))
    
    hv = calculate_hypervolume(pareto_front)
    
    return {
        'solutions': solutions,
        'pareto_front': pareto_front,
        'costs': costs,
        'tardiness_penalties': tardiness_penalties,
        'solve_time': solve_time,
        'hypervolume': hv
    }


def calculate_hypervolume(pareto_front, ref_cost=None, ref_tardiness=None):
    """Calculate hypervolume indicator for 2D Pareto front (min-cost, min-tardiness).

    Args:
        pareto_front: list of (cost, tardiness) tuples.
        ref_cost: reference point x-coordinate. If None, auto = max_x * 1.1.
        ref_tardiness: reference point y-coordinate. If None, auto = max_y * 1.1.

    When ref_cost/ref_tardiness are provided, all algorithms in the same experiment
    share the same reference point, making HV values directly comparable.
    """
    if not pareto_front:
        return 0.0

    points = np.array(pareto_front)
    if len(points) == 0:
        return 0.0

    if ref_cost is None:
        ref_cost = np.max(points[:, 0]) * 1.1
    if ref_tardiness is None:
        ref_tardiness = np.max(points[:, 1]) * 1.1

    points = points[np.lexsort((points[:, 1], points[:, 0]))]

    hypervolume = 0.0
    prev_x = ref_cost

    for i in range(len(points) - 1, -1, -1):
        x, y = points[i]
        width = prev_x - x
        height = ref_tardiness - y
        if width > 0 and height > 0:
            hypervolume += width * height
        prev_x = x

    return hypervolume


def _compute_shared_reference_point(*results):
    """Compute a shared HV reference point across all algorithms in one experiment.

    Takes the max of all Pareto fronts' extremes + 10% buffer, so every algorithm
    is evaluated against the same reference point.
    """
    all_costs = []
    all_tardiness = []
    for result in results:
        if result is None:
            continue
        pf = result.get('all_pareto_fronts', [])
        if not pf:
            continue
        # pf is a list of per-run Pareto fronts (each a list of (cost, tardiness) tuples)
        for run_pf in pf:
            if run_pf:
                pts = np.array(run_pf)
                all_costs.extend(pts[:, 0].tolist())
                all_tardiness.extend(pts[:, 1].tolist())
    if not all_costs:
        return 200.0, 200.0
    return np.max(all_costs) * 1.1, np.max(all_tardiness) * 1.1


def run_nsga2_single(model: VRPTruckDroneModel, pop_size: int = 100, max_gen: int = 120) -> Dict:
    """Run single NSGA-II trial and calculate metrics."""
    start_time = time.time()
    nsga2 = NSGA2VRP(model, pop_size=pop_size, max_gen=max_gen)
    solutions, pareto_front = nsga2.solve()
    solve_time = time.time() - start_time
    
    costs = []
    tardiness_penalties = []
    for sol in solutions:
        cost, _ = model.evaluate_solution(sol)
        tardiness = calculate_tardiness_penalty(model, sol)
        costs.append(cost)
        tardiness_penalties.append(tardiness)
    
    hv = calculate_hypervolume(pareto_front)
    
    return {
        'solutions': solutions,
        'pareto_front': pareto_front,
        'costs': costs,
        'tardiness_penalties': tardiness_penalties,
        'solve_time': solve_time,
        'hypervolume': hv
    }


def run_paco_experiment(model: VRPTruckDroneModel, n_ants: int = 50, max_iter: int = 200,
                        n_runs: int = 10) -> Dict:
    """Run P-ACO experiment with multiple repetitions."""
    print("\n--- P-ACO ---")
    
    all_costs = []
    all_tardiness = []
    all_solve_times = []
    all_hypervolumes = []
    all_solutions = []
    all_pareto_fronts = []
    
    for run_idx in range(n_runs):
        print(f"  Run {run_idx+1}/{n_runs}...", end='')
        result = run_paco_single(model, n_ants, max_iter)
        print(f" done")
        
        all_costs.extend(result['costs'])
        all_tardiness.extend(result['tardiness_penalties'])
        all_solve_times.append(result['solve_time'])
        all_hypervolumes.append(result['hypervolume'])
        all_solutions.extend(result['solutions'])
        all_pareto_fronts.append(result['pareto_front'])
    
    mean_cost = np.mean(all_costs)
    std_cost = np.std(all_costs)
    mean_tardiness = np.mean(all_tardiness)
    best_tardiness =np.min(all_tardiness)
    std_tardiness = np.std(all_tardiness)
    mean_solve_time = np.mean(all_solve_times)
    mean_hv = np.mean(all_hypervolumes)
    std_hv = np.std(all_hypervolumes)
    median_hv = np.median(all_hypervolumes)
    
    flat_pf = [pt for pf in all_pareto_fronts for pt in pf]
    makespans = [f[0] for f in flat_pf]
    satisfactions = [f[1] for f in flat_pf]
    best_makespan = min(makespans) if makespans else float('inf')
    best_satisfaction = max(satisfactions) if satisfactions else 0.0
    
    print(f"  Total Solutions: {len(all_solutions)}")
    print(f"  Best Cost: {best_makespan:.2f}")
    print(f"  best Tardiness:{best_tardiness:.2f}")
    print(f"  Best Satisfaction: {best_satisfaction:.4f}")
    print(f"  Mean Cost: {mean_cost:.2f} ± {std_cost:.2f}")
    print(f"  Mean Tardiness: {mean_tardiness:.2f} ± {std_tardiness:.2f}")
    print(f"  Mean Hypervolume: {mean_hv:.2f} ± {std_hv:.2f}")
    print(f"  Median Hypervolume: {median_hv:.2f}")
    print(f"  Avg Solve Time: {mean_solve_time:.2f}s")
    
    return {
        'n_runs': n_runs,
        'mean_cost': mean_cost,
        'std_cost': std_cost,
        'mean_tardiness': mean_tardiness,
        'std_tardiness': std_tardiness,
        'mean_solve_time': mean_solve_time,
        'best_makespan': best_makespan,
        'best_satisfaction': best_satisfaction,
        'mean_hv': mean_hv,
        'std_hv': std_hv,
        'median_hv': median_hv,
        'all_hypervolumes': all_hypervolumes,
        'all_costs': all_costs,
        'all_tardiness': all_tardiness,
        'all_solutions': all_solutions,
        'all_pareto_fronts': all_pareto_fronts
    }


def run_nsga2_experiment(model: VRPTruckDroneModel, pop_size: int = 100, max_gen: int = 120,
                         n_runs: int = 10) -> Dict:
    """Run NSGA-II experiment with multiple repetitions."""
    print("\n--- NSGA-II ---")
    
    all_costs = []
    all_tardiness = []
    all_solve_times = []
    all_hypervolumes = []
    all_solutions = []
    all_pareto_fronts = []
    
    for run_idx in range(n_runs):
        print(f"  Run {run_idx+1}/{n_runs}...", end='')
        result = run_nsga2_single(model, pop_size, max_gen)
        print(f" done")
        
        all_costs.extend(result['costs'])
        all_tardiness.extend(result['tardiness_penalties'])
        all_solve_times.append(result['solve_time'])
        all_hypervolumes.append(result['hypervolume'])
        all_solutions.extend(result['solutions'])
        all_pareto_fronts.append(result['pareto_front'])
    
    mean_cost = np.mean(all_costs)
    std_cost = np.std(all_costs)
    mean_tardiness = np.mean(all_tardiness)
    std_tardiness = np.std(all_tardiness)
    mean_solve_time = np.mean(all_solve_times)
    mean_hv = np.mean(all_hypervolumes)
    std_hv = np.std(all_hypervolumes)
    median_hv = np.median(all_hypervolumes)
    best_tardiness = np.min(all_tardiness)
    
    flat_pf = [pt for pf in all_pareto_fronts for pt in pf]
    makespans = [f[0] for f in flat_pf]
    satisfactions = [f[1] for f in flat_pf]
    best_makespan = min(makespans) if makespans else float('inf')
    best_satisfaction = max(satisfactions) if satisfactions else 0.0
    
    print(f"  Total Solutions: {len(all_solutions)}")
    print(f"  Best Cost: {best_makespan:.2f}")
    print(f"  Best Tardiness: {best_tardiness:.4f}")
    print(f"  Mean Cost: {mean_cost:.2f} ± {std_cost:.2f}")
    print(f"  Mean Tardiness: {mean_tardiness:.2f} ± {std_tardiness:.2f}")
    print(f"  Mean Hypervolume: {mean_hv:.2f} ± {std_hv:.2f}")
    print(f"  Median Hypervolume: {median_hv:.2f}")
    print(f"  Avg Solve Time: {mean_solve_time:.2f}s")
    
    return {
        'n_runs': n_runs,
        'mean_cost': mean_cost,
        'std_cost': std_cost,
        'mean_tardiness': mean_tardiness,
        'std_tardiness': std_tardiness,
        'mean_solve_time': mean_solve_time,
        'best_makespan': best_makespan,
        'best_satisfaction': best_satisfaction,
        'mean_hv': mean_hv,
        'std_hv': std_hv,
        'median_hv': median_hv,
        'all_hypervolumes': all_hypervolumes,
        'all_costs': all_costs,
        'all_tardiness': all_tardiness,
        'all_solutions': all_solutions,
        'all_pareto_fronts': all_pareto_fronts
    }


def run_no_drone_experiment(model: VRPTruckDroneModel, n_runs: int = 10) -> Dict:
    """Run No-Drone baseline experiment with multiple repetitions.
    
    Uses NSGA-II without drones (pure truck routing).
    """
    print("\n--- No-Drone Baseline ---")
    
    all_costs = []
    all_tardiness = []
    all_solve_times = []
    all_solutions = []
    all_pareto_fronts = []
    
    for run_idx in range(n_runs):
        print(f"  Run {run_idx+1}/{n_runs}...", end='')
        
        start_time = time.time()
        nsga2 = NSGA2VRP(model, pop_size=30, max_gen=120)
        solutions, pareto_front = nsga2.solve()
        solve_time = time.time() - start_time
        
        costs = []
        tardiness_penalties = []
        pf_run = []
        for sol in solutions:
            cost, _ = model.evaluate_solution(sol)
            tardiness = calculate_tardiness_penalty(model, sol)
            costs.append(cost)
            tardiness_penalties.append(tardiness)
            pf_run.append((cost, tardiness))
        
        all_costs.extend(costs)
        all_tardiness.extend(tardiness_penalties)
        all_solve_times.append(solve_time)
        all_solutions.extend(solutions)
        all_pareto_fronts.append(pf_run)
        
        print(f" done")
    
    mean_cost = np.mean(all_costs)
    std_cost = np.std(all_costs)
    mean_tardiness = np.mean(all_tardiness)
    std_tardiness = np.std(all_tardiness)
    mean_solve_time = np.mean(all_solve_times)
    
    print(f"  Total Solutions: {len(all_solutions)}")
    print(f"  Mean Cost: {mean_cost:.2f} ± {std_cost:.2f}")
    print(f"  Mean Tardiness: {mean_tardiness:.2f} ± {std_tardiness:.2f}")
    print(f"  Avg Solve Time: {mean_solve_time:.2f}s")
    
    return {
        'n_runs': n_runs,
        'mean_cost': mean_cost,
        'std_cost': std_cost,
        'mean_tardiness': mean_tardiness,
        'std_tardiness': std_tardiness,
        'mean_solve_time': mean_solve_time,
        'all_costs': all_costs,
        'all_tardiness': all_tardiness,
        'all_solutions': all_solutions,
        'all_pareto_fronts': all_pareto_fronts
    }


def validate_routes(model: VRPTruckDroneModel, routes: List[Route], algorithm_name: str = ""):
    """Validate vehicle IDs and drone usage."""
    actual_trucks = set()
    actual_drones = set()
    drone_mission_count = 0
    drone_customer_count = 0
    
    expected_trucks = model.get_number_of_trucks()
    expected_drones = model.get_number_of_drones()
    
    for route in routes:
        if route.vehicle_type == 'truck':
            actual_trucks.add(route.vehicle_id)
            for mission in route.drone_missions:
                actual_drones.add(mission.drone_id)
                drone_mission_count += len(mission.customer_ids)
        elif route.vehicle_type == 'drone':
            actual_drones.add(route.vehicle_id)
            drone_customer_count += len(route.customers)
    
    truck_ids_ok = actual_trucks == set(range(expected_trucks))
    expected_drone_ids = set(range(expected_trucks, expected_trucks + expected_drones))
    drone_ids_ok = actual_drones == expected_drone_ids
    
    if not truck_ids_ok:
        print(f"  WARNING [{algorithm_name}]: Expected truck IDs 0-{expected_trucks-1}, got {sorted(actual_trucks)}")
    if not drone_ids_ok and actual_drones:
        print(f"  WARNING [{algorithm_name}]: Expected drone IDs {expected_trucks}-{expected_trucks+expected_drones-1}, got {sorted(actual_drones)}")
    
    print(f"  [{algorithm_name}] Used: {len(actual_trucks)}/{expected_trucks} trucks, {len(actual_drones)}/{expected_drones} drones")
    if drone_mission_count > 0:
        print(f"  [{algorithm_name}] Drone missions: {len(actual_drones)} drones serving {drone_mission_count} customers")
    if drone_customer_count > 0:
        print(f"  [{algorithm_name}] Drone direct: {len(actual_drones)} drones serving {drone_customer_count} customers")
    
    total_drone_served = drone_mission_count + drone_customer_count
    if total_drone_served == 0 and expected_drones > 0:
        print(f"  WARNING [{algorithm_name}]: No customers served by drones!")
    
    return truck_ids_ok and (len(actual_drones) == expected_drones or total_drone_served > 0)


def _compute_pareto_front(costs: np.ndarray, tardiness: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Filter to non-dominated points (minimize both cost and tardiness)."""
    if len(costs) == 0:
        return np.array([]), np.array([])
    
    points = np.column_stack([costs, tardiness])
    is_dominated = np.zeros(len(points), dtype=bool)
    
    for i in range(len(points)):
        if is_dominated[i]:
            continue
        for j in range(len(points)):
            if i == j or is_dominated[j]:
                continue
            if points[i, 0] <= points[j, 0] and points[i, 1] <= points[j, 1]:
                if points[i, 0] < points[j, 0] or points[i, 1] < points[j, 1]:
                    is_dominated[j] = True
    
    pf = points[~is_dominated]
    sort_idx = np.argsort(pf[:, 0])
    return pf[sort_idx, 0], pf[sort_idx, 1]


def plot_pareto_front(paco_costs: List, nsga2_costs: List, no_drone_costs: List,
                      paco_tardiness: List, nsga2_tardiness: List, no_drone_tardiness: List,
                      save_path: str):
    """
    Plot true Pareto front (non-dominated points only) with raw objective values.
    X-axis: Cost of Travel
    Y-axis: Penalty due to tardiness
    """
    plt.figure(figsize=(10, 8))
    
    all_costs = []
    all_tardiness = []
    
    if nsga2_costs and nsga2_tardiness:
        pf_costs, pf_tardiness = _compute_pareto_front(np.array(nsga2_costs), np.array(nsga2_tardiness))
        all_costs.extend(pf_costs)
        all_tardiness.extend(pf_tardiness)
        plt.scatter(pf_costs, pf_tardiness, facecolors='none', edgecolors='#1f77b4',
                    marker='o', s=60, label='NSGA II')
        plt.plot(pf_costs, pf_tardiness, c='#1f77b4', linestyle='-',
                 linewidth=1.5, alpha=0.7)
    
    if paco_costs and paco_tardiness:
        pf_costs, pf_tardiness = _compute_pareto_front(np.array(paco_costs), np.array(paco_tardiness))
        all_costs.extend(pf_costs)
        all_tardiness.extend(pf_tardiness)
        plt.scatter(pf_costs, pf_tardiness, facecolors='none', edgecolors='#2ca02c',
                    marker='o', s=60, label='Collaborative P-ACO')
        plt.plot(pf_costs, pf_tardiness, c='#2ca02c', linestyle='-',
                 linewidth=1.5, alpha=0.7)
    
    if no_drone_costs and no_drone_tardiness:
        pf_costs, pf_tardiness = _compute_pareto_front(np.array(no_drone_costs), np.array(no_drone_tardiness))
        all_costs.extend(pf_costs)
        all_tardiness.extend(pf_tardiness)
        plt.scatter(pf_costs, pf_tardiness, facecolors='none', edgecolors='#ff7f0e',
                    marker='s', s=50, label='No-Drone')
        plt.plot(pf_costs, pf_tardiness, c='#ff7f0e', linestyle='--',
                 linewidth=2, alpha=0.7)
    
    plt.xlabel('Cost of Travel', fontsize=12)
    plt.ylabel('Penalty due to tardiness', fontsize=12)
    
    if all_costs:
        x_min, x_max = min(all_costs), max(all_costs)
        x_margin = (x_max - x_min) * 0.1 if x_max != x_min else 5
        plt.xlim(x_min - x_margin, x_max + x_margin)
    
    plt.title('Non dominated front', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  Pareto Plot: {save_path}")


def plot_routes(model: VRPTruckDroneModel, routes: List[Route], title: str, save_path: str):
    """Plot vehicle routes."""
    visualizer = Visualizer(model)
    visualizer.plot_routes(routes, title=title, save_path=save_path, show_all_nodes=True)


def run_single_comparison(n_customers: int, rc_type: str, instance_id: int,
                          results_dir: str, n_vehicles: int = 2,
                          endurance_type: str = 'medium', n_runs: int = 10) -> Dict:
    """Run single comparison experiment with multiple repetitions."""
    print(f"\n{'='*60}")
    print(f"Experiment: {n_customers} Customers, {rc_type}-{instance_id:02d}")
    print(f"{'='*60}")
    
    model_with_drones = load_solomon_instance(n_customers, rc_type, instance_id, 
                                              n_vehicles, endurance_type, use_drones=True)
    model_no_drone = load_solomon_instance(n_customers, rc_type, instance_id, 
                                           n_vehicles, endurance_type, use_drones=False)
    
    n_trucks = model_with_drones.get_number_of_trucks()
    n_drones = model_with_drones.get_number_of_drones()
    
    print(f"  Customers: {model_with_drones.get_number_of_customers()}")
    print(f"  Trucks: {n_trucks}")
    print(f"  Drones: {n_drones}")
    print(f"  Drone Endurance: {endurance_type} ({model_with_drones.drone_range} km)")
    print(f"  Repetitions: {n_runs}")
    
    paco_n_ants = 50 if n_customers == 25 else 80
    paco_max_iter = 100 if n_customers == 25 else 100
    nsga2_pop_size = 50 if n_customers == 25 else 80
    nsga2_max_gen = 120
    
    paco_result = run_paco_experiment(model_with_drones, n_ants=paco_n_ants, 
                                      max_iter=paco_max_iter, n_runs=n_runs)
    nsga2_result = run_nsga2_experiment(model_with_drones, pop_size=nsga2_pop_size, 
                                        max_gen=nsga2_max_gen, n_runs=n_runs)
    no_drone_result = run_no_drone_experiment(model_no_drone, n_runs=n_runs)

    # Recalculate HV using shared reference point for fair comparison.
    ref_cost, ref_tardiness = _compute_shared_reference_point(paco_result, nsga2_result, no_drone_result)
    paco_result['mean_hv'] = np.mean([calculate_hypervolume(pf, ref_cost, ref_tardiness)
                                       for pf in paco_result.get('all_pareto_fronts', [])])
    paco_result['std_hv'] = np.std([calculate_hypervolume(pf, ref_cost, ref_tardiness)
                                     for pf in paco_result.get('all_pareto_fronts', [])])
    nsga2_result['mean_hv'] = np.mean([calculate_hypervolume(pf, ref_cost, ref_tardiness)
                                        for pf in nsga2_result.get('all_pareto_fronts', [])])
    nsga2_result['std_hv'] = np.std([calculate_hypervolume(pf, ref_cost, ref_tardiness)
                                      for pf in nsga2_result.get('all_pareto_fronts', [])])
    no_drone_result['mean_hv'] = np.mean([calculate_hypervolume(pf, ref_cost, ref_tardiness)
                                           for pf in no_drone_result.get('all_pareto_fronts', [])])
    no_drone_result['std_hv'] = np.std([calculate_hypervolume(pf, ref_cost, ref_tardiness)
                                         for pf in no_drone_result.get('all_pareto_fronts', [])])

    print(f"\n  Shared HV ref: cost={ref_cost:.1f}, tardiness={ref_tardiness:.1f}")
    print(f"  P-ACO   HV: {paco_result['mean_hv']:.2f} ± {paco_result['std_hv']:.2f}")
    print(f"  NSGA-II HV: {nsga2_result['mean_hv']:.2f} ± {nsga2_result['std_hv']:.2f}")
    print(f"  No-Drone HV: {no_drone_result['mean_hv']:.2f} ± {no_drone_result['std_hv']:.2f}")
    
    if paco_result['all_solutions']:
        validate_routes(model_with_drones, paco_result['all_solutions'][0], 'P-ACO')
    
    if nsga2_result['all_solutions']:
        validate_routes(model_with_drones, nsga2_result['all_solutions'][0], 'NSGA-II')
    
    pareto_path = os.path.join(results_dir, f'pareto_{n_customers}c_{rc_type}{instance_id:02d}_{n_vehicles}V_{endurance_type}.png')
    plot_pareto_front(
        paco_result.get('all_costs', []), nsga2_result.get('all_costs', []), no_drone_result.get('all_costs', []),
        paco_result.get('all_tardiness', []), nsga2_result.get('all_tardiness', []), no_drone_result.get('all_tardiness', []),
        pareto_path
    )
    
    if paco_result['all_solutions']:
        best_paco_idx = np.argmin([f[0] for f in paco_result['all_pareto_fronts']])
        best_paco_route = paco_result['all_solutions'][best_paco_idx]
        paco_route_path = os.path.join(results_dir, f'paco_{n_customers}c_{rc_type}{instance_id:02d}_{n_vehicles}V_{endurance_type}.png')
        title = f'P-ACO | {n_customers}C | {n_trucks}T+{n_drones}D | {endurance_type}'
        plot_routes(model_with_drones, best_paco_route, title, paco_route_path)
    
    if nsga2_result['all_solutions']:
        best_nsga2_idx = np.argmin([f[0] for f in nsga2_result['all_pareto_fronts']])
        best_nsga2_route = nsga2_result['all_solutions'][best_nsga2_idx]
        nsga2_route_path = os.path.join(results_dir, f'nsga2_{n_customers}c_{rc_type}{instance_id:02d}_{n_vehicles}V_{endurance_type}.png')
        title = f'NSGA-II | {n_customers}C | {n_trucks}T+{n_drones}D | {endurance_type}'
        plot_routes(model_with_drones, best_nsga2_route, title, nsga2_route_path)
    
    return {
        'n_customers': n_customers,
        'rc_type': rc_type,
        'instance_id': instance_id,
        'n_trucks': n_trucks,
        'n_drones': n_drones,
        'endurance_type': endurance_type,
        'n_runs': n_runs,
        'paco': paco_result,
        'nsga2': nsga2_result,
        'no_drone': no_drone_result
    }


def main():
    """Run comparison experiments using Solomon RC benchmark.
    
    Solomon RC series:
    - RC1: Short scheduling horizon (tight time windows)
    - RC2: Long scheduling horizon (wide time windows)
    - Customer scales: 25, 50
    - Drone endurance: medium (4km), high (6km)
    - Multi-run: 10 repetitions per experiment
    """
    results_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(results_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    n_runs_per_exp = 10
    
    experiments = [
        {'n_customers': 25, 'rc_type': 'RC1', 'instance_id': 1, 'n_vehicles': 2, 'endurance_type': 'medium'},
        {'n_customers': 25, 'rc_type': 'RC1', 'instance_id': 1, 'n_vehicles': 2, 'endurance_type': 'high'},
        {'n_customers': 25, 'rc_type': 'RC2', 'instance_id': 1, 'n_vehicles': 2, 'endurance_type': 'medium'},
        {'n_customers': 25, 'rc_type': 'RC2', 'instance_id': 1, 'n_vehicles': 2, 'endurance_type': 'high'},
        {'n_customers': 50, 'rc_type': 'RC1', 'instance_id': 1, 'n_vehicles': 4, 'endurance_type': 'medium'},
        {'n_customers': 50, 'rc_type': 'RC1', 'instance_id': 1, 'n_vehicles': 4, 'endurance_type': 'high'},
        {'n_customers': 50, 'rc_type': 'RC2', 'instance_id': 1, 'n_vehicles': 4, 'endurance_type': 'medium'},
        {'n_customers': 50, 'rc_type': 'RC2', 'instance_id': 1, 'n_vehicles': 4, 'endurance_type': 'high'},
        {'n_customers': 50, 'rc_type': 'RC1', 'instance_id': 1, 'n_vehicles': 6, 'endurance_type': 'medium'},
        {'n_customers': 50, 'rc_type': 'RC1', 'instance_id': 1, 'n_vehicles': 6, 'endurance_type': 'high'},
        {'n_customers': 50, 'rc_type': 'RC2', 'instance_id': 1, 'n_vehicles': 6, 'endurance_type': 'medium'},
        {'n_customers': 50, 'rc_type': 'RC2', 'instance_id': 1, 'n_vehicles': 6, 'endurance_type': 'high'},
    ]
    
    all_results = []
    
    for exp in experiments:
        result = run_single_comparison(
            exp['n_customers'], exp['rc_type'], exp['instance_id'],
            results_dir, exp['n_vehicles'], exp['endurance_type'], n_runs_per_exp
        )
        all_results.append(result)
    
    with open(os.path.join(results_dir, 'solomon_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    summary = "# P-ACO vs NSGA-II Results (Solomon RC Benchmark)\n\n"
    summary += "## Experimental Setup\n"
    summary += "- RC1: Short scheduling horizon (tight time windows)\n"
    summary += "- RC2: Long scheduling horizon (wide time windows)\n"
    summary += "- Linear scaling: Solomon [0,100] -> Urban [0,12] km\n"
    summary += "- Drone endurance: medium (4km), high (6km)\n"
    summary += f"- Repetitions per experiment: {n_runs_per_exp}\n"
    summary += "- Metrics: Mean Cost ± Std (averaged across all Pareto solutions from all runs)\n\n"
    
    summary += "## Results\n\n"
    summary += "| Config | Customers | RC Type | Vehicles | Endurance | Method | Mean Cost ± Std | Mean Tardiness ± Std |\n"
    summary += "|--------|-----------|---------|----------|-----------|--------|-----------------|----------------------|\n"
    
    for r in all_results:
        exp_key = f"{r['n_customers']}c_{r['rc_type']}{r['instance_id']:02d}_{r['endurance_type']}"
        summary += f"| {exp_key} | {r['n_customers']} | {r['rc_type']} | {r['n_trucks']}T+{r['n_drones']}D | {r['endurance_type']} | "
        summary += f"P-ACO | {r['paco']['mean_cost']:.2f} ± {r['paco']['std_cost']:.2f} | "
        summary += f"{r['paco']['mean_tardiness']:.2f} ± {r['paco']['std_tardiness']:.2f} |\n"
        
        summary += f"| {exp_key} | {r['n_customers']} | {r['rc_type']} | {r['n_trucks']}T+{r['n_drones']}D | {r['endurance_type']} | "
        summary += f"NSGA-II | {r['nsga2']['mean_cost']:.2f} ± {r['nsga2']['std_cost']:.2f} | "
        summary += f"{r['nsga2']['mean_tardiness']:.2f} ± {r['nsga2']['std_tardiness']:.2f} |\n"
        
        summary += f"| {exp_key} | {r['n_customers']} | {r['rc_type']} | {r['n_trucks']}T+0D | - | "
        summary += f"No-Drone | {r['no_drone']['mean_cost']:.2f} ± {r['no_drone']['std_cost']:.2f} | "
        summary += f"{r['no_drone']['mean_tardiness']:.2f} ± {r['no_drone']['std_tardiness']:.2f} |\n"
    
    with open(os.path.join(results_dir, 'solomon_summary.md'), 'w') as f:
        f.write(summary)
    
    print(f"\n{'='*60}")
    print("Experiment Complete!")
    print(f"{'='*60}")
    print(f"Results: {os.path.join(results_dir, 'solomon_results.json')}")
    print(f"Summary: {os.path.join(results_dir, 'solomon_summary.md')}")
    print(f"Pareto plots: {results_dir}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=str, default='all', choices=['small', 'large', 'all'])
    parser.add_argument('--runs', type=int, default=10, help='Number of repetitions per experiment')
    args = parser.parse_args()
    
    if args.size == 'small':
        print("Running small experiment (25 customers, RC1-01, medium)...")
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        model_with_drones = load_solomon_instance(25, 'RC1', 1, 2, 'medium', use_drones=True)
        model_no_drone = load_solomon_instance(25, 'RC1', 1, 2, 'medium', use_drones=False)
        
        paco_result = run_paco_experiment(model_with_drones, n_ants=50, max_iter=100, n_runs=args.runs)
        nsga2_result = run_nsga2_experiment(model_with_drones, pop_size=50, max_gen=120, n_runs=args.runs)
        no_drone_result = run_no_drone_experiment(model_no_drone, n_runs=args.runs)
        
        pareto_path = os.path.join(results_dir, 'pareto_25c_RC101_medium.png')
        plot_pareto_front(
            paco_result.get('all_costs', []), nsga2_result.get('all_costs', []), no_drone_result.get('all_costs', []),
            paco_result.get('all_tardiness', []), nsga2_result.get('all_tardiness', []), no_drone_result.get('all_tardiness', []),
            pareto_path
        )
        
        result = {
            'n_customers': 25, 'rc_type': 'RC1', 'instance_id': 1,
            'n_trucks': 2, 'n_drones': 2, 'endurance_type': 'medium',
            'n_runs': args.runs,
            'paco': paco_result, 'nsga2': nsga2_result, 'no_drone': no_drone_result
        }
        
        with open(os.path.join(results_dir, 'solomon_results_small.json'), 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        print(f"\nResults saved to {results_dir}")
        
    elif args.size == 'large':
        print("Running large experiment (50 customers, RC1-01, medium)...")
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        model_with_drones = load_solomon_instance(50, 'RC1', 1, 4, 'medium', use_drones=True)
        model_no_drone = load_solomon_instance(50, 'RC1', 1, 4, 'medium', use_drones=False)
        
        paco_result = run_paco_experiment(model_with_drones, n_ants=70, max_iter=170, n_runs=args.runs)
        nsga2_result = run_nsga2_experiment(model_with_drones, pop_size=200, max_gen=120, n_runs=args.runs)
        no_drone_result = run_no_drone_experiment(model_no_drone, n_runs=args.runs)
        
        pareto_path = os.path.join(results_dir, 'pareto_50c_RC101_medium.png')
        plot_pareto_front(
            paco_result.get('all_costs', []), nsga2_result.get('all_costs', []), no_drone_result.get('all_costs', []),
            paco_result.get('all_tardiness', []), nsga2_result.get('all_tardiness', []), no_drone_result.get('all_tardiness', []),
            pareto_path
        )
        
        result = {
            'n_customers': 50, 'rc_type': 'RC1', 'instance_id': 1,
            'n_trucks': 4, 'n_drones': 4, 'endurance_type': 'medium',
            'n_runs': args.runs,
            'paco': paco_result, 'nsga2': nsga2_result, 'no_drone': no_drone_result
        }
        
        with open(os.path.join(results_dir, 'solomon_results_large.json'), 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        print(f"\nResults saved to {results_dir}")
        
    else:
        main()