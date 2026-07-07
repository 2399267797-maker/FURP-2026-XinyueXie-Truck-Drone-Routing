"""
PACO-imp vs PACO+ALNS Comparison using Solomon RC Benchmark Dataset.

Compares the improved P-ACO (paco_imp.py) against the hybrid PACO+ALNS
on Solomon RC benchmark instances with 25 and 50 customers.
Metrics: Cost, Tardiness, Hypervolume (HV), Drone Utilization, Solve Time.
"""
import os
import sys
import json
import time
import importlib.util
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

# ---------- path setup ----------
ALNS_DIR = os.path.dirname(os.path.abspath(__file__))          # e.g. .../PACO+ALNS/
EXPERIMENTS_DIR = os.path.dirname(ALNS_DIR)                    # e.g. .../experiments/
PACO_DIR = os.path.join(EXPERIMENTS_DIR, 'PACO')               # e.g. .../experiments/PACO/
VS_DIR = os.path.join(EXPERIMENTS_DIR, 'PACO_vs_NSGA2')        # e.g. .../experiments/PACO_vs_NSGA2/

for p in [PACO_DIR, VS_DIR, ALNS_DIR, EXPERIMENTS_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ---------- shared imports ----------
from models.vrp_model import VRPTruckDroneModel, Route, DroneMission
from utils.evaluator import Evaluator
from utils.visualizer import Visualizer
from data.solomon_loader import SolomonLoader

# ---------- load PACO+ALNS by importlib (filename contains '+') ----------
paco_alns_spec = importlib.util.spec_from_file_location(
    "paco_alns", os.path.join(ALNS_DIR, "PACO+ALNS.py")
)
paco_alns_module = importlib.util.module_from_spec(paco_alns_spec)
paco_alns_spec.loader.exec_module(paco_alns_module)
CollaborativePACOALNS = paco_alns_module.CollaborativePACOALNS

# ---------- load PACO-imp ----------
from algorithms.paco_imp import CollaborativePACO as CollaborativePACOImp


# ============================================================
#  Helper functions
# ============================================================

def load_solomon_instance(n_customers: int, rc_type: str = 'RC1', instance_id: int = 1,
                          n_vehicles: int = 2, endurance_type: str = 'medium',
                          use_drones: bool = True) -> VRPTruckDroneModel:
    loader = SolomonLoader()
    if rc_type == 'RC1':
        return loader.load_rc1_instance(n_customers, instance_id, n_vehicles, endurance_type, use_drones)
    else:
        return loader.load_rc2_instance(n_customers, instance_id, n_vehicles, endurance_type, use_drones)


def calculate_tardiness_penalty(model: VRPTruckDroneModel, routes: List[Route]) -> float:
    evaluator = Evaluator(model)
    return evaluator.calculate_pure_tardiness(routes)


def calculate_hypervolume(pareto_front, ref_cost=None, ref_tardiness=None):
    """Calculate hypervolume indicator for 2D Pareto front (min-cost, min-tardiness)."""
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
    """Compute a shared HV reference point across all algorithms in one experiment."""
    all_costs = []
    all_tardiness = []
    for result in results:
        if result is None:
            continue
        pf = result.get('all_pareto_fronts', [])
        if not pf:
            continue
        for run_pf in pf:
            if run_pf:
                pts = np.array(run_pf)
                all_costs.extend(pts[:, 0].tolist())
                all_tardiness.extend(pts[:, 1].tolist())
    if not all_costs:
        return 200.0, 200.0
    return np.max(all_costs) * 1.1, np.max(all_tardiness) * 1.1


def validate_routes(model: VRPTruckDroneModel, routes: List[Route], algorithm_name: str = ""):
    """Validate vehicle IDs and drone usage."""
    actual_trucks = set()
    actual_drones = set()
    drone_mission_count = 0
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
    total_drone_served = drone_mission_count
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


def count_drone_utilization(solutions: List[List[Route]]) -> Dict:
    """Count how many solutions use drones and how many drone missions exist."""
    total_sols = len(solutions)
    sols_with_drones = 0
    total_drone_missions = 0
    total_drone_customers = 0
    for sol in solutions:
        has_drone = False
        for route in sol:
            if route.vehicle_type == 'truck' and route.drone_missions:
                has_drone = True
                for m in route.drone_missions:
                    total_drone_missions += 1
                    total_drone_customers += len(m.customer_ids)
        if has_drone:
            sols_with_drones += 1
    drone_ratio = sols_with_drones / total_sols if total_sols > 0 else 0.0
    return {
        'total_solutions': total_sols,
        'solutions_with_drones': sols_with_drones,
        'drone_ratio': drone_ratio,
        'total_drone_missions': total_drone_missions,
        'total_drone_customers': total_drone_customers
    }


# ============================================================
#  Single-run & experiment wrappers
# ============================================================

def run_paco_imp_single(model: VRPTruckDroneModel, n_ants: int = 50, max_iter: int = 100) -> Dict:
    """Run single PACO-imp trial."""
    start_time = time.time()
    paco = CollaborativePACOImp(model, n_ants=n_ants, max_iter=max_iter)
    solutions, pareto_front = paco.solve()
    solve_time = time.time() - start_time
    costs, tardiness_penalties = [], []
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


def run_paco_alns_single(model: VRPTruckDroneModel, n_ants: int = 30, max_iter: int = 100, alns_iter: int = 15) -> Dict:
    """Run single PACO+ALNS trial."""
    start_time = time.time()
    paco_alns = CollaborativePACOALNS(model, n_ants=n_ants, max_iter=max_iter, alns_iter=alns_iter)
    solutions, pareto_front = paco_alns.solve()
    solve_time = time.time() - start_time
    costs, tardiness_penalties = [], []
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


def run_paco_imp_experiment(model: VRPTruckDroneModel, n_ants: int = 50, max_iter: int = 100,
                            n_runs: int = 10) -> Dict:
    """Run PACO-imp experiment with multiple repetitions."""
    print("\n--- PACO-imp ---")
    all_costs, all_tardiness, all_solve_times = [], [], []
    all_hypervolumes, all_solutions, all_pareto_fronts = [], [], []
    for run_idx in range(n_runs):
        print(f"  Run {run_idx+1}/{n_runs}...", end='')
        result = run_paco_imp_single(model, n_ants, max_iter)
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
    drone_stats = count_drone_utilization(all_solutions)
    print(f"  Total Solutions: {len(all_solutions)}")
    print(f"  Mean Cost: {mean_cost:.2f} ± {std_cost:.2f}")
    print(f"  Mean Tardiness: {mean_tardiness:.2f} ± {std_tardiness:.2f}")
    print(f"  Mean Hypervolume: {mean_hv:.2f} ± {std_hv:.2f}")
    print(f"  Median Hypervolume: {median_hv:.2f}")
    print(f"  Avg Solve Time: {mean_solve_time:.2f}s")
    print(f"  Drone Utilization: {drone_stats['drone_ratio']*100:.1f}% of solutions ({drone_stats['total_drone_missions']} missions)")
    return {
        'n_runs': n_runs,
        'mean_cost': mean_cost, 'std_cost': std_cost,
        'mean_tardiness': mean_tardiness, 'std_tardiness': std_tardiness,
        'mean_solve_time': mean_solve_time,
        'mean_hv': mean_hv, 'std_hv': std_hv, 'median_hv': median_hv,
        'all_hypervolumes': all_hypervolumes,
        'all_costs': all_costs, 'all_tardiness': all_tardiness,
        'all_solutions': all_solutions, 'all_pareto_fronts': all_pareto_fronts,
        'drone_stats': drone_stats
    }


def run_paco_alns_experiment(model: VRPTruckDroneModel, n_ants: int = 30, max_iter: int = 100,
                             alns_iter: int = 15, n_runs: int = 10) -> Dict:
    """Run PACO+ALNS experiment with multiple repetitions."""
    print("\n--- PACO+ALNS ---")
    all_costs, all_tardiness, all_solve_times = [], [], []
    all_hypervolumes, all_solutions, all_pareto_fronts = [], [], []
    for run_idx in range(n_runs):
        print(f"  Run {run_idx+1}/{n_runs}...", end='')
        result = run_paco_alns_single(model, n_ants, max_iter, alns_iter)
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
    drone_stats = count_drone_utilization(all_solutions)
    print(f"  Total Solutions: {len(all_solutions)}")
    print(f"  Mean Cost: {mean_cost:.2f} ± {std_cost:.2f}")
    print(f"  Mean Tardiness: {mean_tardiness:.2f} ± {std_tardiness:.2f}")
    print(f"  Mean Hypervolume: {mean_hv:.2f} ± {std_hv:.2f}")
    print(f"  Median Hypervolume: {median_hv:.2f}")
    print(f"  Avg Solve Time: {mean_solve_time:.2f}s")
    print(f"  Drone Utilization: {drone_stats['drone_ratio']*100:.1f}% of solutions ({drone_stats['total_drone_missions']} missions)")
    return {
        'n_runs': n_runs,
        'mean_cost': mean_cost, 'std_cost': std_cost,
        'mean_tardiness': mean_tardiness, 'std_tardiness': std_tardiness,
        'mean_solve_time': mean_solve_time,
        'mean_hv': mean_hv, 'std_hv': std_hv, 'median_hv': median_hv,
        'all_hypervolumes': all_hypervolumes,
        'all_costs': all_costs, 'all_tardiness': all_tardiness,
        'all_solutions': all_solutions, 'all_pareto_fronts': all_pareto_fronts,
        'drone_stats': drone_stats
    }


# ============================================================
#  Plotting
# ============================================================

def plot_pareto_front(paco_imp_costs: List, alns_costs: List,
                      paco_imp_tardiness: List, alns_tardiness: List,
                      save_path: str):
    """Plot true Pareto front (non-dominated points only)."""
    plt.figure(figsize=(10, 8))
    all_costs, all_tardiness = [], []
    if paco_imp_costs and paco_imp_tardiness:
        pf_costs, pf_tardiness = _compute_pareto_front(np.array(paco_imp_costs), np.array(paco_imp_tardiness))
        all_costs.extend(pf_costs)
        all_tardiness.extend(pf_tardiness)
        plt.scatter(pf_costs, pf_tardiness, facecolors='none', edgecolors='#2ca02c',
                    marker='o', s=60, label='P-ACO-imp')
        plt.plot(pf_costs, pf_tardiness, c='#2ca02c', linestyle='-', linewidth=1.5, alpha=0.7)
    if alns_costs and alns_tardiness:
        pf_costs, pf_tardiness = _compute_pareto_front(np.array(alns_costs), np.array(alns_tardiness))
        all_costs.extend(pf_costs)
        all_tardiness.extend(pf_tardiness)
        plt.scatter(pf_costs, pf_tardiness, facecolors='none', edgecolors='#d62728',
                    marker='^', s=60, label='P-ACO+ALNS')
        plt.plot(pf_costs, pf_tardiness, c='#d62728', linestyle='-', linewidth=1.5, alpha=0.7)
    plt.xlabel('Cost of Travel', fontsize=12)
    plt.ylabel('Penalty due to tardiness', fontsize=12)
    if all_costs:
        x_min, x_max = min(all_costs), max(all_costs)
        x_margin = (x_max - x_min) * 0.1 if x_max != x_min else 5
        plt.xlim(x_min - x_margin, x_max + x_margin)
    plt.title('PACO-imp vs PACO+ALNS — Non-dominated Front', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  Pareto Plot: {save_path}")


def plot_routes(model: VRPTruckDroneModel, routes: List[Route], title: str, save_path: str):
    visualizer = Visualizer(model)
    visualizer.plot_routes(routes, title=title, save_path=save_path, show_all_nodes=True)


# ============================================================
#  Single experiment comparator
# ============================================================

def run_single_comparison(n_customers: int, rc_type: str, instance_id: int,
                          results_dir: str, n_vehicles: int = 2,
                          endurance_type: str = 'medium', n_runs: int = 10) -> Dict:
    """Run single comparison experiment with multiple repetitions."""
    print(f"\n{'='*60}")
    print(f"Experiment: {n_customers} Customers, {rc_type}-{instance_id:02d}, {n_vehicles}V, {endurance_type}")
    print(f"{'='*60}")

    model = load_solomon_instance(n_customers, rc_type, instance_id,
                                  n_vehicles, endurance_type, use_drones=True)
    n_trucks = model.get_number_of_trucks()
    n_drones = model.get_number_of_drones()
    print(f"  Customers: {model.get_number_of_customers()}")
    print(f"  Trucks: {n_trucks}, Drones: {n_drones}")
    print(f"  Drone Endurance: {endurance_type} ({model.drone_range} km)")
    print(f"  Repetitions: {n_runs}")

    # PACO-imp: larger ant population for better comparison
    paco_imp_ants = 50 if n_customers == 25 else 80
    paco_imp_iter = 100

    # PACO+ALNS: fewer ants since ALNS provides additional refinement
    alns_ants = 30 if n_customers == 25 else 50
    alns_iter = 100
    alns_local = 15

    paco_imp_result = run_paco_imp_experiment(model, n_ants=paco_imp_ants,
                                              max_iter=paco_imp_iter, n_runs=n_runs)
    alns_result = run_paco_alns_experiment(model, n_ants=alns_ants,
                                           max_iter=alns_iter, alns_iter=alns_local, n_runs=n_runs)

    # Recalculate HV using shared reference point
    ref_cost, ref_tardiness = _compute_shared_reference_point(paco_imp_result, alns_result)
    paco_imp_result['mean_hv'] = np.mean([calculate_hypervolume(pf, ref_cost, ref_tardiness)
                                          for pf in paco_imp_result.get('all_pareto_fronts', [])])
    paco_imp_result['std_hv'] = np.std([calculate_hypervolume(pf, ref_cost, ref_tardiness)
                                        for pf in paco_imp_result.get('all_pareto_fronts', [])])
    alns_result['mean_hv'] = np.mean([calculate_hypervolume(pf, ref_cost, ref_tardiness)
                                      for pf in alns_result.get('all_pareto_fronts', [])])
    alns_result['std_hv'] = np.std([calculate_hypervolume(pf, ref_cost, ref_tardiness)
                                    for pf in alns_result.get('all_pareto_fronts', [])])

    print(f"\n  Shared HV ref: cost={ref_cost:.1f}, tardiness={ref_tardiness:.1f}")
    print(f"  PACO-imp   HV: {paco_imp_result['mean_hv']:.2f} ± {paco_imp_result['std_hv']:.2f}")
    print(f"  PACO+ALNS  HV: {alns_result['mean_hv']:.2f} ± {alns_result['std_hv']:.2f}")

    if paco_imp_result['all_solutions']:
        validate_routes(model, paco_imp_result['all_solutions'][0], 'PACO-imp')
    if alns_result['all_solutions']:
        validate_routes(model, alns_result['all_solutions'][0], 'PACO+ALNS')

    # Pareto plot
    n_veh_tag = n_vehicles
    pareto_path = os.path.join(results_dir, f'pareto_{n_customers}c_{rc_type}{instance_id:02d}_{n_veh_tag}V_{endurance_type}.png')
    plot_pareto_front(
        paco_imp_result.get('all_costs', []), alns_result.get('all_costs', []),
        paco_imp_result.get('all_tardiness', []), alns_result.get('all_tardiness', []),
        pareto_path
    )

    # Route plots
    if paco_imp_result['all_solutions']:
        best_idx = np.argmin([f[0] for f in paco_imp_result['all_pareto_fronts']])
        best_route = paco_imp_result['all_solutions'][best_idx]
        route_path = os.path.join(results_dir, f'paco_imp_{n_customers}c_{rc_type}{instance_id:02d}_{n_veh_tag}V_{endurance_type}.png')
        title = f'PACO-imp | {n_customers}C | {n_trucks}T+{n_drones}D | {endurance_type}'
        plot_routes(model, best_route, title, route_path)

    if alns_result['all_solutions']:
        best_idx = np.argmin([f[0] for f in alns_result['all_pareto_fronts']])
        best_route = alns_result['all_solutions'][best_idx]
        route_path = os.path.join(results_dir, f'paco_alns_{n_customers}c_{rc_type}{instance_id:02d}_{n_veh_tag}V_{endurance_type}.png')
        title = f'PACO+ALNS | {n_customers}C | {n_trucks}T+{n_drones}D | {endurance_type}'
        plot_routes(model, best_route, title, route_path)

    return {
        'n_customers': n_customers, 'rc_type': rc_type, 'instance_id': instance_id,
        'n_trucks': n_trucks, 'n_drones': n_drones, 'endurance_type': endurance_type, 'n_runs': n_runs,
        'paco_imp': paco_imp_result, 'paco_alns': alns_result
    }


# ============================================================
#  Main
# ============================================================

def main():
    results_dir = os.path.join(ALNS_DIR, 'results')
    os.makedirs(results_dir, exist_ok=True)
    n_runs_per_exp = 10

    experiments = [
        # {'n_customers': 25, 'rc_type': 'RC1', 'instance_id': 1, 'n_vehicles': 2, 'endurance_type': 'medium'},
        # {'n_customers': 25, 'rc_type': 'RC1', 'instance_id': 1, 'n_vehicles': 2, 'endurance_type': 'high'},
        # {'n_customers': 25, 'rc_type': 'RC2', 'instance_id': 1, 'n_vehicles': 2, 'endurance_type': 'medium'},
        # {'n_customers': 25, 'rc_type': 'RC2', 'instance_id': 1, 'n_vehicles': 2, 'endurance_type': 'high'},
        # {'n_customers': 50, 'rc_type': 'RC1', 'instance_id': 1, 'n_vehicles': 4, 'endurance_type': 'medium'},
        # {'n_customers': 50, 'rc_type': 'RC1', 'instance_id': 1, 'n_vehicles': 4, 'endurance_type': 'high'},
        # {'n_customers': 50, 'rc_type': 'RC2', 'instance_id': 1, 'n_vehicles': 4, 'endurance_type': 'medium'},
        # {'n_customers': 50, 'rc_type': 'RC2', 'instance_id': 1, 'n_vehicles': 4, 'endurance_type': 'high'},
        # {'n_customers': 50, 'rc_type': 'RC1', 'instance_id': 1, 'n_vehicles': 6, 'endurance_type': 'medium'},
        # {'n_customers': 50, 'rc_type': 'RC1', 'instance_id': 1, 'n_vehicles': 6, 'endurance_type': 'high'},
        # {'n_customers': 50, 'rc_type': 'RC2', 'instance_id': 1, 'n_vehicles': 6, 'endurance_type': 'medium'},
        # {'n_customers': 50, 'rc_type': 'RC2', 'instance_id': 1, 'n_vehicles': 6, 'endurance_type': 'high'},
         {'n_customers': 100, 'rc_type': 'RC1', 'instance_id': 1, 'n_vehicles': 10, 'endurance_type': 'medium'},
        {'n_customers': 100, 'rc_type': 'RC1', 'instance_id': 1, 'n_vehicles': 10, 'endurance_type': 'high'},
        {'n_customers': 100, 'rc_type': 'RC2', 'instance_id': 1, 'n_vehicles': 10, 'endurance_type': 'medium'},
        {'n_customers': 100, 'rc_type': 'RC2', 'instance_id': 1, 'n_vehicles': 10, 'endurance_type': 'high'},
    ]

    all_results = []
    for exp in experiments:
        result = run_single_comparison(
            exp['n_customers'], exp['rc_type'], exp['instance_id'],
            results_dir, exp['n_vehicles'], exp['endurance_type'], n_runs_per_exp
        )
        all_results.append(result)

    # Save JSON
    with open(os.path.join(results_dir, 'compare_paco_imp_vs_alns.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Generate markdown summary
    summary = "# PACO-imp vs PACO+ALNS Comparison (Solomon RC Benchmark)\n\n"
    summary += "## Experimental Setup\n"
    summary += "- RC1: Short scheduling horizon (tight time windows)\n"
    summary += "- RC2: Long scheduling horizon (wide time windows)\n"
    summary += "- Linear scaling: Solomon [0,100] -> Urban [0,12] km\n"
    summary += f"- Repetitions per experiment: {n_runs_per_exp}\n\n"
    summary += "## Results\n\n"
    summary += "| Config | Customers | RC Type | Vehicles | Endurance | Method | Mean Cost ± Std | Mean Tardiness ± Std | HV | Drone Ratio |\n"
    summary += "|--------|-----------|---------|----------|-----------|--------|-----------------|----------------------|----|-------------|\n"

    for r in all_results:
        exp_key = f"{r['n_customers']}c_{r['rc_type']}{r['instance_id']:02d}_{r['endurance_type']}"

        pi = r['paco_imp']
        summary += f"| {exp_key} | {r['n_customers']} | {r['rc_type']} | {r['n_trucks']}T+{r['n_drones']}D | {r['endurance_type']} | "
        summary += f"PACO-imp | {pi['mean_cost']:.2f} ± {pi['std_cost']:.2f} | {pi['mean_tardiness']:.2f} ± {pi['std_tardiness']:.2f} | "
        summary += f"{pi['mean_hv']:.1f} | {pi['drone_stats']['drone_ratio']*100:.0f}% |\n"

        al = r['paco_alns']
        summary += f"| {exp_key} | {r['n_customers']} | {r['rc_type']} | {r['n_trucks']}T+{r['n_drones']}D | {r['endurance_type']} | "
        summary += f"PACO+ALNS | {al['mean_cost']:.2f} ± {al['std_cost']:.2f} | {al['mean_tardiness']:.2f} ± {al['std_tardiness']:.2f} | "
        summary += f"{al['mean_hv']:.1f} | {al['drone_stats']['drone_ratio']*100:.0f}% |\n"

    with open(os.path.join(results_dir, 'compare_paco_imp_vs_alns_summary.md'), 'w') as f:
        f.write(summary)

    print(f"\n{'='*60}")
    print("Experiment Complete!")
    print(f"{'='*60}")
    print(f"Results: {os.path.join(results_dir, 'compare_paco_imp_vs_alns.json')}")
    print(f"Summary: {os.path.join(results_dir, 'compare_paco_imp_vs_alns_summary.md')}")
    print(f"Pareto plots: {results_dir}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=str, default='all', choices=['small', 'large', 'all'])
    parser.add_argument('--runs', type=int, default=10, help='Number of repetitions per experiment')
    args = parser.parse_args()

    results_dir = os.path.join(ALNS_DIR, 'results')
    os.makedirs(results_dir, exist_ok=True)

    if args.size == 'small':
        print("Running small experiment (25 customers, RC1-01, medium)...")
        model = load_solomon_instance(25, 'RC1', 1, 2, 'medium', use_drones=True)
        paco_imp_result = run_paco_imp_experiment(model, n_ants=50, max_iter=100, n_runs=args.runs)
        alns_result = run_paco_alns_experiment(model, n_ants=30, max_iter=100, alns_iter=15, n_runs=args.runs)
        ref_cost, ref_tardiness = _compute_shared_reference_point(paco_imp_result, alns_result)
        pi_hv = [calculate_hypervolume(pf, ref_cost, ref_tardiness) for pf in paco_imp_result.get('all_pareto_fronts', [])]
        al_hv = [calculate_hypervolume(pf, ref_cost, ref_tardiness) for pf in alns_result.get('all_pareto_fronts', [])]
        paco_imp_result['mean_hv'] = np.mean(pi_hv)
        paco_imp_result['std_hv'] = np.std(pi_hv)
        alns_result['mean_hv'] = np.mean(al_hv)
        alns_result['std_hv'] = np.std(al_hv)

        pareto_path = os.path.join(results_dir, 'pareto_25c_RC101_medium.png')
        plot_pareto_front(
            paco_imp_result.get('all_costs', []), alns_result.get('all_costs', []),
            paco_imp_result.get('all_tardiness', []), alns_result.get('all_tardiness', []),
            pareto_path
        )

        result = {
            'n_customers': 25, 'rc_type': 'RC1', 'instance_id': 1,
            'n_trucks': 2, 'n_drones': 2, 'endurance_type': 'medium', 'n_runs': args.runs,
            'paco_imp': paco_imp_result, 'paco_alns': alns_result
        }
        with open(os.path.join(results_dir, 'compare_paco_imp_vs_alns_small.json'), 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nResults saved to {results_dir}")

    elif args.size == 'large':
        print("Running large experiment (50 customers, RC1-01, medium)...")
        model = load_solomon_instance(50, 'RC1', 1, 4, 'medium', use_drones=True)
        paco_imp_result = run_paco_imp_experiment(model, n_ants=80, max_iter=100, n_runs=args.runs)
        alns_result = run_paco_alns_experiment(model, n_ants=50, max_iter=100, alns_iter=15, n_runs=args.runs)
        ref_cost, ref_tardiness = _compute_shared_reference_point(paco_imp_result, alns_result)
        pi_hv = [calculate_hypervolume(pf, ref_cost, ref_tardiness) for pf in paco_imp_result.get('all_pareto_fronts', [])]
        al_hv = [calculate_hypervolume(pf, ref_cost, ref_tardiness) for pf in alns_result.get('all_pareto_fronts', [])]
        paco_imp_result['mean_hv'] = np.mean(pi_hv)
        paco_imp_result['std_hv'] = np.std(pi_hv)
        alns_result['mean_hv'] = np.mean(al_hv)
        alns_result['std_hv'] = np.std(al_hv)

        pareto_path = os.path.join(results_dir, 'pareto_50c_RC101_medium.png')
        plot_pareto_front(
            paco_imp_result.get('all_costs', []), alns_result.get('all_costs', []),
            paco_imp_result.get('all_tardiness', []), alns_result.get('all_tardiness', []),
            pareto_path
        )

        result = {
            'n_customers': 50, 'rc_type': 'RC1', 'instance_id': 1,
            'n_trucks': 4, 'n_drones': 4, 'endurance_type': 'medium', 'n_runs': args.runs,
            'paco_imp': paco_imp_result, 'paco_alns': alns_result
        }
        with open(os.path.join(results_dir, 'compare_paco_imp_vs_alns_large.json'), 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nResults saved to {results_dir}")

    else:
        main()