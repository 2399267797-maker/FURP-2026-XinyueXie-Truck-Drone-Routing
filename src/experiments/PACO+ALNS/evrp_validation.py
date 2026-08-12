"""Validate NSGA-II / PACO-imp2 / PACO+ALNS W8 on the WCCI-2020 E-CVRP
benchmark instances from https://github.com/Mavrovouniotis/e-cvrp_benchmark_instances

Mapping used by this validation:
  - E-CVRP has no time windows: all customers get a wide [0, 1e9] window,
    so tardiness is zero and the solvers reduce to cost minimisation.
  - Drones are disabled (capacity=0, range=0) because the benchmark is a
    single-objective capacitated VRP without drones.
  - Truck fixed cost = 0, variable cost = 1, so model cost = total distance,
    which is comparable to the benchmark's OPTIMAL_VALUE.
  - Energy/recharging constraints are not modelled by these solvers and are
    therefore outside the validation scope.

Outputs (results/<outdir>):
  <instance>_<algo>.json      per instance / algorithm metrics
  evrp_validation_<outdir>.json  combined results
  evrp_analysis_<outdir>.md   gap vs best-known values and rankings
"""

import os
import sys
import json
import math
import re
import time
import random
import argparse
import importlib.util
from multiprocessing import Pool
from typing import Dict, List

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(os.path.dirname(BASE)))
EXPERIMENTS = os.path.join(PROJ, 'src', 'experiments')

sys.path.insert(0, os.path.join(EXPERIMENTS, 'PACO_vs_NSGA2'))

from models.vrp_model import (  # noqa: E402
    VRPTruckDroneModel, Customer, Vehicle,
)

ALGORITHMS = ['nsga2', 'imp2', 'w8', 'alns']

DEFAULT_INSTANCES = [
    'E-n29-k4-s7', 'E-n30-k3-s7', 'E-n35-k3-s5', 'E-n37-k4-s4',
    'E-n60-k5-s9', 'E-n89-k7-s13', 'F-n49-k4-s4', 'F-n80-k4-s8',
    'F-n140-k5-s5', 'M-n110-k10-s9', 'M-n126-k7-s5',
]

EXTRA_INSTANCES = [
    'X-n147-k7-s4', 'X-n221-k11-s7', 'X-n360-k40-s9',
]

MISSING_PENALTY = 10000.0
OVERLOAD_PENALTY = 1000.0

# F-n140-k5-s5 declares VEHICLES: 5, but total demand 14620 > 5*2210.
# Its header NAME is F-n140-k7-s5 and the source instance is F-n135-k7,
# which uses 7 vehicles, so the validation uses 7 to keep the instance feasible.
VEHICLE_OVERRIDES = {'F-n140-k5-s5': 7}


def parse_evrp(path: str) -> Dict:
    spec = {}
    sections = {}
    current = None
    with open(path, encoding='utf-8') as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            upper = line.upper()
            if ':' in line and upper.split(':')[0] in {
                    'NAME', 'COMMENT', 'TYPE', 'OPTIMAL_VALUE', 'VEHICLES',
                    'DIMENSION', 'ENERGY_CAPACITY', 'ENERGY_CONSUMPTION',
                    'STATIONS', 'CAPACITY', 'EDGE_WEIGHT_TYPE'}:
                key, _, val = line.partition(':')
                spec[key.strip().upper()] = val.strip()
                continue
            sec = upper.rstrip(':').strip()
            if sec in {
                    'NODE_COORD_SECTION', 'DEMAND_SECTION',
                    'STATION_COORD_SECTION', 'STATIONS_COORD_SECTION',
                    'DEPOT_SECTION'}:
                current = sec
                sections[current] = []
                continue
            if upper == 'EOF':
                break
            if current:
                sections[current].append(line)
    return {'spec': spec, 'sections': sections}


def load_evrp_model(data: Dict, n_vehicles_override: int = None) -> VRPTruckDroneModel:
    spec = data['spec']
    sections = data['sections']
    coords = {}
    for line in sections.get('NODE_COORD_SECTION', []):
        parts = line.split()
        if len(parts) >= 3:
            coords[int(parts[0])] = (float(parts[1]), float(parts[2]))
    demands = {}
    for line in sections.get('DEMAND_SECTION', []):
        parts = line.split()
        if len(parts) >= 2:
            demands[int(parts[0])] = float(parts[1])
    stations = set()
    for key in ('STATION_COORD_SECTION', 'STATIONS_COORD_SECTION'):
        for line in sections.get(key, []):
            parts = line.split()
            if parts:
                try:
                    stations.add(int(parts[0]))
                except ValueError:
                    continue
    depot_node = None
    for line in sections.get('DEPOT_SECTION', []):
        parts = line.split()
        if parts and parts[0] != '-1':
            depot_node = int(parts[0])
            break
    if depot_node is None:
        depot_node = 1

    capacity = float(spec.get('CAPACITY', 1000))
    n_vehicles = n_vehicles_override or int(spec.get('VEHICLES', 2))

    model = VRPTruckDroneModel()
    dx, dy = coords[depot_node]
    model.add_depot(dx, dy)
    model.drone_range = 0.0
    model.launch_prep_time = 0.5
    model.retrieval_time = 0.5

    customer_nodes = sorted(
        n for n in coords if n != depot_node and n not in stations
    )
    for idx, node in enumerate(customer_nodes):
        x, y = coords[node]
        model.add_customer(Customer(
            id=idx, x=x, y=y,
            demand=demands.get(node, 0.0),
            service_time=0.0,
            time_window=(0.0, 1e9),
            priority=1,
        ))
    for i in range(n_vehicles):
        model.add_truck(Vehicle(
            id=i, type='truck', capacity=capacity, speed=1.0,
            fixed_cost=0.0, variable_cost=1.0,
        ))
        # Drones are present but unusable (capacity 0 / range 0), which keeps
        # the algorithm code paths valid without changing the benchmark cost.
        model.add_drone(Vehicle(
            id=n_vehicles + i, type='drone', capacity=0.0, speed=1.0,
            fixed_cost=0.0, variable_cost=1.0,
        ))
    return model


_ALGO_MODS: Dict[str, object] = {}


def load_algorithm(algo: str):
    if algo in _ALGO_MODS:
        return _ALGO_MODS[algo]
    if algo == 'nsga2':
        path = os.path.join(EXPERIMENTS, 'NSGA2', 'nsga2_vrp.py')
        spec = importlib.util.spec_from_file_location('nsga2_evrp_cmp', path)
    elif algo == 'imp2':
        path = os.path.join(EXPERIMENTS, 'PACO', 'algorithms', 'paco_imp2.py')
        spec = importlib.util.spec_from_file_location('paco_imp2_evrp_cmp', path)
    elif algo == 'alns':
        path = os.path.join(EXPERIMENTS, 'ALNS', 'pure_alns.py')
        spec = importlib.util.spec_from_file_location('pure_alns_evrp_cmp', path)
    else:
        path = os.path.join(BASE, 'PACO+ALNSW8.py')
        spec = importlib.util.spec_from_file_location('paco_alns_w8_evrp_cmp', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _ALGO_MODS[algo] = mod
    return mod


def evaluate_solution(model: VRPTruckDroneModel, sol) -> Dict:
    n_customers = model.get_number_of_customers()
    raw_cost, _ = model.evaluate_solution(sol)
    raw_tard = model.calculate_pure_tardiness(sol)
    served = set()
    for r in sol:
        served.update(r.customers)
        for m in r.drone_missions:
            served.update(m.customer_ids)
    missing = n_customers - len(served)
    overload = sum(
        max(0.0, sum(model.customers[c].demand for c in r.customers) - model.trucks[0].capacity)
        for r in sol
    )
    cost = raw_cost + missing * MISSING_PENALTY + overload * OVERLOAD_PENALTY
    return {
        'raw_cost': float(raw_cost),
        'raw_tardiness': float(raw_tard),
        'cost': float(cost),
        'missing': int(missing),
        'overload': float(overload),
        'drone_missions': int(sum(len(r.drone_missions) for r in sol)),
        'routes': int(sum(1 for r in sol if r.customers or r.drone_missions)),
    }


def run_single(model, algo_mod, algo: str, max_iter: int, seed: int) -> Dict:
    random.seed(seed)
    np.random.seed(seed)
    if algo == 'nsga2':
        algo_obj = algo_mod.NSGA2VRP(model, pop_size=100, max_gen=max_iter)
    elif algo == 'imp2':
        algo_obj = algo_mod.CollaborativePACO(model, max_iter=max_iter)
    elif algo == 'alns':
        algo_obj = algo_mod.PureALNS(model, max_iter=max_iter)
    else:
        algo_obj = algo_mod.CollaborativePACOALNS(model, max_iter=max_iter)
    t0 = time.time()
    solutions, _ = algo_obj.solve()
    elapsed = time.time() - t0
    costs, tards, missing_counts, overload_counts = [], [], [], []
    for sol in solutions:
        ev = evaluate_solution(model, sol)
        costs.append(ev['cost'])
        tards.append(ev['raw_tardiness'])
        missing_counts.append(ev['missing'])
        overload_counts.append(ev['overload'])
    feasible_costs = [
        c for c, m, o in zip(costs, missing_counts, overload_counts)
        if m == 0 and o <= 1e-6
    ]
    return {
        'solutions': solutions,
        'solve_time': elapsed,
        'n_solutions': len(solutions),
        'best_cost': float(min(feasible_costs)) if feasible_costs else None,
        'mean_cost': float(np.mean(costs)) if costs else 0.0,
        'std_cost': float(np.std(costs)) if costs else 0.0,
        'n_feasible_solutions': len(feasible_costs),
        'n_missing_solutions': int(sum(1 for x in missing_counts if x > 0)),
        'n_overload_solutions': int(sum(1 for x in overload_counts if x > 1e-6)),
    }


def run_experiment(model, algo_mod, algo: str, n_runs: int, max_iter: int,
                   base_seed: int) -> Dict:
    best_costs, feasible_counts = [], []
    solve_times, mean_costs, std_costs = [], [], []
    missing_runs = overload_runs = 0
    for run_idx in range(n_runs):
        seed = base_seed + run_idx * 37
        res = run_single(model, algo_mod, algo, max_iter, seed)
        solve_times.append(res['solve_time'])
        mean_costs.append(res['mean_cost'])
        std_costs.append(res['std_cost'])
        feasible_counts.append(res['n_feasible_solutions'])
        if res['best_cost'] is not None:
            best_costs.append(res['best_cost'])
        missing_runs += res['n_missing_solutions']
        overload_runs += res['n_overload_solutions']
    return {
        'n_runs': n_runs,
        'mean_solve_time': float(np.mean(solve_times)),
        'mean_best_cost': float(np.mean(best_costs)) if best_costs else None,
        'std_best_cost': float(np.std(best_costs)) if best_costs else None,
        'best_cost': float(min(best_costs)) if best_costs else None,
        'mean_cost': float(np.mean(mean_costs)),
        'mean_std_cost': float(np.mean(std_costs)),
        'mean_feasible_solutions': float(np.mean(feasible_counts)),
        'n_missing_solutions': int(missing_runs),
        'n_overload_solutions': int(overload_runs),
    }


def run_config(task: Dict) -> Dict:
    inst, algo = task['instance'], task['algo']
    print(f"[start] {inst} {algo} runs={task['runs']} max_iter={task['max_iter']}",
          flush=True)
    random.seed(task['seed'])
    np.random.seed(task['seed'])
    path = os.path.join(task['data_dir'], f'{inst}.evrp')
    model = load_evrp_model(parse_evrp(path),
                            VEHICLE_OVERRIDES.get(inst))
    algo_mod = load_algorithm(algo)
    result = run_experiment(model, algo_mod, algo, task['runs'],
                            task['max_iter'], task['seed'])
    out_path = os.path.join(task['out_dir'], f'{inst}_{algo}.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[done] {inst} {algo} best={result['best_cost']} "
          f"time={result['mean_solve_time']:.1f}s", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description='E-CVRP benchmark validation')
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--max-iter', type=int, default=30)
    parser.add_argument('--instances', nargs='*', default=None,
                        help='instance names, e.g. E-n29-k4-s7 F-n49-k4-s4')
    parser.add_argument('--include-x', action='store_true',
                        help='include extra X instances (up to 360 nodes)')
    parser.add_argument('--algo', choices=ALGORITHMS + ['all'], default='all',
                        help='which algorithm to run (default: all)')
    parser.add_argument('--outdir', type=str, default='20260805_evrp')
    parser.add_argument('--workers', type=int, default=min(16, os.cpu_count() or 4))
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    data_dir = os.path.join(EXPERIMENTS, 'e-cvrp_benchmark_instances')
    results_dir = os.path.join(BASE, 'results', args.outdir)
    os.makedirs(results_dir, exist_ok=True)

    instances = args.instances if args.instances else list(DEFAULT_INSTANCES)
    if args.include_x:
        instances += EXTRA_INSTANCES
    algos = ALGORITHMS if args.algo == 'all' else [args.algo]

    tasks = []
    for idx, inst in enumerate(instances):
        for algo in algos:
            out_path = os.path.join(results_dir, f'{inst}_{algo}.json')
            if os.path.exists(out_path) and not args.force:
                continue
            tasks.append({
                'instance': inst, 'algo': algo, 'runs': args.runs,
                'max_iter': args.max_iter,
                'seed': 20260805 + idx * 1000 + ALGORITHMS.index(algo),
                'data_dir': data_dir, 'out_dir': results_dir,
            })
    if tasks:
        with Pool(args.workers) as pool:
            pool.map(run_config, tasks)

    optimal = {}
    results_by_inst = {}
    for inst in instances:
        data = parse_evrp(os.path.join(data_dir, f'{inst}.evrp'))
        raw_opt = data['spec'].get('OPTIMAL_VALUE', 'nan')
        try:
            optimal[inst] = float(raw_opt)
        except ValueError:
            m = re.search(r'-?\d+(?:\.\d+)?', raw_opt)
            optimal[inst] = float(m.group()) if m else float('nan')
        results_by_inst[inst] = {}
        for algo in ALGORITHMS:
            path = os.path.join(results_dir, f'{inst}_{algo}.json')
            if os.path.exists(path):
                with open(path, encoding='utf-8') as f:
                    results_by_inst[inst][algo] = json.load(f)

    combined = {
        'instances': {
            inst: {'optimal_value': optimal[inst], 'algorithms': results_by_inst[inst]}
            for inst in instances
        }
    }
    combined_path = os.path.join(results_dir, f'evrp_validation_{args.outdir}.json')
    with open(combined_path, 'w', encoding='utf-8') as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    print(f"Saved: {combined_path}", flush=True)

    lines = ["# E-CVRP Benchmark Validation (NSGA-II / PACO-imp2 / PACO+ALNS W8 / Pure ALNS)", ""]
    lines.append("## Mapping")
    lines.append("- E-CVRP has no time windows: solvers receive wide [0, 1e9] windows (tardiness = 0).")
    lines.append("- Drones disabled (capacity 0, range 0); truck fixed cost 0, variable cost 1, so cost = distance.")
    lines.append("- Energy/recharging constraints are not part of these solvers and are not validated here.")
    lines.append("- Pure ALNS uses the same outer-iteration budget (max_iter) and scale-adaptive alns_iter as W8, "
                 "with the PACO construction and pheromone update removed.")
    lines.append("")
    lines.append("| Instance | Opt/BKS | Algo | Best Cost | Gap % | Feasible sols | Missing | Overload | Time (s) |")
    lines.append("|----------|---------|------|-----------|-------|---------------|---------|----------|----------|")
    gaps = {a: [] for a in ALGORITHMS}
    wins = {a: 0 for a in ALGORITHMS}
    for inst in instances:
        opt = optimal[inst]
        for algo in ALGORITHMS:
            r = results_by_inst[inst].get(algo)
            if not r:
                continue
            bc = r['best_cost']
            gap = ((bc / opt - 1.0) * 100.0
                   if bc is not None and math.isfinite(opt) and opt > 0
                   else float('nan'))
            if bc is not None and math.isfinite(opt) and opt > 0:
                gaps[algo].append(gap)
            if bc is not None and math.isfinite(opt) and opt > 0 and bc <= opt:
                wins[algo] += 1
            opt_str = f"{opt:.0f}" if math.isfinite(opt) else '-'
            gap_str = f"{gap:.2f}" if math.isfinite(gap) else '-'
            lines.append(
                f"| {inst} | {opt_str} | {algo} | "
                f"{bc if bc is not None else float('nan'):.2f} | {gap_str} | "
                f"{r['mean_feasible_solutions']:.1f} | {r['n_missing_solutions']} | "
                f"{r['n_overload_solutions']} | {r['mean_solve_time']:.1f} |")
    lines.append("")
    lines.append("| Algo | Mean gap % vs best-known | Instances at best-known |")
    lines.append("|------|--------------------------|-------------------------|")
    for algo in ALGORITHMS:
        mean_gap = float(np.mean(gaps[algo])) if gaps[algo] else float('nan')
        lines.append(f"| {algo} | {mean_gap:.2f} | {wins[algo]}/{len(instances)} |")
    lines.append("")
    lines.append("Note: NSGA-II now applies a capacity-aware decode repair and an "
                 "overload penalty in its fitness; infeasible solutions are still "
                 "excluded from best-cost statistics.")
    md_path = os.path.join(results_dir, f'evrp_analysis_{args.outdir}.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved: {md_path}", flush=True)


if __name__ == '__main__':
    main()
