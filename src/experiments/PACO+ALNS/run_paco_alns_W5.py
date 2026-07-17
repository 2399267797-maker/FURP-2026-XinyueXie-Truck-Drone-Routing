"""
PACO+ALNS v2.0 — Standalone experiment runner (adaptive parameter scaling).

Runs CollaborativePACOALNS (v2.0) on all Solomon RC configurations
(25c/50c/100c, RC1/RC2, medium/high endurance) and saves results,
Pareto plots, and route visualizations to results/20260716/.

Usage:
    python run_paco_alns_v2.py                    # full 12-config experiment
    python run_paco_alns_v2.py --size small       # 25c only (4 configs)
    python run_paco_alns_v2.py --size large       # 100c only (4 configs)
    python run_paco_alns_v2.py --runs 3           # quick test
"""
import os
import sys
import json
import time
import argparse
import importlib.util
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

# ── Path setup ──────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(os.path.dirname(BASE)))  # project root

sys.path.insert(0, os.path.join(PROJ, 'src', 'experiments', 'PACO', 'data'))
sys.path.insert(0, os.path.join(PROJ, 'src', 'experiments', 'PACO_vs_NSGA2'))

from models.vrp_model import VRPTruckDroneModel, Route
from utils.visualizer import Visualizer

# ── Load PACO+ALNS ──────────────────────────────────────────────────────────
alns_mod_path = os.path.join(BASE, 'PACO+ALNSW5.py')
spec = importlib.util.spec_from_file_location("paco_alns", alns_mod_path)
paco_alns_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paco_alns_mod)
CollaborativePACOALNS = paco_alns_mod.CollaborativePACOALNS

# ── Load Solomon loader ─────────────────────────────────────────────────────
from solomon_loader_imp import SolomonLoaderImp


# ═════════════════════════════════════════════════════════════════════════════
#  Core helpers
# ═════════════════════════════════════════════════════════════════════════════

def load_model(n_customers: int, rc_type: str = 'RC1', instance_id: int = 1,
               n_vehicles: int = 2, endurance_type: str = 'medium',
               use_drones: bool = True) -> VRPTruckDroneModel:
    loader = SolomonLoaderImp()
    kwargs = dict(n_customers=n_customers, instance_id=instance_id,
                  n_vehicles=n_vehicles, endurance_type=endurance_type,
                  use_drones=use_drones)
    if rc_type == 'RC1':
        return loader.load_rc1_instance(**kwargs)
    else:
        return loader.load_rc2_instance(**kwargs)


def calculate_hypervolume(pareto_front, ref_cost=None, ref_tard=None):
    if not pareto_front:
        return 0.0
    pts = np.array(pareto_front)
    if ref_cost is None:
        ref_cost = np.max(pts[:, 0]) * 1.1
    if ref_tard is None:
        ref_tard = np.max(pts[:, 1]) * 1.1
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    hv, prev_x = 0.0, ref_cost
    for i in range(len(pts) - 1, -1, -1):
        x, y = pts[i]
        hv += max(0, prev_x - x) * max(0, ref_tard - y)
        prev_x = x
    return hv


def _compute_pareto_front(costs, tardiness):
    pts = np.column_stack([np.array(costs), np.array(tardiness)])
    if len(pts) == 0:
        return np.array([]), np.array([])
    dominated = np.zeros(len(pts), dtype=bool)
    for i in range(len(pts)):
        if dominated[i]: continue
        for j in range(len(pts)):
            if i == j or dominated[j]: continue
            if (pts[i, 0] <= pts[j, 0] and pts[i, 1] <= pts[j, 1] and
                (pts[i, 0] < pts[j, 0] or pts[i, 1] < pts[j, 1])):
                dominated[j] = True
    pf = pts[~dominated]
    pf = pf[np.argsort(pf[:, 0])]
    return pf[:, 0], pf[:, 1]


# ═════════════════════════════════════════════════════════════════════════════
#  Experiment runner
# ═════════════════════════════════════════════════════════════════════════════

def run_alns_single(model: VRPTruckDroneModel) -> Dict:
    """Single PACO+ALNS v2.0 trial."""
    algo = CollaborativePACOALNS(model, max_iter=100)
    start = time.time()
    solutions, _ = algo.solve()
    elapsed = time.time() - start

    costs, tardiness, pf = [], [], []
    for sol in solutions:
        c, _ = model.evaluate_solution(sol)
        t = model.calculate_pure_tardiness(sol)
        costs.append(c)
        tardiness.append(t)
        pf.append((c, t))

    hv = calculate_hypervolume(pf)
    return {'solutions': solutions, 'pareto_front': pf,
            'costs': costs, 'tardiness': tardiness,
            'solve_time': elapsed, 'hypervolume': hv}


def run_experiment(model: VRPTruckDroneModel, n_runs: int = 10) -> Dict:
    """Multi-run experiment for PACO+ALNS v2.0."""
    print("\n--- PACO+ALNS v2.0 ---")
    keys = ['costs', 'tardiness', 'solve_times', 'hypervolumes',
            'solutions', 'pareto_fronts']
    acc = {k: [] for k in keys}

    for run_idx in range(n_runs):
        print(f"  Run {run_idx + 1}/{n_runs} ...", end='')
        res = run_alns_single(model)
        print(f" done  ({res['solve_time']:.1f}s)")
        acc['costs'].extend(res['costs'])
        acc['tardiness'].extend(res['tardiness'])
        acc['solve_times'].append(res['solve_time'])
        acc['hypervolumes'].append(res['hypervolume'])
        acc['solutions'].extend(res['solutions'])
        acc['pareto_fronts'].append(res['pareto_front'])

    mean_hv = np.mean(acc['hypervolumes'])
    std_hv  = np.std(acc['hypervolumes'])

    print(f"  Total solutions: {len(acc['solutions'])}")
    print(f"  Mean Cost:       {np.mean(acc['costs']):.2f} ± {np.std(acc['costs']):.2f}")
    print(f"  Mean Tardiness:  {np.mean(acc['tardiness']):.2f} ± {np.std(acc['tardiness']):.2f}")
    print(f"  HV:              {mean_hv:.2f} ± {std_hv:.2f}")
    print(f"  Avg Time:        {np.mean(acc['solve_times']):.2f}s")

    return {
        'n_runs': n_runs,
        'mean_cost': float(np.mean(acc['costs'])),
        'std_cost': float(np.std(acc['costs'])),
        'mean_tardiness': float(np.mean(acc['tardiness'])),
        'std_tardiness': float(np.std(acc['tardiness'])),
        'mean_solve_time': float(np.mean(acc['solve_times'])),
        'mean_hv': float(mean_hv),
        'std_hv': float(std_hv),
        'all_costs': acc['costs'],
        'all_tardiness': acc['tardiness'],
        'all_solutions': acc['solutions'],
        'all_pareto_fronts': acc['pareto_fronts'],
    }


# ═════════════════════════════════════════════════════════════════════════════
#  Plotting
# ═════════════════════════════════════════════════════════════════════════════

def plot_pareto(result: Dict, save_path: str):
    """Plot Pareto front of PACO+ALNS v2.0."""
    costs = result.get('all_costs', [])
    tardiness = result.get('all_tardiness', [])
    if not costs or not tardiness:
        return
    pf_c, pf_t = _compute_pareto_front(costs, tardiness)

    plt.figure(figsize=(10, 8))
    plt.scatter(pf_c, pf_t, facecolors='none', edgecolors='#d62728',
                marker='^', s=60, label='PACO+ALNS v2.0')
    plt.plot(pf_c, pf_t, c='#d62728', linestyle='-', linewidth=1.5, alpha=0.7)

    plt.xlabel('Cost of Travel', fontsize=12)
    plt.ylabel('Penalty due to Tardiness', fontsize=12)
    plt.title('PACO+ALNS v2.0 — Non-dominated Front', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  Pareto plot: {save_path}")


def plot_routes(model: VRPTruckDroneModel, result: Dict, title: str, save_base: str):
    """Plot three route maps: min cost, min tardiness, and compromise."""
    if not result['all_solutions']:
        return
    pf = result.get('all_pareto_fronts', [])
    sols = result['all_solutions']
    if not pf or not sols:
        return
    flat = [pt for run in pf for pt in run]
    if not flat:
        return

    # Step 1: compute joint non-dominated front across all runs
    pts = np.array(flat)
    dominated = np.zeros(len(pts), dtype=bool)
    for i in range(len(pts)):
        if dominated[i]:
            continue
        for j in range(len(pts)):
            if i == j or dominated[j]:
                continue
            if (pts[i, 0] <= pts[j, 0] and pts[i, 1] <= pts[j, 1] and
                (pts[i, 0] < pts[j, 0] or pts[i, 1] < pts[j, 1])):
                dominated[j] = True
    pareto_indices = np.where(~dominated)[0]
    flat_pf = [flat[i] for i in pareto_indices]

    # Step 2: select representatives from the joint non-dominated front only
    local_idx = np.argmin([f[0] for f in flat_pf])
    idx_cost = int(pareto_indices[local_idx])
    local_idx = np.argmin([f[1] for f in flat_pf])
    idx_tard = int(pareto_indices[local_idx])
    # Compromise: knee point — farthest from the line connecting the two extremes
    pf_sorted = sorted(flat_pf, key=lambda x: x[0])
    c1, t1 = pf_sorted[0]       # min cost extreme
    c2, t2 = pf_sorted[-1]      # min tardiness extreme
    line_len = max(np.sqrt((c2 - c1)**2 + (t2 - t1)**2), 1e-6)
    best_dist = -1.0
    knee_local = 0
    for i, (c, t) in enumerate(pf_sorted):
        dist = abs((c2 - c1) * (t1 - t) - (c1 - c) * (t2 - t1)) / line_len
        if dist > best_dist:
            best_dist = dist
            knee_local = i
    idx_comp = int(pareto_indices[knee_local])

    vis = Visualizer(model)
    for label, idx in [('min_cost', idx_cost), ('min_tardiness', idx_tard), ('compromise', idx_comp)]:
        if idx < len(sols):
            route = sols[idx]
            save_path = save_base.replace('.png', f'_{label}.png')
            t = f'{title} | {label}'
            vis.plot_routes(route, title=t, save_path=save_path, show_all_nodes=True)
            print(f"  Route plot ({label}): {save_path}")


# ═════════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', choices=['small', 'medium', 'large', 'all'],
                        default='all', help='Customer size to run')
    parser.add_argument('--runs', type=int, default=10, help='Repetitions per config')
    args = parser.parse_args()

    results_dir = os.path.join(BASE, 'results', '20260716')
    os.makedirs(results_dir, exist_ok=True)

    configs = [
        # 25 customers
        {'n_customers': 25, 'rc_type': 'RC1', 'n_vehicles': 2, 'endurance_type': 'medium', 'size': 'small'},
        {'n_customers': 25, 'rc_type': 'RC1', 'n_vehicles': 2, 'endurance_type': 'high',   'size': 'small'},
        {'n_customers': 25, 'rc_type': 'RC2', 'n_vehicles': 2, 'endurance_type': 'medium', 'size': 'small'},
        {'n_customers': 25, 'rc_type': 'RC2', 'n_vehicles': 2, 'endurance_type': 'high',   'size': 'small'},
        # 50 customers
        {'n_customers': 50, 'rc_type': 'RC1', 'n_vehicles': 4, 'endurance_type': 'medium', 'size': 'medium'},
        {'n_customers': 50, 'rc_type': 'RC1', 'n_vehicles': 4, 'endurance_type': 'high',   'size': 'medium'},
        {'n_customers': 50, 'rc_type': 'RC2', 'n_vehicles': 4, 'endurance_type': 'medium', 'size': 'medium'},
        {'n_customers': 50, 'rc_type': 'RC2', 'n_vehicles': 4, 'endurance_type': 'high',   'size': 'medium'},
        {'n_customers': 50, 'rc_type': 'RC1', 'n_vehicles': 6, 'endurance_type': 'medium', 'size': 'medium'},
        {'n_customers': 50, 'rc_type': 'RC1', 'n_vehicles': 6, 'endurance_type': 'high',   'size': 'medium'},
        {'n_customers': 50, 'rc_type': 'RC2', 'n_vehicles': 6, 'endurance_type': 'medium', 'size': 'medium'},
        {'n_customers': 50, 'rc_type': 'RC2', 'n_vehicles': 6, 'endurance_type': 'high',   'size': 'medium'},
        # 100 customers
        {'n_customers': 100, 'rc_type': 'RC1', 'n_vehicles': 10, 'endurance_type': 'medium', 'size': 'large'},
        {'n_customers': 100, 'rc_type': 'RC1', 'n_vehicles': 10, 'endurance_type': 'high',   'size': 'large'},
        {'n_customers': 100, 'rc_type': 'RC2', 'n_vehicles': 10, 'endurance_type': 'medium', 'size': 'large'},
        {'n_customers': 100, 'rc_type': 'RC2', 'n_vehicles': 10, 'endurance_type': 'high',   'size': 'large'},
    ]

    if args.size != 'all':
        configs = [c for c in configs if c['size'] == args.size]

    all_results = []

    for cfg in configs:
        n, rc, inst, nv, end = cfg['n_customers'], cfg['rc_type'], 1, cfg['n_vehicles'], cfg['endurance_type']
        print(f"\n{'='*60}")
        print(f"PACO+ALNS v2.0 | {n}c | {rc} | {nv}V | {end}")
        print(f"{'='*60}")

        model = load_model(n, rc, inst, nv, end, use_drones=True)
        print(f"  Trucks={model.get_number_of_trucks()}, Drones={model.get_number_of_drones()}, "
              f"Range={model.drone_range}km")

        result = run_experiment(model, n_runs=args.runs)

        result_dict = {
            'n_customers': n,
            'rc_type': rc,
            'instance_id': inst,
            'n_vehicles': nv,
            'endurance_type': end,
            'n_runs': args.runs,
            'algo': 'PACO+ALNS_v2.0',
            'mean_cost': result['mean_cost'],
            'std_cost': result['std_cost'],
            'mean_tardiness': result['mean_tardiness'],
            'std_tardiness': result['std_tardiness'],
            'mean_hv': result['mean_hv'],
            'std_hv': result['std_hv'],
            'mean_solve_time': result['mean_solve_time'],
            'all_costs': result['all_costs'],
            'all_tardiness': result['all_tardiness'],
            'all_pareto_fronts': result['all_pareto_fronts'],
        }

        # Pareto plot
        exp_key = f"{n}c_{rc}{inst:02d}_{nv}V_{end}"
        plot_pareto(result, os.path.join(results_dir, f'pareto_{exp_key}.png'))

        # Route plots (min cost, min tardiness, compromise)
        n_trucks = model.get_number_of_trucks()
        n_drones = model.get_number_of_drones()
        title = f'PACO+ALNS v2.0 | {n}C | {n_trucks}T+{n_drones}D | {end}'
        plot_routes(model, result, title,
                    os.path.join(results_dir, f'alns_{exp_key}.png'))

        all_results.append(result_dict)

    # ── Save JSON ──
    json_path = os.path.join(results_dir, 'alns_v2_results.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved: {json_path}")

    # ── Save summary markdown ──
    summary = "# PACO+ALNS v2.0 Results\n\n"
    summary += "**Date**: 2026-07-16\n"
    summary += f"**Repetitions**: {args.runs} per config\n\n"
    summary += "| Config | Customers | Type | Vehicles | Endurance | Mean Cost ± Std | Mean Tardiness ± Std | HV | Time (s) |\n"
    summary += "|--------|-----------|------|----------|-----------|-----------------|----------------------|----|----------|\n"

    for r in all_results:
        ek = f"{r['n_customers']}c_{r['rc_type']}{r['instance_id']:02d}_{r['endurance_type']}"
        summary += (
            f"| {ek} | {r['n_customers']} | {r['rc_type']} | "
            f"{r['n_vehicles']}V | {r['endurance_type']} | "
            f"{r['mean_cost']:.2f} ± {r['std_cost']:.2f} | "
            f"{r['mean_tardiness']:.2f} ± {r['std_tardiness']:.2f} | "
            f"{r['mean_hv']:.2f} | {r['mean_solve_time']:.1f} |\n"
        )

    md_path = os.path.join(results_dir, 'alns_v2_summary.md')
    with open(md_path, 'w') as f:
        f.write(summary)
    print(f"Saved: {md_path}")

    print(f"\n{'='*60}")
    print("All experiments complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()