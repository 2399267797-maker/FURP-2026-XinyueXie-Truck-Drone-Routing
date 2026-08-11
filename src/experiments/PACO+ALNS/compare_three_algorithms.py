"""Three-way comparison: NSGA-II vs PACO-imp2 vs PACO+ALNS (W7 or W8).

The experiment follows the Solomon RC configuration list used by
run_paco_alns_W6.py (16 configs: 25/50/100 customers x RC1/RC2 x
medium/high endurance), and writes:

  results/<outdir>/<key>_<algo>.json          per config / algorithm
  results/<outdir>/compare_three_<date>.json  combined results
  results/<outdir>/pareto_compare_<key>.png   three-way Pareto plot
  results/<outdir>/routes_<algo>_<key>_*.png  representative route maps
  results/<outdir>/analysis_<date>.md         comparison analysis

Usage:
  python compare_three_algorithms.py --full --runs 10
  python compare_three_algorithms.py --full --size small --runs 2
  python compare_three_algorithms.py --n_customers 25 --rc_type RC1 --runs 1
"""

import os
import sys
import json
import time
import random
import argparse
import importlib.util
from multiprocessing import Pool
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(os.path.dirname(BASE)))
EXPERIMENTS = os.path.join(PROJ, 'src', 'experiments')

sys.path.insert(0, os.path.join(EXPERIMENTS, 'PACO', 'data'))
sys.path.insert(0, os.path.join(EXPERIMENTS, 'PACO_vs_NSGA2'))

from models.vrp_model import VRPTruckDroneModel  # noqa: E402
from utils.visualizer import Visualizer  # noqa: E402
from solomon_loader_imp import SolomonLoaderImp  # noqa: E402


ALL_CONFIGS = [
    {'n_customers': 25, 'rc_type': 'RC1', 'instance_id': 1, 'n_vehicles': 2, 'endurance_type': 'medium', 'size': 'small'},
    {'n_customers': 25, 'rc_type': 'RC1', 'instance_id': 1, 'n_vehicles': 2, 'endurance_type': 'high',   'size': 'small'},
    {'n_customers': 25, 'rc_type': 'RC2', 'instance_id': 1, 'n_vehicles': 2, 'endurance_type': 'medium', 'size': 'small'},
    {'n_customers': 25, 'rc_type': 'RC2', 'instance_id': 1, 'n_vehicles': 2, 'endurance_type': 'high',   'size': 'small'},
    {'n_customers': 50, 'rc_type': 'RC1', 'instance_id': 1, 'n_vehicles': 4, 'endurance_type': 'medium', 'size': 'medium'},
    {'n_customers': 50, 'rc_type': 'RC1', 'instance_id': 1, 'n_vehicles': 4, 'endurance_type': 'high',   'size': 'medium'},
    {'n_customers': 50, 'rc_type': 'RC2', 'instance_id': 1, 'n_vehicles': 4, 'endurance_type': 'medium', 'size': 'medium'},
    {'n_customers': 50, 'rc_type': 'RC2', 'instance_id': 1, 'n_vehicles': 4, 'endurance_type': 'high',   'size': 'medium'},
    {'n_customers': 50, 'rc_type': 'RC1', 'instance_id': 1, 'n_vehicles': 6, 'endurance_type': 'medium', 'size': 'medium'},
    {'n_customers': 50, 'rc_type': 'RC1', 'instance_id': 1, 'n_vehicles': 6, 'endurance_type': 'high',   'size': 'medium'},
    {'n_customers': 50, 'rc_type': 'RC2', 'instance_id': 1, 'n_vehicles': 6, 'endurance_type': 'medium', 'size': 'medium'},
    {'n_customers': 50, 'rc_type': 'RC2', 'instance_id': 1, 'n_vehicles': 6, 'endurance_type': 'high',   'size': 'medium'},
    {'n_customers': 100, 'rc_type': 'RC1', 'instance_id': 1, 'n_vehicles': 10, 'endurance_type': 'medium', 'size': 'large'},
    {'n_customers': 100, 'rc_type': 'RC1', 'instance_id': 1, 'n_vehicles': 10, 'endurance_type': 'high',   'size': 'large'},
    {'n_customers': 100, 'rc_type': 'RC2', 'instance_id': 1, 'n_vehicles': 10, 'endurance_type': 'medium', 'size': 'large'},
    {'n_customers': 100, 'rc_type': 'RC2', 'instance_id': 1, 'n_vehicles': 10, 'endurance_type': 'high',   'size': 'large'},
]

ALGORITHMS = ['nsga2', 'imp2', 'w7']

ALGO_STYLES = {
    'nsga2': {'label': 'NSGA-II', 'color': '#1f77b4', 'marker': 'o'},
    'imp2': {'label': 'PACO-imp2', 'color': '#2ca02c', 'marker': 's'},
    'w7': {'label': 'PACO+ALNS W7', 'color': '#d62728', 'marker': '^'},
}

# Penalties applied uniformly to every returned solution before computing
# Pareto fronts and aggregate metrics.  They keep infeasible solutions from
# artificially dominating feasible ones.
MISSING_PENALTY = 10000.0
OVERLOAD_PENALTY = 1000.0


def load_model(n_customers: int, rc_type: str, instance_id: int,
               n_vehicles: int, endurance_type: str,
               use_drones: bool = True) -> VRPTruckDroneModel:
    loader = SolomonLoaderImp()
    family = rc_type[:-1]
    typ = int(rc_type[-1])
    if n_vehicles is None:
        n_vehicles = loader._auto_n_vehicles(typ, n_customers)
    return loader.load_instance(family, typ, instance_id, n_customers,
                                n_vehicles, endurance_type, use_drones)


def all_solomon_configs():
    """All standard Solomon instances in PACO_vs_NSGA2/data/text that were
    not covered by the W6-style RC101/RC201 experiment (25/50/100 x medium/high)."""
    import re
    data_dir = os.path.join(EXPERIMENTS, 'PACO_vs_NSGA2', 'data', 'text')
    tested = {('RC', 1, 1), ('RC', 2, 1)}
    loader = SolomonLoaderImp()
    configs = []
    for fn in sorted(os.listdir(data_dir)):
        m = re.match(r'^([A-Z]+)([12])(\d+)\.txt$', fn)
        if not m:
            continue
        family, typ, iid = m.group(1), int(m.group(2)), int(m.group(3))
        if (family, typ, iid) in tested:
            continue
        for n_customers in (25, 50, 100):
            nv = loader._auto_n_vehicles(typ, n_customers)
            for endurance in ('medium', 'high'):
                configs.append({
                    'n_customers': n_customers,
                    'rc_type': f'{family}{typ}',
                    'instance_id': iid,
                    'n_vehicles': nv,
                    'endurance_type': endurance,
                    'size': 'all',
                })
    return configs


_ALGO_MODS: Dict[str, object] = {}


def load_algorithm(algo: str, w7_module: str = 'w8'):
    cache_key = f"{algo}:{w7_module}"
    if cache_key in _ALGO_MODS:
        return _ALGO_MODS[cache_key]
    if algo == 'nsga2':
        path = os.path.join(EXPERIMENTS, 'NSGA2', 'nsga2_vrp.py')
        spec = importlib.util.spec_from_file_location('nsga2_vrp_cmp', path)
    elif algo == 'imp2':
        path = os.path.join(EXPERIMENTS, 'PACO', 'algorithms', 'paco_imp2.py')
        spec = importlib.util.spec_from_file_location('paco_imp2_cmp', path)
    else:
        fname = 'PACO+ALNSW8.py' if w7_module == 'w8' else 'PACO+ALNSW7.py'
        path = os.path.join(BASE, fname)
        spec = importlib.util.spec_from_file_location(f'paco_alns_{w7_module}_cmp', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _ALGO_MODS[cache_key] = mod
    return mod


def calculate_hypervolume(pareto_front, ref_cost=None, ref_tard=None):
    if not pareto_front:
        return 0.0
    pts = np.array(pareto_front, dtype=float)
    if len(pts) == 0:
        return 0.0
    if ref_cost is None:
        ref_cost = float(np.max(pts[:, 0]) * 1.1)
    if ref_tard is None:
        ref_tard = float(np.max(pts[:, 1]) * 1.1)
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    hv, prev_x = 0.0, ref_cost
    for i in range(len(pts) - 1, -1, -1):
        x, y = pts[i]
        hv += max(0.0, prev_x - x) * max(0.0, ref_tard - y)
        prev_x = x
    return hv


def compute_pareto_front(points):
    if len(points) == 0:
        return np.array([])
    pts = np.array(points, dtype=float)
    if len(pts) == 0:
        return pts
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
    pf = pts[~dominated]
    return pf[np.argsort(pf[:, 0])]


def evaluate_solution(model: VRPTruckDroneModel, sol) -> Dict:
    n_customers = model.get_number_of_customers()
    raw_cost, _ = model.evaluate_solution(sol)
    raw_tard = model.calculate_pure_tardiness(sol)

    served = set()
    drone_n = 0
    route_n = 0
    for r in sol:
        served.update(r.customers)
        route_n += 1
        for m in r.drone_missions:
            served.update(m.customer_ids)
            drone_n += len(m.customer_ids)
    missing = n_customers - len(served)
    overload = sum(
        max(0.0, sum(model.customers[c].demand for c in r.customers) - model.trucks[0].capacity)
        for r in sol
    )
    cost = raw_cost + missing * MISSING_PENALTY + overload * OVERLOAD_PENALTY
    tard = raw_tard + missing * MISSING_PENALTY
    return {
        'raw_cost': float(raw_cost),
        'raw_tardiness': float(raw_tard),
        'cost': float(cost),
        'tardiness': float(tard),
        'missing': int(missing),
        'overload': float(overload),
        'drone_missions': int(drone_n),
        'routes': int(route_n),
    }


def run_single(model, algo_mod, algo: str, max_iter: int, seed: int) -> Dict:
    random.seed(seed)
    np.random.seed(seed)

    if algo == 'nsga2':
        algo_obj = algo_mod.NSGA2VRP(model, pop_size=100, max_gen=max_iter)
    elif algo == 'imp2':
        algo_obj = algo_mod.CollaborativePACO(model, max_iter=max_iter)
    else:
        algo_obj = algo_mod.CollaborativePACOALNS(model, max_iter=max_iter)

    t0 = time.time()
    solutions, _ = algo_obj.solve()
    elapsed = time.time() - t0

    costs, tardiness, front = [], [], []
    missing_counts, overload_amounts, drone_counts, route_counts = [], [], [], []
    for sol in solutions:
        ev = evaluate_solution(model, sol)
        costs.append(ev['cost'])
        tardiness.append(ev['tardiness'])
        front.append((ev['cost'], ev['tardiness']))
        missing_counts.append(ev['missing'])
        overload_amounts.append(ev['overload'])
        drone_counts.append(ev['drone_missions'])
        route_counts.append(ev['routes'])

    return {
        'solutions': solutions,
        'pareto_front': front,
        'costs': costs,
        'tardiness': tardiness,
        'solve_time': elapsed,
        'n_solutions': len(solutions),
        'mean_drone_missions': float(np.mean(drone_counts)) if drone_counts else 0.0,
        'mean_routes': float(np.mean(route_counts)) if route_counts else 0.0,
        'n_missing_solutions': int(sum(1 for x in missing_counts if x > 0)),
        'max_missing': int(max(missing_counts)) if missing_counts else 0,
        'n_overload_solutions': int(sum(1 for x in overload_amounts if x > 1e-6)),
        'max_overload': float(max(overload_amounts)) if overload_amounts else 0.0,
    }


def run_experiment(model, algo_mod, algo: str, n_runs: int, max_iter: int,
                   base_seed: int) -> Dict:
    acc = {'costs': [], 'tardiness': [], 'solve_times': [],
           'pareto_fronts': [], 'solutions': []}
    n_sol_list, drone_means, route_means = [], [], []
    missing_runs = overload_runs = max_missing = 0
    max_overload = 0.0

    for run_idx in range(n_runs):
        seed = base_seed + run_idx * 37
        res = run_single(model, algo_mod, algo, max_iter, seed)
        acc['costs'].extend(res['costs'])
        acc['tardiness'].extend(res['tardiness'])
        acc['solve_times'].append(res['solve_time'])
        acc['pareto_fronts'].append(res['pareto_front'])
        acc['solutions'].extend(res['solutions'])
        n_sol_list.append(res['n_solutions'])
        drone_means.append(res['mean_drone_missions'])
        route_means.append(res['mean_routes'])
        missing_runs += res['n_missing_solutions']
        overload_runs += res['n_overload_solutions']
        max_missing = max(max_missing, res['max_missing'])
        max_overload = max(max_overload, res['max_overload'])

    return {
        'n_runs': n_runs,
        'mean_cost': float(np.mean(acc['costs'])) if acc['costs'] else 0.0,
        'std_cost': float(np.std(acc['costs'])) if acc['costs'] else 0.0,
        'mean_tardiness': float(np.mean(acc['tardiness'])) if acc['tardiness'] else 0.0,
        'std_tardiness': float(np.std(acc['tardiness'])) if acc['tardiness'] else 0.0,
        'mean_solve_time': float(np.mean(acc['solve_times'])),
        'mean_n_solutions': float(np.mean(n_sol_list)),
        'mean_drone_missions': float(np.mean(drone_means)),
        'mean_routes': float(np.mean(route_means)),
        'n_missing_solutions': int(missing_runs),
        'n_overload_solutions': int(overload_runs),
        'max_missing': int(max_missing),
        'max_overload': float(max_overload),
        'all_costs': [float(x) for x in acc['costs']],
        'all_tardiness': [float(x) for x in acc['tardiness']],
        'all_pareto_fronts': [[[float(a), float(b)] for a, b in pf] for pf in acc['pareto_fronts']],
        'all_solutions': acc['solutions'],
    }


def finalize_result(result: Dict, ref_cost: float, ref_tard: float) -> Dict:
    out = dict(result)
    fronts = result.get('all_pareto_fronts', [])
    per_run_hv = [calculate_hypervolume(pf, ref_cost, ref_tard) for pf in fronts]
    union = [pt for pf in fronts for pt in pf]
    pf_union = compute_pareto_front(union)
    out['mean_hv'] = float(np.mean(per_run_hv)) if per_run_hv else 0.0
    out['std_hv'] = float(np.std(per_run_hv)) if per_run_hv else 0.0
    out['combined_hv'] = float(calculate_hypervolume(pf_union.tolist(), ref_cost, ref_tard))
    out['n_front_points'] = int(len(pf_union))
    out['all_hypervolumes'] = [float(x) for x in per_run_hv]
    out.pop('all_solutions', None)
    return out


def config_key(cfg: Dict) -> str:
    return (f"{cfg['n_customers']}c_{cfg['rc_type']}{cfg['instance_id']:02d}_"
            f"{cfg['n_vehicles']}V_{cfg['endurance_type']}")


def run_config(task: Dict) -> Dict:
    cfg, algo = task['config'], task['algo']
    key = config_key(cfg)
    print(f"[start] {key} {algo} runs={task['runs']} max_iter={task['max_iter']} "
          f"seed={task['seed']}", flush=True)

    random.seed(task['seed'])
    np.random.seed(task['seed'])
    model = load_model(cfg['n_customers'], cfg['rc_type'], cfg['instance_id'],
                       cfg['n_vehicles'], cfg['endurance_type'], use_drones=True)
    algo_mod = load_algorithm(algo, task.get('w7_module', 'w8'))
    result = run_experiment(model, algo_mod, algo, task['runs'],
                            task['max_iter'], task['seed'])

    if task.get('route_plots', True):
        if algo == 'nsga2':
            algo_label = 'NSGA-II'
        elif algo == 'imp2':
            algo_label = 'PACO-imp2'
        else:
            algo_label = ('PACO+ALNS W8' if task.get('w7_module', 'w8') == 'w8'
                          else 'PACO+ALNS W7')
        title = (f"{algo_label} | {cfg['n_customers']}C | "
                 f"{model.get_number_of_trucks()}T+{model.get_number_of_drones()}D | "
                 f"{cfg['endurance_type']}")
        plot_routes(model, result, title,
                    os.path.join(task['out_dir'], f"routes_{algo}_{key}.png"))

    raw = dict(result)
    raw.pop('all_solutions', None)
    raw['module'] = task.get('w7_module', 'w8')
    out_path = os.path.join(task['out_dir'], f"{key}_{algo}.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(raw, f, indent=2, ensure_ascii=False)

    print(f"[done] {key} {algo} time={result['mean_solve_time']:.1f}s "
          f"cost={result['mean_cost']:.1f} tard={result['mean_tardiness']:.1f} "
          f"sols={result['mean_n_solutions']:.1f}", flush=True)
    return result


def plot_compare_pareto(results: Dict[str, Dict], save_path: str, title: str):
    plt.figure(figsize=(10, 8))
    for algo in ALGORITHMS:
        res = results.get(algo)
        if not res:
            continue
        pf = compute_pareto_front(list(zip(res['all_costs'], res['all_tardiness'])))
        if len(pf) == 0:
            continue
        style = ALGO_STYLES[algo]
        plt.scatter(pf[:, 0], pf[:, 1], facecolors='none', edgecolors=style['color'],
                    marker=style['marker'], s=55, label=style['label'])
        plt.plot(pf[:, 0], pf[:, 1], c=style['color'], linestyle='-',
                 linewidth=1.2, alpha=0.75)
    plt.xlabel('Travel Cost', fontsize=12)
    plt.ylabel('Tardiness Penalty', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  Pareto plot: {save_path}", flush=True)


def plot_routes(model, result: Dict, title: str, save_base: str):
    sols = result.get('all_solutions', [])
    fronts = result.get('all_pareto_fronts', [])
    if not sols or not fronts:
        return
    flat = [tuple(pt) for run in fronts for pt in run]
    if not flat:
        return

    pf_pts = compute_pareto_front(flat)
    pareto_indices = []
    pf_set = set((round(p[0], 6), round(p[1], 6)) for p in pf_pts.tolist())
    for i, pt in enumerate(flat):
        if (round(pt[0], 6), round(pt[1], 6)) in pf_set:
            pareto_indices.append(i)
    flat_pf = [flat[i] for i in pareto_indices]
    if not flat_pf:
        return

    local_idx = int(np.argmin([f[0] for f in flat_pf]))
    idx_cost = pareto_indices[local_idx]
    local_idx = int(np.argmin([f[1] for f in flat_pf]))
    idx_tard = pareto_indices[local_idx]

    pf_sorted = sorted(flat_pf, key=lambda x: x[0])
    c1, t1 = pf_sorted[0]
    c2, t2 = pf_sorted[-1]
    line_len = max(np.hypot(c2 - c1, t2 - t1), 1e-6)
    best_dist, knee_local = -1.0, 0
    for i, (c, t) in enumerate(pf_sorted):
        dist = abs((c2 - c1) * (t1 - t) - (c1 - c) * (t2 - t1)) / line_len
        if dist > best_dist:
            best_dist, knee_local = dist, i
    idx_comp = pareto_indices[knee_local]

    vis = Visualizer(model)
    for label, idx in [('min_cost', idx_cost), ('min_tardiness', idx_tard),
                       ('compromise', idx_comp)]:
        if idx < len(sols):
            save_path = save_base.replace('.png', f'_{label}.png')
            vis.plot_routes(sols[idx], title=f'{title} | {label}',
                            save_path=save_path, show_all_nodes=True)


def write_analysis(results_by_key: Dict, out_path: str, args) -> str:
    w_label = ALGO_STYLES[ALGORITHMS[-1]]['label']
    lines = [f"# 三种算法对比分析（NSGA-II / PACO-imp2 / {w_label}）", ""]
    lines.append("## 实验设置")
    lines.append("")
    lines.append(f"- 数据集：Solomon RC1/RC2（instance_id=1），配置与 `run_paco_alns_W6.py` 一致，共 16 组。")
    lines.append(f"- 规模：25c(2V)/50c(4V、6V)/100c(10V)，续航 medium/high。")
    lines.append(f"- 重复次数：每组 {args.runs} 次；迭代/世代预算：{args.max_iter}。")
    lines.append(f"- NSGA-II：pop_size=100；PACO-imp2：n_ants=30；{w_label}：自适应参数。")
    lines.append("- 指标口径：cost 来自 `model.evaluate_solution`；延迟为 `calculate_pure_tardiness`；")
    lines.append("  对缺客户解施加 10000/客户、对超载量施加 1000/单位惩罚后再计算前沿、HV 与均值。")
    lines.append("")

    lines.append("## 汇总表")
    lines.append("")
    header = ("| Config | Algo | Cost ± Std | Tardiness ± Std | HV | Time (s) | "
              "Sols | Drones | Routes | Miss | Overload |")
    lines.append(header)
    lines.append("|--------|------|------------|----------------|----|----------|"
                 "------|--------|--------|------|----------|")

    keys = sorted(results_by_key)
    cost_wins = {a: 0 for a in ALGORITHMS}
    tard_wins = {a: 0 for a in ALGORITHMS}
    hv_wins = {a: 0 for a in ALGORITHMS}
    rank_cost = {a: [] for a in ALGORITHMS}
    rank_tard = {a: [] for a in ALGORITHMS}
    rank_hv = {a: [] for a in ALGORITHMS}
    rel_cost = {a: [] for a in ALGORITHMS}
    rel_tard = {a: [] for a in ALGORITHMS}

    for key in keys:
        pair = results_by_key[key]
        for algo in ALGORITHMS:
            r = pair[algo]
            lines.append(
                f"| {key} | {ALGO_STYLES[algo]['label']} | "
                f"{r['mean_cost']:.2f} ± {r['std_cost']:.2f} | "
                f"{r['mean_tardiness']:.2f} ± {r['std_tardiness']:.2f} | "
                f"{r['mean_hv']:.2f} | {r['mean_solve_time']:.1f} | "
                f"{r['mean_n_solutions']:.1f} | {r['mean_drone_missions']:.1f} | "
                f"{r['mean_routes']:.1f} | {r['n_missing_solutions']} | "
                f"{r['n_overload_solutions']} |")

        best_cost = min(pair[a]['mean_cost'] for a in ALGORITHMS)
        best_tard = min(pair[a]['mean_tardiness'] for a in ALGORITHMS)
        best_hv = max(pair[a]['mean_hv'] for a in ALGORITHMS)
        for algo in ALGORITHMS:
            r = pair[algo]
            if r['mean_cost'] <= best_cost + 1e-9:
                cost_wins[algo] += 1
            if r['mean_tardiness'] <= best_tard + 1e-9:
                tard_wins[algo] += 1
            if r['mean_hv'] >= best_hv - 1e-9:
                hv_wins[algo] += 1
            rank_cost[algo].append(sorted(ALGORITHMS, key=lambda a: pair[a]['mean_cost']).index(algo) + 1)
            rank_tard[algo].append(sorted(ALGORITHMS, key=lambda a: pair[a]['mean_tardiness']).index(algo) + 1)
            rank_hv[algo].append(sorted(ALGORITHMS, key=lambda a: -pair[a]['mean_hv']).index(algo) + 1)
            rel_cost[algo].append((r['mean_cost'] / best_cost - 1.0) * 100.0 if best_cost > 0 else 0.0)
            rel_tard[algo].append((r['mean_tardiness'] / best_tard - 1.0) * 100.0 if best_tard > 0 else 0.0)

    lines.append("")
    lines.append("## 统计排名")
    lines.append("")
    lines.append("| Algo | Cost 最优次数 | Tardiness 最优次数 | HV 最优次数 | "
                 "Cost 平均排名 | Tard 平均排名 | HV 平均排名 | Cost 相对最优平均增幅 | Tard 相对最优平均增幅 |")
    lines.append("|------|--------------|-------------------|-------------|----------------|----------------|-------------|------------------------|------------------------|")
    for algo in ALGORITHMS:
        lines.append(
            f"| {ALGO_STYLES[algo]['label']} | {cost_wins[algo]} | {tard_wins[algo]} | "
            f"{hv_wins[algo]} | {np.mean(rank_cost[algo]):.2f} | "
            f"{np.mean(rank_tard[algo]):.2f} | {np.mean(rank_hv[algo]):.2f} | "
            f"{np.mean(rel_cost[algo]):.2f}% | {np.mean(rel_tard[algo]):.2f}% |")

    lines.append("")
    lines.append("## 结论")
    lines.append("")
    best_algo_cost = min(ALGORITHMS, key=lambda a: np.mean(rank_cost[a]))
    best_algo_tard = min(ALGORITHMS, key=lambda a: np.mean(rank_tard[a]))
    best_algo_hv = min(ALGORITHMS, key=lambda a: np.mean(rank_hv[a]))
    lines.append(f"- 成本维度：{ALGO_STYLES[best_algo_cost]['label']} 平均排名最优（{np.mean(rank_cost[best_algo_cost]):.2f}）。")
    lines.append(f"- 延迟维度：{ALGO_STYLES[best_algo_tard]['label']} 平均排名最优（{np.mean(rank_tard[best_algo_tard]):.2f}）。")
    lines.append(f"- 超体积维度：{ALGO_STYLES[best_algo_hv]['label']} 平均排名最优（{np.mean(rank_hv[best_algo_hv]):.2f}）。")
    lines.append("- 详细数据见同目录 JSON 文件（含每轮前沿、每解成本/延迟、HV 参考点等）。")

    text = "\n".join(lines) + "\n"
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(text)
    return text


def main():
    parser = argparse.ArgumentParser(description='NSGA-II vs PACO-imp2 vs PACO+ALNS W7')
    parser.add_argument('--n_customers', type=int, default=25)
    parser.add_argument('--rc_type', choices=['RC1', 'RC2'], default='RC1')
    parser.add_argument('--instance_id', type=int, default=1)
    parser.add_argument('--n_vehicles', type=int, default=2)
    parser.add_argument('--endurance', choices=['medium', 'high'], default='medium')
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--max-iter', type=int, default=100)
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--size', choices=['small', 'medium', 'large', 'all'], default='all')
    parser.add_argument('--outdir', type=str, default='20260805')
    parser.add_argument('--workers', type=int, default=min(16, os.cpu_count() or 4))
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--no-route-plots', action='store_true')
    parser.add_argument('--w7-module', choices=['w7', 'w8'], default='w8',
                        help='which PACO+ALNS variant to use for the w7 slot')
    parser.add_argument('--all-solomon', action='store_true',
                        help='run every standard Solomon instance except tested RC101/RC201')
    parser.add_argument('--limit', type=int, default=0,
                        help='only run the first N configs (smoke test)')
    args = parser.parse_args()

    w_slot = 'w8' if args.w7_module == 'w8' else 'w7'
    ALGORITHMS[2] = w_slot
    style = ALGO_STYLES.pop('w7', {'label': 'PACO+ALNS',
                                   'color': '#d62728', 'marker': '^'})
    ALGO_STYLES[w_slot] = style
    ALGO_STYLES[w_slot]['label'] = ('PACO+ALNS W8' if w_slot == 'w8'
                                    else 'PACO+ALNS W7')

    results_dir = os.path.join(BASE, 'results', args.outdir)
    os.makedirs(results_dir, exist_ok=True)
    date_tag = args.outdir

    if args.all_solomon:
        configs = all_solomon_configs()
        if args.limit > 0:
            configs = configs[:args.limit]
    elif args.full:
        configs = [dict(c) for c in ALL_CONFIGS]
        if args.size != 'all':
            configs = [c for c in configs if c['size'] == args.size]
    else:
        configs = [{'n_customers': args.n_customers, 'rc_type': args.rc_type,
                    'instance_id': args.instance_id, 'n_vehicles': args.n_vehicles,
                    'endurance_type': args.endurance, 'size': 'custom'}]

    tasks = []
    for idx, cfg in enumerate(configs):
        key = config_key(cfg)
        for algo in ALGORITHMS:
            out_path = os.path.join(results_dir, f"{key}_{algo}.json")
            if os.path.exists(out_path) and not args.force:
                print(f"[skip] {key} {algo} (already exists)", flush=True)
                continue
            tasks.append({
                'config': cfg, 'algo': algo, 'runs': args.runs,
                'max_iter': args.max_iter,
                'seed': 20260805 + idx * 1000 + ALGORITHMS.index(algo),
                'out_dir': results_dir,
                'route_plots': not args.no_route_plots,
                'w7_module': args.w7_module,
            })

    if tasks:
        with Pool(args.workers) as pool:
            pool.map(run_config, tasks)

    results_by_key = {}
    for cfg in configs:
        key = config_key(cfg)
        results_by_key[key] = {}
        for algo in ALGORITHMS:
            path = os.path.join(results_dir, f"{key}_{algo}.json")
            if not os.path.exists(path):
                print(f"[warn] missing {key} {algo}", flush=True)
                continue
            with open(path, encoding='utf-8') as f:
                results_by_key[key][algo] = json.load(f)

    # Shared HV reference per config, then finalize every algorithm result.
    for key, pair in results_by_key.items():
        all_costs, all_tards = [], []
        for algo in ALGORITHMS:
            r = pair.get(algo)
            if r:
                all_costs.extend(r['all_costs'])
                all_tards.extend(r['all_tardiness'])
        if not all_costs or not all_tards:
            continue
        ref_cost = float(np.max(all_costs) * 1.1)
        ref_tard = float(np.max(all_tards) * 1.1)
        for algo in ALGORITHMS:
            if algo not in pair:
                continue
            pair[algo] = finalize_result(pair[algo], ref_cost, ref_tard)
            pair[algo].update({
                'config': key,
                'algo': algo,
                'n_customers': next(c['n_customers'] for c in configs if config_key(c) == key),
                'rc_type': next(c['rc_type'] for c in configs if config_key(c) == key),
                'instance_id': next(c['instance_id'] for c in configs if config_key(c) == key),
                'n_vehicles': next(c['n_vehicles'] for c in configs if config_key(c) == key),
                'endurance_type': next(c['endurance_type'] for c in configs if config_key(c) == key),
                'hv_reference': [ref_cost, ref_tard],
            })
            with open(os.path.join(results_dir, f"{key}_{algo}.json"), 'w', encoding='utf-8') as f:
                json.dump(pair[algo], f, indent=2, ensure_ascii=False)

        title = f"NSGA-II vs PACO-imp2 vs {ALGO_STYLES[ALGORITHMS[-1]]['label']} | {key}"
        plot_compare_pareto(pair, os.path.join(results_dir, f"pareto_compare_{key}.png"), title)

    combined_path = os.path.join(results_dir, f'compare_three_{date_tag}.json')
    with open(combined_path, 'w', encoding='utf-8') as f:
        json.dump(results_by_key, f, indent=2, ensure_ascii=False)
    print(f"Saved: {combined_path}", flush=True)

    md_path = os.path.join(results_dir, f'analysis_{date_tag}.md')
    write_analysis(results_by_key, md_path, args)
    print(f"Saved: {md_path}", flush=True)


if __name__ == '__main__':
    main()
