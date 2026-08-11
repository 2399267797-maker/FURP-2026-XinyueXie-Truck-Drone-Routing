# -*- coding: utf-8 -*-
"""W7 vs PACO-imp2 完整对比实验脚本（16 配置，逻辑对齐 run_paco_alns_W7.py）。

用法：
    python compare_run_w7_imp2.py --full                          # 16 配置，默认 runs=10, max_iter=100
    python compare_run_w7_imp2.py --full --size small --runs 2    # 快速验证
    python compare_run_w7_imp2.py --n_customers 25 --runs 3       # 单配置

输出：
    results/<outdir>/<key>_imp2.json / <key>_w7.json    每个配置单独结果
    results/<outdir>/compare_w7_imp2.json                合并 JSON
    results/<outdir>/compare_w7_imp2.md                  汇总 Markdown
    results/<outdir>/pareto_compare_<key>.png            双算法帕累托对比图

已完成的任务文件默认跳过，可直接断点续跑。
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

sys.path.insert(0, os.path.join(PROJ, 'src', 'experiments', 'PACO', 'data'))
sys.path.insert(0, os.path.join(PROJ, 'src', 'experiments', 'PACO_vs_NSGA2'))

from models.vrp_model import VRPTruckDroneModel  # noqa: E402
from utils.visualizer import Visualizer  # noqa: E402
from solomon_loader_imp import SolomonLoaderImp  # noqa: E402


ALL_CONFIGS = [
    {'n_customers': 25, 'rc_type': 'RC1', 'n_vehicles': 2, 'endurance_type': 'medium', 'size': 'small'},
    {'n_customers': 25, 'rc_type': 'RC1', 'n_vehicles': 2, 'endurance_type': 'high',   'size': 'small'},
    {'n_customers': 25, 'rc_type': 'RC2', 'n_vehicles': 2, 'endurance_type': 'medium', 'size': 'small'},
    {'n_customers': 25, 'rc_type': 'RC2', 'n_vehicles': 2, 'endurance_type': 'high',   'size': 'small'},
    {'n_customers': 50, 'rc_type': 'RC1', 'n_vehicles': 4, 'endurance_type': 'medium', 'size': 'medium'},
    {'n_customers': 50, 'rc_type': 'RC1', 'n_vehicles': 4, 'endurance_type': 'high',   'size': 'medium'},
    {'n_customers': 50, 'rc_type': 'RC2', 'n_vehicles': 4, 'endurance_type': 'medium', 'size': 'medium'},
    {'n_customers': 50, 'rc_type': 'RC2', 'n_vehicles': 4, 'endurance_type': 'high',   'size': 'medium'},
    {'n_customers': 50, 'rc_type': 'RC1', 'n_vehicles': 6, 'endurance_type': 'medium', 'size': 'medium'},
    {'n_customers': 50, 'rc_type': 'RC1', 'n_vehicles': 6, 'endurance_type': 'high',   'size': 'medium'},
    {'n_customers': 50, 'rc_type': 'RC2', 'n_vehicles': 6, 'endurance_type': 'medium', 'size': 'medium'},
    {'n_customers': 50, 'rc_type': 'RC2', 'n_vehicles': 6, 'endurance_type': 'high',   'size': 'medium'},
    {'n_customers': 100, 'rc_type': 'RC1', 'n_vehicles': 10, 'endurance_type': 'medium', 'size': 'large'},
    {'n_customers': 100, 'rc_type': 'RC1', 'n_vehicles': 10, 'endurance_type': 'high',   'size': 'large'},
    {'n_customers': 100, 'rc_type': 'RC2', 'n_vehicles': 10, 'endurance_type': 'medium', 'size': 'large'},
    {'n_customers': 100, 'rc_type': 'RC2', 'n_vehicles': 10, 'endurance_type': 'high',   'size': 'large'},
]


def load_algorithm(algo_name: str):
    """按名称加载 W7 或 imp2 模块。"""
    if algo_name == 'w7':
        path = os.path.join(BASE, 'PACO+ALNSW7.py')
    else:
        path = os.path.join(PROJ, 'src', 'experiments', 'PACO', 'algorithms', 'paco_imp2.py')
    spec = importlib.util.spec_from_file_location('paco_' + algo_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_model(n_customers: int, rc_type: str, instance_id: int,
               n_vehicles: int, endurance_type: str, use_drones: bool = True) -> VRPTruckDroneModel:
    loader = SolomonLoaderImp()
    kwargs = dict(n_customers=n_customers, instance_id=instance_id,
                  n_vehicles=n_vehicles, endurance_type=endurance_type,
                  use_drones=use_drones)
    if rc_type == 'RC1':
        return loader.load_rc1_instance(**kwargs)
    return loader.load_rc2_instance(**kwargs)


def calculate_hypervolume(pareto_front, ref_cost=None, ref_tard=None):
    if not pareto_front:
        return 0.0
    pts = np.array(pareto_front)
    if ref_cost is None:
        ref_cost = float(np.max(pts[:, 0]) * 1.1)
    if ref_tard is None:
        ref_tard = float(np.max(pts[:, 1]) * 1.1)
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
        if dominated[i]:
            continue
        for j in range(len(pts)):
            if i == j or dominated[j]:
                continue
            if (pts[i, 0] <= pts[j, 0] and pts[i, 1] <= pts[j, 1] and
                    (pts[i, 0] < pts[j, 0] or pts[i, 1] < pts[j, 1])):
                dominated[j] = True
    pf = pts[~dominated]
    pf = pf[np.argsort(pf[:, 0])]
    return pf[:, 0], pf[:, 1]


def run_single(model, algo_mod, algo_name: str, max_iter: int, seed: int) -> Dict:
    random.seed(seed)
    np.random.seed(seed)

    if algo_name == 'w7':
        algo = algo_mod.CollaborativePACOALNS(model, max_iter=max_iter)
    else:
        algo = algo_mod.CollaborativePACO(model, max_iter=max_iter)

    t0 = time.time()
    solutions, _ = algo.solve()
    elapsed = time.time() - t0

    n_customers = model.get_number_of_customers()
    costs, tardiness, pf = [], [], []
    missing_counts, overload_amounts, drone_counts, route_counts = [], [], [], []
    for sol in solutions:
        c, _ = model.evaluate_solution(sol)
        t = model.calculate_pure_tardiness(sol)
        served = set()
        for r in sol:
            served.update(r.customers)
            for m in r.drone_missions:
                served.update(m.customer_ids)
        missing = n_customers - len(served)
        overload = sum(max(0.0, sum(model.customers[cc].demand for cc in r.customers) - model.trucks[0].capacity)
                       for r in sol)
        if missing > 0:
            c += missing * 10000.0
            t += missing * 10000.0
        costs.append(c)
        tardiness.append(t)
        pf.append((c, t))
        missing_counts.append(missing)
        overload_amounts.append(overload)
        drone_counts.append(sum(len(r.drone_missions) for r in sol))
        route_counts.append(sum(1 for r in sol if r.customers or r.drone_missions))

    return {
        'solutions': solutions,
        'pareto_front': pf,
        'costs': costs,
        'tardiness': tardiness,
        'solve_time': elapsed,
        'hypervolume': calculate_hypervolume(pf),
        'n_solutions': len(solutions),
        'mean_drone_missions': float(np.mean(drone_counts)) if drone_counts else 0.0,
        'mean_routes': float(np.mean(route_counts)) if route_counts else 0.0,
        'n_missing_solutions': int(sum(1 for x in missing_counts if x > 0)),
        'max_missing': int(max(missing_counts)) if missing_counts else 0,
        'n_overload_solutions': int(sum(1 for x in overload_amounts if x > 1e-6)),
        'max_overload': float(max(overload_amounts)) if overload_amounts else 0.0,
    }


def run_experiment(model, algo_mod, algo_name: str, n_runs: int, max_iter: int) -> Dict:
    keys = ['costs', 'tardiness', 'solve_times', 'hypervolumes',
            'solutions', 'pareto_fronts']
    acc = {k: [] for k in keys}
    n_sol_list, drone_means, route_means = [], [], []
    missing_runs, overload_runs, max_missing, max_overload = 0, 0, 0, 0.0

    for run_idx in range(n_runs):
        res = run_single(model, algo_mod, algo_name, max_iter, seed=1000 + run_idx)
        acc['costs'].extend(res['costs'])
        acc['tardiness'].extend(res['tardiness'])
        acc['solve_times'].append(res['solve_time'])
        acc['hypervolumes'].append(res['hypervolume'])
        acc['solutions'].extend(res['solutions'])
        acc['pareto_fronts'].append(res['pareto_front'])
        n_sol_list.append(res['n_solutions'])
        drone_means.append(res['mean_drone_missions'])
        route_means.append(res['mean_routes'])
        missing_runs += res['n_missing_solutions']
        overload_runs += res['n_overload_solutions']
        max_missing = max(max_missing, res['max_missing'])
        max_overload = max(max_overload, res['max_overload'])

    return {
        'n_runs': n_runs,
        'mean_cost': float(np.mean(acc['costs'])),
        'std_cost': float(np.std(acc['costs'])),
        'mean_tardiness': float(np.mean(acc['tardiness'])),
        'std_tardiness': float(np.std(acc['tardiness'])),
        'mean_solve_time': float(np.mean(acc['solve_times'])),
        'mean_hv': float(np.mean(acc['hypervolumes'])),
        'std_hv': float(np.std(acc['hypervolumes'])),
        'mean_n_solutions': float(np.mean(n_sol_list)),
        'mean_drone_missions': float(np.mean(drone_means)),
        'mean_routes': float(np.mean(route_means)),
        'n_missing_solutions': int(missing_runs),
        'n_overload_solutions': int(overload_runs),
        'max_missing': int(max_missing),
        'max_overload': float(max_overload),
        'all_costs': acc['costs'],
        'all_tardiness': acc['tardiness'],
        'all_pareto_fronts': acc['pareto_fronts'],
        'all_solutions': acc['solutions'],
    }


def _clean_result(result: Dict) -> Dict:
    out = dict(result)
    out['all_pareto_fronts'] = [[list(pt) for pt in front] for front in result['all_pareto_fronts']]
    out.pop('solutions', None)
    out.pop('all_solutions', None)
    return out


def config_key(cfg: Dict) -> str:
    return f"{cfg['n_customers']}c_{cfg['rc_type']}{cfg['instance_id']:02d}_{cfg['n_vehicles']}V_{cfg['endurance_type']}"


def run_config(task: Dict) -> Dict:
    cfg, algo_name = task['config'], task['algo']
    key = config_key(cfg)
    print(f"[start] {key} {algo_name} runs={task['runs']} max_iter={task['max_iter']}", flush=True)

    model = load_model(cfg['n_customers'], cfg['rc_type'], cfg['instance_id'],
                       cfg['n_vehicles'], cfg['endurance_type'], use_drones=True)
    algo_mod = load_algorithm(algo_name)
    result = run_experiment(model, algo_mod, algo_name, task['runs'], task['max_iter'])

    result.update({
        'n_customers': cfg['n_customers'],
        'rc_type': cfg['rc_type'],
        'instance_id': cfg['instance_id'],
        'n_vehicles': cfg['n_vehicles'],
        'endurance_type': cfg['endurance_type'],
        'algo': algo_name,
        'key': key,
    })

    out_path = os.path.join(task['out_dir'], f"{key}_{algo_name}.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(_clean_result(result), f, indent=2, ensure_ascii=False)

    if task.get('route_plots', True):
        title = f"{algo_name.upper()} | {cfg['n_customers']}C | {cfg['n_vehicles']}T+{cfg['n_vehicles']}D | {cfg['endurance_type']}"
        plot_routes(model, result, title,
                    os.path.join(task['out_dir'], f"routes_{algo_name}_{key}.png"))

    print(f"[done] {key} {algo_name} time={result['mean_solve_time']:.1f}s "
          f"cost={result['mean_cost']:.1f} tard={result['mean_tardiness']:.1f}", flush=True)
    return _clean_result(result)


def plot_compare_pareto(res_imp2: Dict, res_w7: Dict, save_path: str):
    plt.figure(figsize=(10, 8))
    for res, color, marker, label in [
        (res_imp2, '#2ca02c', 'o', 'PACO-imp2'),
        (res_w7, '#d62728', '^', 'PACO+ALNS W7'),
    ]:
        c, t = _compute_pareto_front(res['all_costs'], res['all_tardiness'])
        if len(c):
            plt.scatter(c, t, facecolors='none', edgecolors=color, marker=marker,
                        s=60, label=label)
            plt.plot(c, t, c=color, linestyle='-', linewidth=1.2, alpha=0.7)
    plt.xlabel('Cost of Travel', fontsize=12)
    plt.ylabel('Penalty due to Tardiness', fontsize=12)
    plt.title(f"W7 vs imp2 | {res_w7['key']}", fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_routes(model, result: Dict, title: str, save_base: str):
    """绘制三个代表解路线图：最小成本、最小延迟、折中解（对齐 run_paco_alns_W7.py）。"""
    if not result.get('all_solutions'):
        return
    pf = result.get('all_pareto_fronts', [])
    sols = result['all_solutions']
    if not pf or not sols:
        return
    flat = [pt for run in pf for pt in run]
    if not flat:
        return

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

    local_idx = np.argmin([f[0] for f in flat_pf])
    idx_cost = int(pareto_indices[local_idx])
    local_idx = np.argmin([f[1] for f in flat_pf])
    idx_tard = int(pareto_indices[local_idx])
    pf_sorted = sorted(flat_pf, key=lambda x: x[0])
    c1, t1 = pf_sorted[0]
    c2, t2 = pf_sorted[-1]
    line_len = max(np.sqrt((c2 - c1) ** 2 + (t2 - t1) ** 2), 1e-6)
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
            save_path = save_base.replace('.png', f'_{label}.png')
            vis.plot_routes(sols[idx], title=f'{title} | {label}',
                            save_path=save_path, show_all_nodes=True)
            print(f"  Route plot ({label}): {save_path}", flush=True)


def write_summary(results_by_key: Dict, out_path: str):
    lines = ["# W7 vs PACO-imp2 Full Comparison", ""]
    lines.append("**Configs**: 16 | **Runs**: %d | **Max iter**: %d" %
                 (results_by_key[list(results_by_key)[0]]['imp2']['n_runs'],
                  results_by_key[list(results_by_key)[0]]['imp2'].get('max_iter', '?')))
    lines.append("")
    header = ("| Config | Algo | Cost ± Std | Tard ± Std | HV | Time (s) | Sols | Drones | Routes | Miss | Overload |")
    lines.append(header)
    lines.append("|--------|------|------------|------------|----|----------|------|--------|--------|------|----------|")

    for key in sorted(results_by_key):
        pair = results_by_key[key]
        for algo in ['imp2', 'w7']:
            r = pair[algo]
            lines.append(
                f"| {key} | {algo} | {r['mean_cost']:.2f} ± {r['std_cost']:.2f} | "
                f"{r['mean_tardiness']:.2f} ± {r['std_tardiness']:.2f} | {r['mean_hv']:.2f} | "
                f"{r['mean_solve_time']:.1f} | {r['mean_n_solutions']:.1f} | "
                f"{r['mean_drone_missions']:.1f} | {r['mean_routes']:.1f} | "
                f"{r['n_missing_solutions']} | {r['n_overload_solutions']} |")

    lines.append("")
    lines.append("## Per-config deltas (W7 vs imp2)")
    lines.append("")
    lines.append("| Config | dCost % | dTard % | Cost winner | Tard winner |")
    lines.append("|--------|---------|---------|-------------|-------------|")
    cost_wins = tard_wins = 0
    for key in sorted(results_by_key):
        imp2, w7 = results_by_key[key]['imp2'], results_by_key[key]['w7']
        dcost = (w7['mean_cost'] - imp2['mean_cost']) / max(1e-9, imp2['mean_cost']) * 100.0
        dtard = (w7['mean_tardiness'] - imp2['mean_tardiness']) / max(1e-9, imp2['mean_tardiness']) * 100.0
        cw = 'W7' if dcost < -1e-6 else ('imp2' if dcost > 1e-6 else 'tie')
        tw = 'W7' if dtard < -1e-6 else ('imp2' if dtard > 1e-6 else 'tie')
        if cw == 'W7': cost_wins += 1
        if tw == 'W7': tard_wins += 1
        lines.append(f"| {key} | {dcost:.1f} | {dtard:.1f} | {cw} | {tw} |")

    lines.append("")
    lines.append(f"**Summary**: W7 wins cost on {cost_wins}/16, tardiness on {tard_wins}/16.")
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description='W7 vs PACO-imp2 完整对比实验')
    parser.add_argument('--n_customers', type=int, default=25)
    parser.add_argument('--rc_type', choices=['RC1', 'RC2'], default='RC1')
    parser.add_argument('--instance_id', type=int, default=1)
    parser.add_argument('--n_vehicles', type=int, default=2)
    parser.add_argument('--endurance', choices=['medium', 'high'], default='medium')
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--max-iter', type=int, default=100)
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--size', choices=['small', 'medium', 'large', 'all'], default='all')
    parser.add_argument('--outdir', type=str, default='compare_w7_imp2_full')
    parser.add_argument('--workers', type=int, default=max(4, min(16, os.cpu_count() - 2)))
    parser.add_argument('--force', action='store_true', help='忽略已有结果重新运行')
    parser.add_argument('--no-plots', action='store_true')
    parser.add_argument('--no-route-plots', action='store_true')
    args = parser.parse_args()

    results_dir = os.path.join(BASE, 'results', args.outdir)
    os.makedirs(results_dir, exist_ok=True)

    if args.full:
        configs = [dict(c, instance_id=1) for c in ALL_CONFIGS]
        if args.size != 'all':
            configs = [c for c in configs if c['size'] == args.size]
    else:
        configs = [{'n_customers': args.n_customers, 'rc_type': args.rc_type,
                    'instance_id': args.instance_id, 'n_vehicles': args.n_vehicles,
                    'endurance_type': args.endurance, 'size': 'custom'}]

    tasks = []
    for cfg in configs:
        for algo in ['imp2', 'w7']:
            key = config_key(cfg)
            out_path = os.path.join(results_dir, f"{key}_{algo}.json")
            if os.path.exists(out_path) and not args.force:
                print(f"[skip] {key} {algo} (already exists)", flush=True)
                continue
            tasks.append({'config': cfg, 'algo': algo, 'runs': args.runs,
                          'max_iter': args.max_iter, 'out_dir': results_dir,
                          'route_plots': not args.no_route_plots})

    if tasks:
        with Pool(args.workers) as pool:
            pool.map(run_config, tasks)

    # 汇总
    results_by_key = {}
    for cfg in configs:
        key = config_key(cfg)
        imp2_path = os.path.join(results_dir, f"{key}_imp2.json")
        w7_path = os.path.join(results_dir, f"{key}_w7.json")
        if not (os.path.exists(imp2_path) and os.path.exists(w7_path)):
            print(f"[warn] missing result for {key}, skip aggregation", flush=True)
            continue
        with open(imp2_path, encoding='utf-8') as f:
            imp2 = json.load(f)
        with open(w7_path, encoding='utf-8') as f:
            w7 = json.load(f)
        imp2['max_iter'] = args.max_iter
        w7['max_iter'] = args.max_iter
        results_by_key[key] = {'imp2': imp2, 'w7': w7}

        if not args.no_plots:
            plot_compare_pareto(imp2, w7, os.path.join(results_dir, f'pareto_compare_{key}.png'))

    combined_path = os.path.join(results_dir, 'compare_w7_imp2.json')
    with open(combined_path, 'w', encoding='utf-8') as f:
        json.dump(results_by_key, f, indent=2, ensure_ascii=False)
    print(f"Saved: {combined_path}")

    md_path = os.path.join(results_dir, 'compare_w7_imp2.md')
    write_summary(results_by_key, md_path)
    print(f"Saved: {md_path}")


if __name__ == '__main__':
    main()
