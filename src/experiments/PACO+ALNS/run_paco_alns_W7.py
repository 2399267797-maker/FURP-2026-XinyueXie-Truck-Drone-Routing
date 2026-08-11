"""
PACO+ALNS W7 — 实验运行脚本.

基于 PACO+ALNSW7.py，支持单配置运行和全配置批量实验。
生成帕累托图、路线图、JSON 摘要。

Usage:
    python run_paco_alns_W7.py                               # 默认：25c RC1 medium 2V 单配置
    python run_paco_alns_W7.py --n_customers 50 --rc_type RC1 --n_vehicles 4 --endurance high
    python run_paco_alns_W7.py --full                         # 跑全部 16 个配置
    python run_paco_alns_W7.py --full --size small            # 仅 25c 配置
    python run_paco_alns_W7.py --runs 3                       # 快速测试 3 轮
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

# ── 路径设置 ─────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(os.path.dirname(BASE)))

sys.path.insert(0, os.path.join(PROJ, 'src', 'experiments', 'PACO', 'data'))
sys.path.insert(0, os.path.join(PROJ, 'src', 'experiments', 'PACO_vs_NSGA2'))

from models.vrp_model import VRPTruckDroneModel, Route
from utils.visualizer import Visualizer

# ── 加载 PACO+ALNS W7 ───────────────────────────────────────────────────────
spec = importlib.util.spec_from_file_location("paco_alns_w7",
                                              os.path.join(BASE, 'PACO+ALNSW7.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
CollaborativePACOALNS = mod.CollaborativePACOALNS

# ── 加载 Solomon loader ─────────────────────────────────────────────────────
from solomon_loader_imp import SolomonLoaderImp


# ═════════════════════════════════════════════════════════════════════════════
#  核心辅助函数
# ═════════════════════════════════════════════════════════════════════════════

def load_model(n_customers: int, rc_type: str = 'RC1', instance_id: int = 1,
               n_vehicles: int = 2, endurance_type: str = 'medium',
               use_drones: bool = True) -> VRPTruckDroneModel:
    """加载 Solomon RC 实例."""
    loader = SolomonLoaderImp()
    kwargs = dict(n_customers=n_customers, instance_id=instance_id,
                  n_vehicles=n_vehicles, endurance_type=endurance_type,
                  use_drones=use_drones)
    if rc_type == 'RC1':
        return loader.load_rc1_instance(**kwargs)
    else:
        return loader.load_rc2_instance(**kwargs)


def calculate_hypervolume(pareto_front, ref_cost=None, ref_tard=None):
    """计算超体积（HV）。默认使用本前沿自适应参考点 max*1.1."""
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
    """计算联合非支配前沿."""
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


# ═════════════════════════════════════════════════════════════════════════════
#  单次运行 / 多轮实验
# ═════════════════════════════════════════════════════════════════════════════

def run_alns_single(model: VRPTruckDroneModel, max_iter: int = 100) -> Dict:
    """单次 W7 运行."""
    algo = CollaborativePACOALNS(model, max_iter=max_iter)
    start = time.time()
    solutions, _ = algo.solve()
    elapsed = time.time() - start

    n_customers = model.get_number_of_customers()
    costs, tardiness, pf = [], [], []
    missing_counts, overload_amounts, drone_counts, route_counts = [], [], [], []
    for sol in solutions:
        c, _ = model.evaluate_solution(sol)
        t = model.calculate_pure_tardiness(sol)
        # 检查缺失客户，施加双重惩罚
        served = set()
        for r in sol:
            served.update(r.customers)
            for m in r.drone_missions:
                served.update(m.customer_ids)
        missing = n_customers - len(served)
        overload = sum(max(0.0, sum(model.customers[cc].demand for cc in r.customers) - model.trucks[0].capacity) for r in sol)
        drone_n = sum(len(r.drone_missions) for r in sol)
        route_n = sum(1 for r in sol if r.customers or r.drone_missions)
        if missing > 0:
            c += missing * 10000.0
            t += missing * 10000.0
        costs.append(c)
        tardiness.append(t)
        pf.append((c, t))
        missing_counts.append(missing)
        overload_amounts.append(overload)
        drone_counts.append(drone_n)
        route_counts.append(route_n)

    hv = calculate_hypervolume(pf)
    return {'solutions': solutions, 'pareto_front': pf,
            'costs': costs, 'tardiness': tardiness,
            'solve_time': elapsed, 'hypervolume': hv,
            'n_solutions': len(solutions),
            'mean_drone_missions': float(np.mean(drone_counts)) if drone_counts else 0.0,
            'mean_routes': float(np.mean(route_counts)) if route_counts else 0.0,
            'n_missing_solutions': int(sum(1 for m in missing_counts if m > 0)),
            'max_missing': int(max(missing_counts)) if missing_counts else 0,
            'n_overload_solutions': int(sum(1 for o in overload_amounts if o > 1e-6)),
            'max_overload': float(max(overload_amounts)) if overload_amounts else 0.0}


def run_experiment(model: VRPTruckDroneModel, n_runs: int = 10, max_iter: int = 100) -> Dict:
    """多轮重复实验."""
    print("\n--- PACO+ALNS W7 ---")
    keys = ['costs', 'tardiness', 'solve_times', 'hypervolumes',
            'solutions', 'pareto_fronts']
    acc = {k: [] for k in keys}
    n_sol_list, drone_means, route_means = [], [], []
    missing_runs, overload_runs, max_missing, max_overload = 0, 0, 0, 0.0

    for run_idx in range(n_runs):
        print(f"  Run {run_idx + 1}/{n_runs} ...", end='', flush=True)
        res = run_alns_single(model, max_iter=max_iter)
        print(f" done  ({res['solve_time']:.1f}s)")
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

    mean_hv = np.mean(acc['hypervolumes'])
    std_hv = np.std(acc['hypervolumes'])

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
        'mean_n_solutions': float(np.mean(n_sol_list)),
        'mean_drone_missions': float(np.mean(drone_means)),
        'mean_routes': float(np.mean(route_means)),
        'n_missing_solutions': int(missing_runs),
        'n_overload_solutions': int(overload_runs),
        'max_missing': int(max_missing),
        'max_overload': float(max_overload),
        'all_costs': acc['costs'],
        'all_tardiness': acc['tardiness'],
        'all_solutions': acc['solutions'],
        'all_pareto_fronts': acc['pareto_fronts'],
    }


# ═════════════════════════════════════════════════════════════════════════════
#  绘图
# ═════════════════════════════════════════════════════════════════════════════

def plot_pareto(result: Dict, save_path: str):
    """绘制帕累托前沿."""
    costs = result.get('all_costs', [])
    tardiness = result.get('all_tardiness', [])
    if not costs or not tardiness:
        return
    pf_c, pf_t = _compute_pareto_front(costs, tardiness)

    plt.figure(figsize=(10, 8))
    plt.scatter(pf_c, pf_t, facecolors='none', edgecolors='#d62728',
                marker='^', s=60, label='PACO+ALNS W7')
    plt.plot(pf_c, pf_t, c='#d62728', linestyle='-', linewidth=1.5, alpha=0.7)

    plt.xlabel('Cost of Travel', fontsize=12)
    plt.ylabel('Penalty due to Tardiness', fontsize=12)
    plt.title('PACO+ALNS W7 — Non-dominated Front', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  Pareto plot: {save_path}")


def plot_routes(model: VRPTruckDroneModel, result: Dict, title: str, save_base: str):
    """绘制三条路线图：最小成本、最小延迟、折中解."""
    if not result['all_solutions']:
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
            route = sols[idx]
            save_path = save_base.replace('.png', f'_{label}.png')
            t = f'{title} | {label}'
            vis.plot_routes(route, title=t, save_path=save_path, show_all_nodes=True)
            print(f"  Route plot ({label}): {save_path}")


# ═════════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='PACO+ALNS W7 实验运行脚本')
    parser.add_argument('--n_customers', type=int, default=25, help='客户数量')
    parser.add_argument('--rc_type', choices=['RC1', 'RC2'], default='RC1', help='RC 类型')
    parser.add_argument('--instance_id', type=int, default=1, help='实例编号')
    parser.add_argument('--n_vehicles', type=int, default=2, help='卡车数量')
    parser.add_argument('--endurance', choices=['medium', 'high'], default='medium', help='无人机续航')
    parser.add_argument('--runs', type=int, default=10, help='重复次数')
    parser.add_argument('--max_iter', type=int, default=100, help='最大迭代次数')
    parser.add_argument('--full', action='store_true', help='运行全部 16 个配置')
    parser.add_argument('--size', choices=['small', 'medium', 'large', 'all'],
                        default='all', help='--full 时筛选客户规模')
    parser.add_argument('--outdir', type=str, default='20260802_w7_fixed',
                        help='结果子目录名（默认 20260802_w7_fixed）')
    args = parser.parse_args()

    # 结果目录
    results_dir = os.path.join(BASE, 'results', args.outdir)
    os.makedirs(results_dir, exist_ok=True)

    # ── 配置列表 ─────────────────────────────────────────────────────────────
    all_configs = [
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

    if args.full:
        configs = all_configs
        if args.size != 'all':
            configs = [c for c in configs if c['size'] == args.size]
    else:
        # 单配置
        configs = [{
            'n_customers': args.n_customers,
            'rc_type': args.rc_type,
            'n_vehicles': args.n_vehicles,
            'endurance_type': args.endurance,
            'size': 'custom',
        }]

    all_results = []

    for cfg in configs:
        n, rc, inst, nv, end = (
            cfg['n_customers'], cfg['rc_type'], args.instance_id if not args.full else 1,
            cfg['n_vehicles'], cfg['endurance_type']
        )
        print(f"\n{'=' * 60}")
        print(f"PACO+ALNS W7 | {n}c | {rc} | {nv}V | {end}")
        print(f"{'=' * 60}")

        model = load_model(n, rc, inst, nv, end, use_drones=True)
        print(f"  Trucks={model.get_number_of_trucks()}, "
              f"Drones={model.get_number_of_drones()}, "
              f"Range={model.drone_range}km")

        result = run_experiment(model, n_runs=args.runs, max_iter=args.max_iter)

        result_dict = {
            'n_customers': n,
            'rc_type': rc,
            'instance_id': inst,
            'n_vehicles': nv,
            'endurance_type': end,
            'n_runs': args.runs,
            'algo': 'PACO+ALNS_W7',
            'mean_cost': result['mean_cost'],
            'std_cost': result['std_cost'],
            'mean_tardiness': result['mean_tardiness'],
            'std_tardiness': result['std_tardiness'],
            'mean_hv': result['mean_hv'],
            'std_hv': result['std_hv'],
            'mean_solve_time': result['mean_solve_time'],
            'mean_n_solutions': result['mean_n_solutions'],
            'mean_drone_missions': result['mean_drone_missions'],
            'mean_routes': result['mean_routes'],
            'n_missing_solutions': result['n_missing_solutions'],
            'n_overload_solutions': result['n_overload_solutions'],
            'max_missing': result['max_missing'],
            'max_overload': result['max_overload'],
            'all_costs': result['all_costs'],
            'all_tardiness': result['all_tardiness'],
            'all_pareto_fronts': result['all_pareto_fronts'],
        }

        # 帕累托图
        exp_key = f"{n}c_{rc}{inst:02d}_{nv}V_{end}"
        plot_pareto(result, os.path.join(results_dir, f'pareto_w7_{exp_key}.png'))

        # 路线图
        n_trucks = model.get_number_of_trucks()
        n_drones = model.get_number_of_drones()
        title = f'PACO+ALNS W7 | {n}C | {n_trucks}T+{n_drones}D | {end}'
        plot_routes(model, result, title,
                    os.path.join(results_dir, f'alns_w7_{exp_key}.png'))

        all_results.append(result_dict)

    # ── 保存 JSON ──
    json_path = os.path.join(results_dir, 'w7_results.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved: {json_path}")

    # ── 保存 Markdown 摘要 ──
    summary = "# PACO+ALNS W7 Results\n\n"
    summary += f"**Repetitions**: {args.runs} per config\n\n"
    summary += "| Config | Customers | Type | Vehicles | Endurance | Mean Cost ± Std | Mean Tardiness ± Std | HV | Time (s) | Sols | Drone | Routes | Miss | Overload |\n"
    summary += "|--------|-----------|------|----------|-----------|-----------------|----------------------|----|----------|------|-------|--------|------|----------|\n"

    for r in all_results:
        ek = f"{r['n_customers']}c_{r['rc_type']}{r['instance_id']:02d}_{r['endurance_type']}"
        summary += (
            f"| {ek} | {r['n_customers']} | {r['rc_type']} | "
            f"{r['n_vehicles']}V | {r['endurance_type']} | "
            f"{r['mean_cost']:.2f} ± {r['std_cost']:.2f} | "
            f"{r['mean_tardiness']:.2f} ± {r['std_tardiness']:.2f} | "
            f"{r['mean_hv']:.2f} | {r['mean_solve_time']:.1f} | "
            f"{r['mean_n_solutions']:.1f} | {r['mean_drone_missions']:.1f} | "
            f"{r['mean_routes']:.1f} | {r['n_missing_solutions']} | {r['n_overload_solutions']} |\n"
        )

    md_path = os.path.join(results_dir, 'w7_summary.md')
    with open(md_path, 'w') as f:
        f.write(summary)
    print(f"Saved: {md_path}")

    print(f"\n{'=' * 60}")
    print("All experiments complete!")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
