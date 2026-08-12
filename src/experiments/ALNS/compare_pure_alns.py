"""Four-way comparison: NSGA-II / PACO-imp2 / PACO+ALNS W8 vs pure ALNS.

The NSGA-II / PACO-imp2 / PACO+ALNS W8 results are the existing all-Solomon
runs stored in ../PACO+ALNS/results/20260809_w8 (3 runs per configuration).
This script enumerates exactly those configurations, generates the pure-ALNS
results for them, and writes the four-way comparison into
../PACO+ALNS/results/<outdir> (default 20260812_4alg).

Usage:
  python compare_pure_alns.py --runs 3
  python compare_pure_alns.py --family RC --size small --limit 4 --runs 1
"""

import os
import re
import sys
import json
import time
import random
import argparse
from multiprocessing import Pool
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


BASE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(os.path.dirname(BASE)))
EXPERIMENTS = os.path.join(PROJ, 'src', 'experiments')
PACO_ALNS = os.path.join(EXPERIMENTS, 'PACO+ALNS')
EXISTING_DIR = os.path.join(PACO_ALNS, 'results', '20260809_w8')
RESULTS_ROOT = os.path.join(PACO_ALNS, 'results')

sys.path.insert(0, PACO_ALNS)
sys.path.insert(0, os.path.join(EXPERIMENTS, 'PACO', 'data'))
sys.path.insert(0, os.path.join(EXPERIMENTS, 'PACO_vs_NSGA2'))

import compare_three_algorithms as c3  # noqa: E402
from pure_alns import PureALNS  # noqa: E402


ALGORITHMS = ['nsga2', 'imp2', 'w8', 'alns']

ALGO_STYLES = {
    'nsga2': {'label': 'NSGA-II', 'color': '#1f77b4', 'marker': 'o'},
    'imp2': {'label': 'PACO-imp2', 'color': '#2ca02c', 'marker': 's'},
    'w8': {'label': 'PACO+ALNS W8', 'color': '#d62728', 'marker': '^'},
    'alns': {'label': 'Pure ALNS', 'color': '#9467bd', 'marker': 'D'},
}

EXISTING_FILE_NAMES = {'nsga2': 'nsga2', 'imp2': 'imp2', 'w8': 'w8'}

KEY_RE = re.compile(
    r'^(?P<n>\d+)c_(?P<fam>[A-Z]+)(?P<typ>[12])(?P<iid>\d+)_'
    r'(?P<nv>\d+)V_(?P<end>medium|high)$'
)


def configs_from_existing() -> List[Dict]:
    configs, seen = [], set()
    for fn in sorted(os.listdir(EXISTING_DIR)):
        if not fn.endswith('_nsga2.json'):
            continue
        key = fn[:-len('_nsga2.json')]
        m = KEY_RE.match(key)
        if not m or key in seen:
            continue
        seen.add(key)
        n = int(m.group('n'))
        configs.append({
            'n_customers': n,
            'rc_type': f"{m.group('fam')}{m.group('typ')}",
            'instance_id': int(m.group('iid')),
            'n_vehicles': int(m.group('nv')),
            'endurance_type': m.group('end'),
            'size': 'small' if n <= 25 else ('medium' if n <= 50 else 'large'),
        })
    return configs


def run_single_alns(model, max_iter: int, n_starts: int, alns_iter: int,
                    seed: int) -> Dict:
    random.seed(seed)
    np.random.seed(seed)
    algo_obj = PureALNS(model, max_iter=max_iter,
                        n_starts=n_starts or None,
                        alns_iter=alns_iter or None)
    t0 = time.time()
    solutions, _ = algo_obj.solve()
    elapsed = time.time() - t0

    costs, tardiness, front = [], [], []
    missing_counts, overload_amounts, drone_counts, route_counts = [], [], [], []
    for sol in solutions:
        ev = c3.evaluate_solution(model, sol)
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
        'n_starts': algo_obj.n_starts,
        'alns_iter': algo_obj.alns_iter,
    }


def run_experiment_alns(model, n_runs: int, max_iter: int, n_starts: int,
                        alns_iter: int, base_seed: int) -> Dict:
    acc = {'costs': [], 'tardiness': [], 'solve_times': [],
           'pareto_fronts': [], 'solutions': []}
    n_sol_list, drone_means, route_means = [], [], []
    missing_runs = overload_runs = max_missing = 0
    max_overload = 0.0
    effective_starts, effective_alns_iter = n_starts, alns_iter

    for run_idx in range(n_runs):
        seed = base_seed + run_idx * 37
        res = run_single_alns(model, max_iter, n_starts, alns_iter, seed)
        effective_starts = res['n_starts']
        effective_alns_iter = res['alns_iter']
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

    has_solutions = bool(acc['costs'])
    return {
        'n_runs': n_runs,
        'mean_cost': float(np.mean(acc['costs'])) if has_solutions else 1e9,
        'std_cost': float(np.std(acc['costs'])) if has_solutions else 0.0,
        'mean_tardiness': float(np.mean(acc['tardiness'])) if has_solutions else 1e9,
        'std_tardiness': float(np.std(acc['tardiness'])) if has_solutions else 0.0,
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
        'n_starts': effective_starts,
        'alns_iter': effective_alns_iter,
    }


def run_config(task: Dict) -> Dict:
    cfg = task['config']
    key = c3.config_key(cfg)
    print(f"[start] {key} alns runs={task['runs']} max_iter={task['max_iter']} "
          f"starts={task['starts']} alns_iter={task['alns_iter']} "
          f"seed={task['seed']}", flush=True)

    random.seed(task['seed'])
    np.random.seed(task['seed'])
    model = c3.load_model(cfg['n_customers'], cfg['rc_type'], cfg['instance_id'],
                          cfg['n_vehicles'], cfg['endurance_type'], use_drones=True)
    result = run_experiment_alns(model, task['runs'], task['max_iter'],
                                 task['starts'], task['alns_iter'], task['seed'])

    if task.get('route_plots', False):
        title = (f"Pure ALNS | {cfg['n_customers']}C | "
                 f"{model.get_number_of_trucks()}T+{model.get_number_of_drones()}D | "
                 f"{cfg['endurance_type']}")
        c3.plot_routes(model, result, title,
                       os.path.join(task['out_dir'], f"routes_alns_{key}.png"))

    raw = dict(result)
    raw.pop('all_solutions', None)
    raw['module'] = 'pure_alns'
    out_path = os.path.join(task['out_dir'], f"{key}_alns.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(raw, f, indent=2, ensure_ascii=False)

    print(f"[done] {key} alns time={result['mean_solve_time']:.1f}s "
          f"cost={result['mean_cost']:.1f} tard={result['mean_tardiness']:.1f} "
          f"sols={result['mean_n_solutions']:.1f}", flush=True)
    return result


def plot_compare_pareto(results: Dict[str, Dict], save_path: str, title: str):
    plt.figure(figsize=(10, 8))
    for algo in ALGORITHMS:
        res = results.get(algo)
        if not res:
            continue
        pf = c3.compute_pareto_front(list(zip(res['all_costs'], res['all_tardiness'])))
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


def write_analysis(results_by_key: Dict, out_path: str, args) -> str:
    n_configs = len(results_by_key)
    lines = ["# 四种算法对比分析（NSGA-II / PACO-imp2 / PACO+ALNS W8 / Pure ALNS）", ""]
    lines.append("## 实验设置")
    lines.append("")
    lines.append(f"- 数据集：Solomon 全系列（C/R/RC），与 `20260809_w8` 完全一致，共 {n_configs} 组。")
    lines.append("- 规模：25c/50c/100c，续航 medium/high，车辆数为该实例自动配置。")
    lines.append(f"- 重复次数：每组 {args.runs} 次。")
    lines.append("- NSGA-II / PACO-imp2 / PACO+ALNS W8：沿用 `20260809_w8` 已有结果。")
    lines.append(f"- Pure ALNS：restarts={args.starts or args.max_iter}（与 W8 的外循环预算 max_iter 一致），"
                 "alns_iter 采用 W8 自适应默认值，标量化权重 tard_penalty_truck=10 固定与 W8 一致；"
                 "仅移除 PACO 构造与信息素更新。")
    lines.append("- 指标口径：cost 来自 `model.evaluate_solution`；延迟为 `calculate_pure_tardiness`；")
    lines.append("  对缺客户解施加 10000/客户、对超载量施加 1000/单位惩罚后再计算前沿、HV 与均值。")
    lines.append("- HV 参考点：沿用 `20260809_w8` 三算法对比存储的同一参考点")
    lines.append("  （原三算法解集最大成本、最大延迟分别 x1.1），Pure ALNS 也按该参考点计算，")
    lines.append("  保证四家的 HV 数值直接可比。")
    lines.append("- 若某算法在某配置下无可行解（mean_n_solutions=0），成本/延迟记为 infeasible，")
    lines.append("  且该配置不参与统计排名汇总。")
    lines.append("")

    lines.append("## 汇总表")
    lines.append("")
    header = ("| Config | Algo | Cost +/- Std | Tardiness +/- Std | HV | Time (s) | "
              "Sols | Drones | Routes | Miss | Overload |")
    lines.append(header)
    lines.append("|--------|------|--------------|------------------|----|----------|"
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
            if r['mean_n_solutions'] <= 0:
                lines.append(
                    f"| {key} | {ALGO_STYLES[algo]['label']} | infeasible | infeasible | "
                    f"{r['mean_hv']:.2f} | {r['mean_solve_time']:.1f} | "
                    f"0.0 | - | - | - | - |")
                continue
            lines.append(
                f"| {key} | {ALGO_STYLES[algo]['label']} | "
                f"{r['mean_cost']:.2f} +/- {r['std_cost']:.2f} | "
                f"{r['mean_tardiness']:.2f} +/- {r['std_tardiness']:.2f} | "
                f"{r['mean_hv']:.2f} | {r['mean_solve_time']:.1f} | "
                f"{r['mean_n_solutions']:.1f} | {r['mean_drone_missions']:.1f} | "
                f"{r['mean_routes']:.1f} | {r['n_missing_solutions']} | "
                f"{r['n_overload_solutions']} |")

        feasible = [a for a in ALGORITHMS if pair[a]['mean_n_solutions'] > 0]
        if not feasible:
            continue
        best_cost = min(pair[a]['mean_cost'] for a in feasible)
        best_tard = min(pair[a]['mean_tardiness'] for a in feasible)
        best_hv = max(pair[a]['mean_hv'] for a in feasible)
        for algo in ALGORITHMS:
            r = pair[algo]
            if algo not in feasible:
                continue
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
    lines.append("- Pure ALNS 计算预算小于 PACO+ALNS W8，耗时差异应在解读中说明。")
    lines.append("- 详细数据见同目录 JSON 文件（含每轮前沿、每解成本/延迟、HV 参考点等）。")

    text = "\n".join(lines) + "\n"
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(text)
    return text


def main():
    parser = argparse.ArgumentParser(
        description='Add Pure ALNS to the 20260809_w8 Solomon comparison')
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--max-iter', type=int, default=100)
    parser.add_argument('--starts', type=int, default=0,
                        help='ALNS restarts per run; 0 = auto scale')
    parser.add_argument('--alns-iter', type=int, default=0,
                        help='ALNS iterations per restart; 0 = W8 adaptive default')
    parser.add_argument('--family', choices=['C', 'R', 'RC', 'all'], default='all')
    parser.add_argument('--size', choices=['small', 'medium', 'large', 'all'], default='all')
    parser.add_argument('--limit', type=int, default=0,
                        help='only run the first N configurations (smoke test)')
    parser.add_argument('--outdir', type=str, default='20260812_4alg')
    parser.add_argument('--workers', type=int, default=min(16, os.cpu_count() or 4))
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--route-plots', action='store_true')
    parser.add_argument('--no-pareto-plots', action='store_true')
    parser.add_argument('--no-copy-existing', action='store_true',
                        help='keep the existing nsga2/imp2/w8 JSONs in the outdir '
                             'instead of re-copying them from 20260809_w8')
    args = parser.parse_args()

    configs = configs_from_existing()
    if args.family != 'all':
        configs = [c for c in configs if c['rc_type'].startswith(args.family)]
    if args.size != 'all':
        configs = [c for c in configs if c['size'] == args.size]
    if args.limit > 0:
        configs = configs[:args.limit]
    if not configs:
        print("No configurations selected.", flush=True)
        return

    results_dir = os.path.join(RESULTS_ROOT, args.outdir)
    os.makedirs(results_dir, exist_ok=True)

    # Copy the existing three-algorithm results (always refresh them so the
    # original stored hv_reference is restored), unless the outdir already
    # contains updated results such as the rerun NSGA-II JSONs.
    if not args.no_copy_existing:
        for cfg in configs:
            key = c3.config_key(cfg)
            for algo, fname in EXISTING_FILE_NAMES.items():
                src = os.path.join(EXISTING_DIR, f"{key}_{fname}.json")
                dst = os.path.join(results_dir, f"{key}_{algo}.json")
                if not os.path.exists(src):
                    print(f"[warn] missing existing result {src}", flush=True)
                    continue
                with open(src, encoding='utf-8') as f:
                    data = json.load(f)
                with open(dst, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

    tasks = []
    for idx, cfg in enumerate(configs):
        key = c3.config_key(cfg)
        out_path = os.path.join(results_dir, f"{key}_alns.json")
        if os.path.exists(out_path) and not args.force:
            print(f"[skip] {key} alns (already exists)", flush=True)
            continue
        tasks.append({
            'config': cfg, 'runs': args.runs, 'max_iter': args.max_iter,
            'starts': args.starts, 'alns_iter': args.alns_iter,
            'seed': 20260812 + idx * 1000 + 3,
            'out_dir': results_dir,
            'route_plots': args.route_plots,
        })

    if tasks:
        print(f"[info] running {len(tasks)} pure-ALNS configurations "
              f"({args.runs} runs each)", flush=True)
        with Pool(args.workers) as pool:
            pool.map(run_config, tasks)

    results_by_key = {}
    for cfg in configs:
        key = c3.config_key(cfg)
        pair = {}
        for algo in ALGORITHMS:
            path = os.path.join(results_dir, f"{key}_{algo}.json")
            if not os.path.exists(path):
                print(f"[warn] missing {key} {algo}", flush=True)
                continue
            with open(path, encoding='utf-8') as f:
                pair[algo] = json.load(f)
        results_by_key[key] = pair

    for key, pair in results_by_key.items():
        # Use the reference stored by the original 20260809_w8 three-way
        # comparison so HV is directly comparable with the PACO results.
        ref = None
        src_w8 = os.path.join(EXISTING_DIR, f"{key}_w8.json")
        if os.path.exists(src_w8):
            with open(src_w8, encoding='utf-8') as f:
                orig = json.load(f)
            ref = orig.get('hv_reference')
        if not ref:
            all_costs, all_tards = [], []
            for algo in ALGORITHMS:
                r = pair.get(algo)
                if r:
                    all_costs.extend(r['all_costs'])
                    all_tards.extend(r['all_tardiness'])
            if not all_costs or not all_tards:
                continue
            ref = [float(np.max(all_costs) * 1.1), float(np.max(all_tards) * 1.1)]
        ref_cost, ref_tard = float(ref[0]), float(ref[1])
        for algo in ALGORITHMS:
            if algo not in pair:
                continue
            pair[algo] = c3.finalize_result(pair[algo], ref_cost, ref_tard)
            cfg = next(c for c in configs if c3.config_key(c) == key)
            pair[algo].update({
                'config': key,
                'algo': algo,
                'n_customers': cfg['n_customers'],
                'rc_type': cfg['rc_type'],
                'instance_id': cfg['instance_id'],
                'n_vehicles': cfg['n_vehicles'],
                'endurance_type': cfg['endurance_type'],
                'hv_reference': [ref_cost, ref_tard],
            })
            with open(os.path.join(results_dir, f"{key}_{algo}.json"), 'w', encoding='utf-8') as f:
                json.dump(pair[algo], f, indent=2, ensure_ascii=False)

        if not args.no_pareto_plots:
            title = ("NSGA-II vs PACO-imp2 vs PACO+ALNS W8 vs Pure ALNS | " + key)
            plot_compare_pareto(pair,
                                os.path.join(results_dir, f"pareto_compare_{key}.png"),
                                title)

    combined_path = os.path.join(results_dir, f'compare_pure_alns_{args.outdir}.json')
    with open(combined_path, 'w', encoding='utf-8') as f:
        json.dump(results_by_key, f, indent=2, ensure_ascii=False)
    print(f"Saved: {combined_path}", flush=True)

    md_path = os.path.join(results_dir, f'analysis_{args.outdir}.md')
    write_analysis(results_by_key, md_path, args)
    print(f"Saved: {md_path}", flush=True)


if __name__ == '__main__':
    main()
