"""Generate figures for the summary paper:
  1. representative Pareto fronts (copied from the full sweep),
  2. convergence curves (best cost / HV vs iteration budget),
  3. route maps for C101.

Usage:
    python make_report_figures.py
"""

import os
import shutil
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import compare_three_algorithms as cmp


BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, 'results', '20260809_w8')
FIG_DIR = os.path.join(RESULTS, 'figures')

ALGOS = ['nsga2', 'imp2', 'w8']
ALGO_STYLE = {
    'nsga2': {'label': 'NSGA-II', 'color': '#1f77b4', 'marker': 'o'},
    'imp2': {'label': 'PACO-imp2', 'color': '#2ca02c', 'marker': 's'},
    'w8': {'label': 'PACO+ALNS', 'color': '#d62728', 'marker': '^'},
}

BUDGETS = [5, 10, 20, 30, 50, 100]


def copy_pareto_figures():
    os.makedirs(FIG_DIR, exist_ok=True)
    mapping = [
        ('pareto_compare_100c_C101_10V_medium.png', 'pareto_C101_100c_medium.png'),
        ('pareto_compare_100c_R101_10V_medium.png', 'pareto_R101_100c_medium.png'),
        ('pareto_compare_100c_RC102_10V_medium.png', 'pareto_RC102_100c_medium.png'),
    ]
    for src, dst in mapping:
        src_path = os.path.join(RESULTS, src)
        if os.path.exists(src_path):
            shutil.copy2(src_path, os.path.join(FIG_DIR, dst))
            print('copied', dst)


def make_convergence_figures():
    model = cmp.load_model(25, 'C1', 1, None, 'medium')
    fronts = {a: [] for a in ALGOS}
    best_costs = {a: [] for a in ALGOS}
    for a in ALGOS:
        algo_mod = cmp.load_algorithm(a, 'w8')
        for budget in BUDGETS:
            res = cmp.run_single(model, algo_mod, a, budget, seed=20260812 + 100 * ALGOS.index(a) + budget)
            best_costs[a].append(min(res['costs']) if res['costs'] else float('nan'))
            fronts[a].append(res['pareto_front'])
    all_pts = [pt for a in ALGOS for f in fronts[a] for pt in f]
    ref_cost = float(np.max([p[0] for p in all_pts]) * 1.1)
    ref_tard = float(np.max([p[1] for p in all_pts]) * 1.1)
    hvs = {a: [] for a in ALGOS}
    for a in ALGOS:
        for f in fronts[a]:
            hvs[a].append(cmp.calculate_hypervolume(f, ref_cost, ref_tard))

    fig1, ax1 = plt.subplots(figsize=(7, 5))
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    for a in ALGOS:
        st = ALGO_STYLE[a]
        ax1.plot(BUDGETS, best_costs[a], color=st['color'], marker=st['marker'],
                 label=st['label'])
        ax2.plot(BUDGETS, hvs[a], color=st['color'], marker=st['marker'],
                 label=st['label'])
    ax1.set_xlabel('Iterations / Generations')
    ax1.set_ylabel('Best Cost')
    ax1.set_title('Convergence: Best Cost (C101, 25c, medium)')
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.set_xlabel('Iterations / Generations')
    ax2.set_ylabel('Hypervolume')
    ax2.set_title('Convergence: Hypervolume (C101, 25c, medium)')
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig1.tight_layout()
    fig2.tight_layout()
    cost_path = os.path.join(FIG_DIR, 'convergence_cost_25c_C101_medium.png')
    hv_path = os.path.join(FIG_DIR, 'convergence_hv_25c_C101_medium.png')
    fig1.savefig(cost_path, dpi=200)
    fig2.savefig(hv_path, dpi=200)
    plt.close(fig1)
    plt.close(fig2)
    print('saved convergence figures')


def make_route_figures():
    model = cmp.load_model(100, 'C1', 1, None, 'medium')
    for a in ALGOS:
        algo_mod = cmp.load_algorithm(a, 'w8')
        res = cmp.run_single(model, algo_mod, a, 50, seed=20260812 + ALGOS.index(a))
        result = {
            'all_solutions': res['solutions'],
            'all_pareto_fronts': [res['pareto_front']],
        }
        title = f"{ALGO_STYLE[a]['label']} | C101 | 100C | {model.get_number_of_trucks()}T+{model.get_number_of_drones()}D | medium"
        cmp.plot_routes(model, result, title,
                        os.path.join(FIG_DIR, f'routes_{a}_C101_100c_medium.png'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-pareto', action='store_true')
    parser.add_argument('--skip-convergence', action='store_true')
    parser.add_argument('--skip-routes', action='store_true')
    args = parser.parse_args()
    os.makedirs(FIG_DIR, exist_ok=True)
    if not args.skip_pareto:
        copy_pareto_figures()
    if not args.skip_convergence:
        make_convergence_figures()
    if not args.skip_routes:
        make_route_figures()
    print('figures ready in', FIG_DIR)


if __name__ == '__main__':
    main()
