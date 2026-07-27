"""
plot_result.py — 轻量级路线图绘制脚本

从 PyPy 运行的 PACO+ALNS 输出的 JSON 文件中读取 Pareto 最优解，
用标准 Python (CPython) + matplotlib 绘制精美的路线图。

用法：
    python plot_result.py results.json                          # 默认输出到 results/ 目录
    python plot_result.py results.json --outdir my_plots        # 指定输出目录
    python plot_result.py results.json --solution 0             # 只绘制第 0 个 Pareto 解
    python plot_result.py results.json --pareto-only            # 只绘制 Pareto 前沿
"""

import json
import os
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Tuple


# ── 配色方案 ──
TRUCK_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
DRONE_COLORS = ['#17becf', '#bcbd22', '#e377c2', '#7f7f7f', '#ff9896',
                '#c5b0d5', '#c49c94', '#f7b6d2', '#dbdb8d', '#9edae5']


def load_data(filepath: str) -> dict:
    with open(filepath, 'r') as f:
        return json.load(f)


def plot_pareto_front(data: dict, save_path: str):
    """绘制 Pareto 前沿散点图。"""
    front = data['pareto_front']
    if not front:
        print("  Pareto front is empty, skipping.")
        return

    costs = [p[0] for p in front]
    tardiness = [p[1] for p in front]

    # 按 cost 排序后连线
    pts = sorted(zip(costs, tardiness), key=lambda x: x[0])
    pf_c, pf_t = [p[0] for p in pts], [p[1] for p in pts]

    plt.figure(figsize=(10, 8))
    plt.scatter(pf_c, pf_t, facecolors='none', edgecolors='#d62728',
                marker='^', s=80, label='Pareto Front', linewidths=1.5)
    plt.plot(pf_c, pf_t, c='#d62728', linestyle='-', linewidth=1.5, alpha=0.6)

    plt.xlabel('Travel Cost', fontsize=12)
    plt.ylabel('Tardiness Penalty', fontsize=12)
    plt.title('PACO+ALNS — Pareto Front', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  Pareto front plot: {save_path}")


def plot_single_solution(data: dict, sol_idx: int, save_path: str):
    """绘制单个 Pareto 解的路线图。

    支持两种数据格式：
      1. 新格式: data['solutions'][sol_idx]['routes']（含 drone_missions）
      2. 旧格式: data['solutions'][sol_idx] 直接是 route 列表
    """
    model = data['model']
    depot = model['depot']
    customers = model['customers']
    n_trucks = model['n_trucks']
    n_drones = model['n_drones']
    drone_range = model.get('drone_range', 4.0)

    sol = data['solutions'][sol_idx]
    # 兼容新旧格式
    if isinstance(sol, dict) and 'routes' in sol:
        routes = sol['routes']
    else:
        routes = sol  # 旧格式：直接是列表

    plt.figure(figsize=(12, 10))

    # ── 绘制仓库 ──
    plt.scatter(depot['x'], depot['y'], c='red', marker='s', s=150,
                label='Depot', zorder=10, edgecolors='black', linewidths=1)

    # ── 绘制卡车路线 ──
    served_customers = set()
    for route in routes:
        vid = route['vehicle_id']
        color = TRUCK_COLORS[vid % len(TRUCK_COLORS)]
        cust_ids = route['customers']
        served_customers.update(cust_ids)

        if cust_ids:
            xs = [depot['x']]
            ys = [depot['y']]
            for cid in cust_ids:
                c = customers[cid]
                xs.append(c['x'])
                ys.append(c['y'])
                plt.annotate(f'C{cid}', (c['x'], c['y']),
                             textcoords="offset points", xytext=(5, 5),
                             fontsize=8, ha='center')
            xs.append(depot['x'])
            ys.append(depot['y'])

            plt.plot(xs, ys, color=color, linestyle='-', marker='o',
                     markersize=8, linewidth=2, label=f'Truck {vid}', zorder=3)
            for cid in cust_ids:
                c = customers[cid]
                plt.scatter(c['x'], c['y'], c=color, marker='o', s=60,
                            zorder=5, edgecolors='white', linewidths=1)

    # ── 绘制无人机任务 ──
    drone_labeled = set()
    for route in routes:
        for mission in route.get('drone_missions', []):
            did = mission['drone_id']
            dc_id = mission['customer_id']
            served_customers.add(dc_id)
            drone_color = DRONE_COLORS[did % len(DRONE_COLORS)]

            # 发射点坐标
            lp = mission['launch_point']
            cust_ids = route['customers']
            if lp == -1 or lp >= len(cust_ids):
                lx, ly = depot['x'], depot['y']
            else:
                lc = customers[cust_ids[lp]]
                lx, ly = lc['x'], lc['y']

            # 回收点坐标
            rp = mission['return_point']
            if rp == -1 or rp >= len(cust_ids):
                rx, ry = depot['x'], depot['y']
            else:
                rc = customers[cust_ids[rp]]
                rx, ry = rc['x'], rc['y']

            # 无人机服务的客户
            dc = customers[dc_id]
            label = f'Drone {did}' if did not in drone_labeled else None
            drone_labeled.add(did)
            plt.scatter(dc['x'], dc['y'], c=drone_color, marker='^', s=100,
                        zorder=6, edgecolors='black', linewidths=1.5, label=label)
            plt.annotate(f'D-C{dc_id}', (dc['x'], dc['y']),
                         textcoords="offset points", xytext=(5, -12),
                         fontsize=8, ha='center', fontweight='bold')

            # 发射路径（虚线）
            plt.plot([lx, dc['x']], [ly, dc['y']],
                     color=drone_color, linestyle='--', linewidth=2, alpha=0.8, zorder=4)
            # 返回路径（点线）
            plt.plot([dc['x'], rx], [dc['y'], ry],
                     color=drone_color, linestyle=':', linewidth=2, alpha=0.8, zorder=4)

    # ── 标记未服务客户 ──
    all_cust_ids = set(c['id'] for c in customers)
    unserved = all_cust_ids - served_customers
    for cid in unserved:
        c = customers[cid]
        plt.scatter(c['x'], c['y'], c='gray', marker='x', s=60,
                    label='Unserved', zorder=4, alpha=0.5)

    # 配置信息
    config_text = (f"Trucks: {n_trucks} | Drones: {n_drones}\n"
                   f"Drone Range: {drone_range} km\n"
                   f"-- Dashed: Launch | : Dotted: Return")
    plt.annotate(config_text, xy=(0.02, 0.02), xycoords='axes fraction',
                 fontsize=10, verticalalignment='bottom',
                 bbox=dict(boxstyle='round', facecolor='#f8f9fa', alpha=0.9, edgecolor='gray'))

    # 去重图例
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right',
               bbox_to_anchor=(1.25, 1), fontsize=10, borderaxespad=0.)

    plt.title(f'PACO+ALNS — Solution {sol_idx} '
              f'(Cost={data["pareto_front"][sol_idx][0]:.1f}, '
              f'Tard={data["pareto_front"][sol_idx][1]:.1f})',
              fontsize=14, fontweight='bold')
    plt.xlabel('X Coordinate (km)', fontsize=12)
    plt.ylabel('Y Coordinate (km)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Route plot (solution {sol_idx}): {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot PACO+ALNS results from JSON')
    parser.add_argument('json_file', help='Path to the JSON result file')
    parser.add_argument('--outdir', default=None,
                        help='Output directory for plots (default: results/ subfolder)')
    parser.add_argument('--solution', type=int, default=None,
                        help='Plot only a specific solution index')
    parser.add_argument('--pareto-only', action='store_true',
                        help='Only plot the Pareto front, skip route maps')
    args = parser.parse_args()

    json_path = args.json_file
    if not os.path.exists(json_path):
        print(f"Error: file not found: {json_path}")
        sys.exit(1)

    data = load_data(json_path)
    model = data['model']
    n_solutions = len(data['solutions'])
    print(f"Loaded: {model['n_customers']} customers, "
          f"{model['n_trucks']} trucks, {model['n_drones']} drones, "
          f"{n_solutions} Pareto solutions")

    # 输出目录
    if args.outdir:
        out_dir = args.outdir
    else:
        base = os.path.dirname(json_path) or '.'
        out_dir = os.path.join(base, 'plots')
    os.makedirs(out_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(json_path))[0]

    # ── Pareto 前沿 ──
    if data['pareto_front']:
        pf_path = os.path.join(out_dir, f'{base_name}_pareto.png')
        plot_pareto_front(data, pf_path)

    # ── 路线图 ──
    if args.pareto_only:
        return

    if args.solution is not None:
        indices = [args.solution]
    else:
        indices = list(range(n_solutions))

    for idx in indices:
        if idx >= n_solutions:
            print(f"Warning: solution index {idx} out of range (max {n_solutions - 1}), skipping")
            continue
        save_path = os.path.join(out_dir, f'{base_name}_solution{idx}.png')
        plot_single_solution(data, idx, save_path)

    print(f"\nAll plots saved to: {out_dir}")


if __name__ == '__main__':
    main()