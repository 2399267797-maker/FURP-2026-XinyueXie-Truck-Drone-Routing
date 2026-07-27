"""compare_plot.py ? Read JSON results, generate all plots.
Usage: python compare_plot.py [--results results/comparison/raw_results.json]
"""
import os, sys, json, argparse
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(BASE), "PACO_vs_NSGA2"))
from utils.visualizer import Visualizer
from models.vrp_model import Route, DroneMission

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"PACO-imp2": "#2ca02c", "NSGA2-imp": "#1f77b4", "PACO+ALNS-W6": "#d62728"}
MARKERS = {"PACO-imp2": "o", "NSGA2-imp": "s", "PACO+ALNS-W6": "^"}
ALGO_ORDER = ["PACO-imp2", "NSGA2-imp", "PACO+ALNS-W6"]

def dict_to_routes(d):
    """Convert serialized route dicts back to Route objects."""
    routes = []
    for rd in d:
        r = Route(rd["vehicle_id"], "truck", rd["customers"].copy() if rd.get("customers") else [])
        for md in rd.get("drone_missions", []):
            r.drone_missions.append(DroneMission(md["drone_id"], md["customer_ids"].copy(),
                                                  md["launch_point"], md["return_point"]))
        routes.append(r)
    return routes

def plot_pareto(all_data, save_dir):
    """One Pareto plot per config, all algos overlaid."""
    for cfg_key, algos in all_data.items():
        plt.figure(figsize=(10, 8))
        for aname in ALGO_ORDER:
            if aname not in algos: continue
            trials = algos[aname].get("trials", [])
            all_c, all_t = [], []
            for t in trials:
                all_c.extend(t.get("costs", [])); all_t.extend(t.get("tardiness", []))
            if not all_c: continue
            pts = np.column_stack([all_c, all_t])
            is_dom = np.zeros(len(pts), dtype=bool)
            for i in range(len(pts)):
                for j in range(len(pts)):
                    if i != j and not is_dom[j]:
                        if pts[i,0] <= pts[j,0] and pts[i,1] <= pts[j,1] and \
                           (pts[i,0] < pts[j,0] or pts[i,1] < pts[j,1]):
                            is_dom[j] = True
            pf = pts[~is_dom]
            pf = pf[np.argsort(pf[:,0])]
            if len(pf) > 0:
                plt.scatter(pf[:,0], pf[:,1], facecolors="none",
                           edgecolors=COLORS[aname], marker=MARKERS[aname], s=60, label=aname)
                plt.plot(pf[:,0], pf[:,1], c=COLORS[aname], linestyle="-", linewidth=1.5, alpha=0.7)
        plt.xlabel("Cost of Travel", fontsize=12)
        plt.ylabel("Tardiness Penalty", fontsize=12)
        plt.title(f"Pareto Front ? {cfg_key}", fontsize=14, fontweight="bold")
        plt.legend(loc="upper right", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(save_dir, f"pareto_{cfg_key}.png")
        plt.savefig(path, dpi=300); plt.close()
        print(f"  Pareto: {path}")

def plot_routes(all_data, save_dir, model_loader):
    """One route plot per algo per config (min_cost and min_tard)."""
    import sys as _sys_mod; _sys_mod.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "PACO", "data")); from solomon_loader_imp import SolomonLoaderImp as SolomonLoader
    loader = SolomonLoader()
    for cfg_key, algos in all_data.items():
        parts = cfg_key.split("_")
        n, rc, end = int(parts[0]), parts[1], parts[2]
        nv = 2 if n <= 25 else 4
        if "6V" in cfg_key: nv = 6
        try:
            model = loader.load_rc1_instance(n, 1, nv, end, True) if rc == "RC1" else \
                    loader.load_rc2_instance(n, 1, nv, end, True)
        except:
            continue
        vis = Visualizer(model)
        for aname in ALGO_ORDER:
            if aname not in algos: continue
            trials = algos[aname].get("trials", [])
            if not trials: continue
            # Best trial by number of solutions
            best = max(trials, key=lambda t: t.get("n_solutions", 0))
            for label, key in [("min_cost", "min_cost"), ("min_tard", "min_tard")]:
                rep_data = best.get("representative", {}).get(key, [])
                if not rep_data: continue
                routes = dict_to_routes(rep_data)
                n_t = model.get_number_of_trucks()
                n_d = model.get_number_of_drones()
                title = f"{aname} | {n}C | {n_t}T+{n_d}D | {end} | {label}"
                fname = f"{aname.replace(chr(43),chr(95)).replace(chr(45),chr(95))}_{cfg_key}_{label}.png"
                path = os.path.join(save_dir, fname)
                try:
                    vis.plot_routes(routes, title=title, save_path=path, show_all_nodes=True)
                    print(f"  Route: {path}")
                except Exception as e:
                    print(f"  Route fail ({fname}): {e}")

def plot_summary(all_data, save_dir):
    """Bar chart comparing HV and time across all configs."""
    cfg_keys = sorted(all_data.keys())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for idx, metric in enumerate(["hypervolume", "mean_time"]):
        ax = axes[idx]
        x = np.arange(len(cfg_keys))
        w = 0.25
        for mi, aname in enumerate(ALGO_ORDER):
            vals = [all_data[k].get(aname, {}).get(metric, 0) for k in cfg_keys]
            ax.bar(x + mi*w - w, vals, w, label=aname, color=COLORS[aname], alpha=0.8)
        ax.set_xlabel("Config")
        ax.set_ylabel("Time (s)" if metric == "mean_time" else "Hypervolume")
        ax.set_title(f"{'Avg Solve Time' if metric == 'mean_time' else 'Hypervolume'} by Config")
        ax.set_xticks(x)
        ax.set_xticklabels(cfg_keys, rotation=30, ha="right", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(save_dir, "summary.png")
    plt.savefig(path, dpi=300); plt.close()
    print(f"  Summary: {path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str,
                        default="results/comparison/raw_results.json")
    args = parser.parse_args()
    path = os.path.join(BASE, args.results)
    if not os.path.exists(path):
        print(f"Results not found: {path}")
        print("Run compare_run.py first.")
        return
    with open(path) as f:
        all_data = json.load(f)
    save_dir = os.path.join(os.path.dirname(path), "plots")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Generating plots in {save_dir}...")
    plot_pareto(all_data, save_dir)
    plot_routes(all_data, save_dir, None)
    plot_summary(all_data, save_dir)
    print(f"\nAll plots saved to {save_dir}")

if __name__ == "__main__":
    main()
