"""PACO+ALNS Proper — 实验运行脚本。

基于P-ACO-imp2架构 + 轻量存档级ALNS精炼。
跑所有16个Solomon RC配置，生成帕累托图、路线图、JSON摘要。
"""
import os, sys, json, time, argparse, importlib.util
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(os.path.dirname(BASE)))
sys.path.insert(0, os.path.join(PROJ, "src", "experiments", "PACO", "data"))
sys.path.insert(0, os.path.join(PROJ, "src", "experiments", "PACO_vs_NSGA2"))
from models.vrp_model import VRPTruckDroneModel, Route
from utils.visualizer import Visualizer
from solomon_loader_imp import SolomonLoaderImp

# 加载算法
spec = importlib.util.spec_from_file_location(
    "paco_proper", os.path.join(BASE, "PACO+ALNS_proper.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
Algo = mod.CollaborativePACOALNS_Proper


def load_model(n_customers, rc_type="RC1", instance_id=1, n_vehicles=2,
               endurance_type="medium", use_drones=True):
    loader = SolomonLoaderImp()
    if rc_type == "RC1":
        return loader.load_rc1_instance(n_customers, instance_id, n_vehicles, endurance_type, use_drones)
    else:
        return loader.load_rc2_instance(n_customers, instance_id, n_vehicles, endurance_type, use_drones)


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


def run_single(model):
    algo = Algo(model, n_ants=30, max_iter=100, alns_iter=5, alns_destroy_k=4)
    start = time.time()
    solutions, _ = algo.solve()
    elapsed = time.time() - start
    n_customers = model.get_number_of_customers()
    costs, tardiness, pf = [], [], []
    for sol in solutions:
        c, _ = model.evaluate_solution(sol)
        t = model.calculate_pure_tardiness(sol)
        served = set()
        for r in sol:
            served.update(r.customers)
            for m in r.drone_missions:
                served.update(m.customer_ids)
        missing = n_customers - len(served)
        if missing > 0:
            c += missing * 10000.0
            t += missing * 10000.0
        costs.append(c)
        tardiness.append(t)
        pf.append((c, t))
    hv = calculate_hypervolume(pf)
    return {"solutions": solutions, "pareto_front": pf,
            "costs": costs, "tardiness": tardiness,
            "solve_time": elapsed, "hypervolume": hv}


def run_experiment(model, n_runs=10):
    print("\n--- PACO+ALNS Proper ---")
    keys = ["costs", "tardiness", "solve_times", "hypervolumes",
            "solutions", "pareto_fronts"]
    acc = {k: [] for k in keys}
    for run_idx in range(n_runs):
        print(f"  Run {run_idx + 1}/{n_runs} ...", end="")
        res = run_single(model)
        print(f" done  ({res['solve_time']:.1f}s)")
        acc["costs"].extend(res["costs"])
        acc["tardiness"].extend(res["tardiness"])
        acc["solve_times"].append(res["solve_time"])
        acc["hypervolumes"].append(res["hypervolume"])
        acc["solutions"].extend(res["solutions"])
        acc["pareto_fronts"].append(res["pareto_front"])
    mean_hv = np.mean(acc["hypervolumes"])
    std_hv = np.std(acc["hypervolumes"])
    print(f"  Total solutions: {len(acc['solutions'])}")
    print(f"  Mean Cost:       {np.mean(acc['costs']):.2f} +/- {np.std(acc['costs']):.2f}")
    print(f"  Mean Tardiness:  {np.mean(acc['tardiness']):.2f} +/- {np.std(acc['tardiness']):.2f}")
    print(f"  HV:              {mean_hv:.2f} +/- {std_hv:.2f}")
    print(f"  Avg Time:        {np.mean(acc['solve_times']):.2f}s")
    return {
        "n_runs": n_runs,
        "mean_cost": float(np.mean(acc["costs"])),
        "std_cost": float(np.std(acc["costs"])),
        "mean_tardiness": float(np.mean(acc["tardiness"])),
        "std_tardiness": float(np.std(acc["tardiness"])),
        "mean_solve_time": float(np.mean(acc["solve_times"])),
        "mean_hv": float(mean_hv),
        "std_hv": float(std_hv),
        "all_costs": acc["costs"],
        "all_tardiness": acc["tardiness"],
        "all_solutions": acc["solutions"],
        "all_pareto_fronts": acc["pareto_fronts"],
    }


def plot_pareto(result, save_path):
    costs = result.get("all_costs", [])
    tardiness = result.get("all_tardiness", [])
    if not costs or not tardiness:
        return
    pf_c, pf_t = _compute_pareto_front(costs, tardiness)
    plt.figure(figsize=(10, 8))
    plt.scatter(pf_c, pf_t, facecolors="none", edgecolors="#2ca02c",
                marker="s", s=60, label="PACO+ALNS Proper")
    plt.plot(pf_c, pf_t, c="#2ca02c", linestyle="-", linewidth=1.5, alpha=0.7)
    plt.xlabel("Cost of Travel", fontsize=12)
    plt.ylabel("Penalty due to Tardiness", fontsize=12)
    plt.title("PACO+ALNS Proper — Non-dominated Front", fontsize=14, fontweight="bold")
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  Pareto plot: {save_path}")


def plot_routes(model, result, title, save_base):
    if not result["all_solutions"]:
        return
    pf = result.get("all_pareto_fronts", [])
    sols = result["all_solutions"]
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
    local_idx = int(np.argmin([f[0] for f in flat_pf]))
    idx_cost = int(pareto_indices[local_idx])
    local_idx = int(np.argmin([f[1] for f in flat_pf]))
    idx_tard = int(pareto_indices[local_idx])
    pf_sorted = sorted(flat_pf, key=lambda x: x[0])
    c1, t1 = pf_sorted[0]
    c2, t2 = pf_sorted[-1]
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
    for label, idx in [("min_cost", idx_cost), ("min_tardiness", idx_tard), ("compromise", idx_comp)]:
        if idx < len(sols):
            save_path = save_base.replace(".png", f"_{label}.png")
            vis.plot_routes(sols[idx], title=f"{title} | {label}",
                            save_path=save_path, show_all_nodes=True)
            print(f"  Route plot ({label}): {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=["small", "medium", "large", "all"], default="all")
    parser.add_argument("--runs", type=int, default=5, help="Per config")
    parser.add_argument("--test", action="store_true", help="Single run on 25c only")
    args = parser.parse_args()

    results_dir = os.path.join(BASE, "results", "20260726")
    os.makedirs(results_dir, exist_ok=True)

    configs = [
        {"n_customers": 25, "rc_type": "RC1", "n_vehicles": 2, "endurance_type": "medium", "size": "small"},
        {"n_customers": 25, "rc_type": "RC1", "n_vehicles": 2, "endurance_type": "high",   "size": "small"},
        {"n_customers": 25, "rc_type": "RC2", "n_vehicles": 2, "endurance_type": "medium", "size": "small"},
        {"n_customers": 25, "rc_type": "RC2", "n_vehicles": 2, "endurance_type": "high",   "size": "small"},
        {"n_customers": 50, "rc_type": "RC1", "n_vehicles": 4, "endurance_type": "medium", "size": "medium"},
        {"n_customers": 50, "rc_type": "RC1", "n_vehicles": 4, "endurance_type": "high",   "size": "medium"},
        {"n_customers": 50, "rc_type": "RC2", "n_vehicles": 4, "endurance_type": "medium", "size": "medium"},
        {"n_customers": 50, "rc_type": "RC2", "n_vehicles": 4, "endurance_type": "high",   "size": "medium"},
        {"n_customers": 50, "rc_type": "RC1", "n_vehicles": 6, "endurance_type": "medium", "size": "medium"},
        {"n_customers": 50, "rc_type": "RC1", "n_vehicles": 6, "endurance_type": "high",   "size": "medium"},
        {"n_customers": 50, "rc_type": "RC2", "n_vehicles": 6, "endurance_type": "medium", "size": "medium"},
        {"n_customers": 50, "rc_type": "RC2", "n_vehicles": 6, "endurance_type": "high",   "size": "medium"},
        {"n_customers": 100, "rc_type": "RC1", "n_vehicles": 10, "endurance_type": "medium", "size": "large"},
        {"n_customers": 100, "rc_type": "RC1", "n_vehicles": 10, "endurance_type": "high",   "size": "large"},
        {"n_customers": 100, "rc_type": "RC2", "n_vehicles": 10, "endurance_type": "medium", "size": "large"},
        {"n_customers": 100, "rc_type": "RC2", "n_vehicles": 10, "endurance_type": "high",   "size": "large"},
    ]

    if args.test:
        configs = [configs[0]]

    if args.size != "all" and not args.test:
        configs = [c for c in configs if c["size"] == args.size]

    all_results = []
    for cfg in configs:
        n, rc, inst, nv, end = cfg["n_customers"], cfg["rc_type"], 1, cfg["n_vehicles"], cfg["endurance_type"]
        print(f"\n{'='*60}")
        print(f"PACO+ALNS Proper | {n}c | {rc} | {nv}V | {end}")
        print(f"{'='*60}")
        model = load_model(n, rc, inst, nv, end, use_drones=True)
        print(f"  Trucks={model.get_number_of_trucks()}, Drones={model.get_number_of_drones()}, Range={model.drone_range}km")

        result = run_experiment(model, n_runs=1 if args.test else args.runs)

        result_dict = {
            "n_customers": n, "rc_type": rc, "instance_id": inst,
            "n_vehicles": nv, "endurance_type": end,
            "n_runs": 1 if args.test else args.runs,
            "algo": "PACO+ALNS_Proper",
            "mean_cost": result["mean_cost"], "std_cost": result["std_cost"],
            "mean_tardiness": result["mean_tardiness"], "std_tardiness": result["std_tardiness"],
            "mean_hv": result["mean_hv"], "std_hv": result["std_hv"],
            "mean_solve_time": result["mean_solve_time"],
            "all_costs": result["all_costs"], "all_tardiness": result["all_tardiness"],
            "all_pareto_fronts": result["all_pareto_fronts"],
        }

        exp_key = f"{n}c_{rc}{inst:02d}_{nv}V_{end}"
        plot_pareto(result, os.path.join(results_dir, f"pareto_proper_{exp_key}.png"))
        n_trucks = model.get_number_of_trucks()
        n_drones = model.get_number_of_drones()
        title = f"PACO+ALNS Proper | {n}C | {n_trucks}T+{n_drones}D | {end}"
        plot_routes(model, result, title, os.path.join(results_dir, f"alns_proper_{exp_key}.png"))
        all_results.append(result_dict)

    json_path = os.path.join(results_dir, "alns_proper_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved: {json_path}")

    md_path = os.path.join(results_dir, "alns_proper_summary.md")
    summary = "# PACO+ALNS Proper Results\n\n"
    summary += f"**Date**: 2026-07-26\n"
    summary += f"**Repetitions**: {args.runs if not args.test else 1} per config\n\n"
    summary += "| Config | Customers | Type | Vehicles | Endurance | Mean Cost +/- Std | Mean Tardiness +/- Std | HV | Time (s) |\n"
    summary += "|--------|-----------|------|----------|-----------|-----------------|------------------------|----|----------|\n"
    for r in all_results:
        ek = f"{r['n_customers']}c_{r['rc_type']}{r['instance_id']:02d}_{r['endurance_type']}"
        summary += (f"| {ek} | {r['n_customers']} | {r['rc_type']} | {r['n_vehicles']}V | {r['endurance_type']} | "
                    f"{r['mean_cost']:.2f} +/- {r['std_cost']:.2f} | {r['mean_tardiness']:.2f} +/- {r['std_tardiness']:.2f} | "
                    f"{r['mean_hv']:.2f} | {r['mean_solve_time']:.1f} |\n")
    with open(md_path, "w") as f:
        f.write(summary)
    print(f"Saved: {md_path}")
    print(f"\n{'='*60}\nAll experiments complete!\n{'='*60}")


if __name__ == "__main__":
    main()
