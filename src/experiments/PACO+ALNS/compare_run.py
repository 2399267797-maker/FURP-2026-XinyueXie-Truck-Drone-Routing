"""compare_run.py ? Run all benchmarks and save results as JSON.
Usage:
    python compare_run.py                          # default: 25c+50c, 3 runs
    python compare_run.py --configs 25_RC1_medium
    python compare_run.py --runs 1 --fast          # quick smoke test
"""
import os, sys, time, json, argparse, importlib.util, multiprocessing as mp
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS = os.path.dirname(BASE)
for p in [BASE, os.path.join(EXPERIMENTS, "PACO"),
          os.path.join(EXPERIMENTS, "PACO", "data"),
          os.path.join(EXPERIMENTS, "PACO_vs_NSGA2"),
          os.path.join(EXPERIMENTS, "NSGA2")]:
    if p not in sys.path: sys.path.insert(0, p)

from models.vrp_model import Route, DroneMission
from solomon_loader_imp import SolomonLoaderImp as SolomonLoader

# ?? Import algorithms (lazy, to support PyPy subprocess mode) ??
# Module-level singletons
_PACO_IMP2 = None
_NSGA2_IMP = None
_W6 = None

def get_algo_classes():
    global _PACO_IMP2, _NSGA2_IMP, _W6
    if _PACO_IMP2 is not None:
        return _PACO_IMP2, _NSGA2_IMP, _W6
    spec = importlib.util.spec_from_file_location("paco_imp2",
        os.path.join(EXPERIMENTS, "PACO", "algorithms", "paco_imp2.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    p2 = m.CollaborativePACO
    spec2 = importlib.util.spec_from_file_location("nsga2_imp",
        os.path.join(EXPERIMENTS, "NSGA2", "nsga2_imp.py"))
    m2 = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(m2)
    n2 = m2.NSGA2VRP
    spec3 = importlib.util.spec_from_file_location("w6",
        os.path.join(BASE, "PACO+ALNSW6.py"))
    m3 = importlib.util.module_from_spec(spec3); spec3.loader.exec_module(m3)
    w6 = m3.CollaborativePACOALNS
    _PACO_IMP2, _NSGA2_IMP, _W6 = p2, n2, w6
    return p2, n2, w6

def load_model(n_customers, rc_type, n_vehicles, endurance):
    loader = SolomonLoader()
    kw = dict(n_customers=n_customers, instance_id=1,
              n_vehicles=n_vehicles, endurance_type=endurance, use_drones=True)
    return loader.load_rc1_instance(**kw) if rc_type == "RC1" else loader.load_rc2_instance(**kw)

# ?? Configs ??
CONFIGS = [
    {"key": "25_RC1_medium", "n": 25, "rc": "RC1", "v": 2, "end": "medium"},
    {"key": "25_RC1_high",   "n": 25, "rc": "RC1", "v": 2, "end": "high"},
    {"key": "25_RC2_medium", "n": 25, "rc": "RC2", "v": 2, "end": "medium"},
    {"key": "25_RC2_high",   "n": 25, "rc": "RC2", "v": 2, "end": "high"},
    {"key": "50_RC1_medium", "n": 50, "rc": "RC1", "v": 4, "end": "medium"},
    {"key": "50_RC1_high",   "n": 50, "rc": "RC1", "v": 4, "end": "high"},
    {"key": "100_RC1_medium", "n": 100, "rc": "RC1", "v": 10, "end": "medium"},
    {"key": "100_RC1_high",   "n": 100, "rc": "RC1", "v": 10, "end": "high"},
    {"key": "100_RC2_medium", "n": 100, "rc": "RC2", "v": 10, "end": "medium"},
    {"key": "100_RC2_high",   "n": 100, "rc": "RC2", "v": 10, "end": "high"},
]

PARAMS = {
    "25": {"paco": {"n_ants": 30, "max_iter": 25},
           "nsga2": {"pop_size": 50, "max_gen": 30},
           "w6": {"max_iter": 15}},
    "50": {"paco": {"n_ants": 20, "max_iter": 15},
           "nsga2": {"pop_size": 40, "max_gen": 20},
           "w6": {"max_iter": 10}},
    "100": {"paco": {"n_ants": 15, "max_iter": 8},
           "nsga2": {"pop_size": 30, "max_gen": 12},
           "w6": {"max_iter": 3}},
}

def run_algo(algo_name, cfg, run_id, conn=None):
    """Run a single algorithm trial. Returns result dict."""
    global _PACO_IMP2, _NSGA2_IMP, _W6
    p2, n2, w6 = get_algo_classes()
    params = PARAMS[str(cfg["n"])]
    model = load_model(cfg["n"], cfg["rc"], cfg["v"], cfg["end"])
    t0 = time.time()
    if algo_name == "PACO-imp2":
        algo = p2(model, **params["paco"])
        sols, pf = algo.solve()
    elif algo_name == "NSGA2-imp":
        algo = n2(model, **params["nsga2"])
        sols, pf = algo.solve()
    else:
        algo = w6(model, **params["w6"])
        sols, pf = algo.solve()
    elapsed = time.time() - t0
    costs, tards, sol_objs = [], [], []
    for sol in sols[:100]:
        c, _ = model.evaluate_solution(sol)
        t = model.calculate_pure_tardiness(sol)
        costs.append(float(c)); tards.append(float(t)); sol_objs.append((float(c), float(t)))
    # Recompute Pareto front
    true_pf, true_sols = [], []
    for i in range(len(sol_objs)):
        if not any(sol_objs[j][0] <= sol_objs[i][0] and sol_objs[j][1] <= sol_objs[i][1] and
                   (sol_objs[j][0] < sol_objs[i][0] or sol_objs[j][1] < sol_objs[i][1])
                   for j in range(len(sol_objs)) if i != j):
            true_pf.append(sol_objs[i]); true_sols.append(sols[i])
    # Save routes serializably
    def routes_to_dict(rts):
        return [{"vehicle_id": r.vehicle_id, "customers": r.customers,
                 "drone_missions": [{"drone_id": m.drone_id, "customer_ids": m.customer_ids,
                                     "launch_point": m.launch_point, "return_point": m.return_point}
                                    for m in r.drone_missions]} for r in rts]
    dr = sum(1 for sol in true_sols if any(m for r in sol for m in r.drone_missions)) / max(1, len(true_sols))
    dm = sum(len(r.drone_missions) for sol in true_sols for r in sol)
    result = {"algo": algo_name, "config": cfg["key"], "run": run_id,
              "time": round(elapsed, 2), "n_solutions": len(true_sols),
              "costs": costs, "tardiness": tards,
              "pareto_front": true_pf,
              "representative": {"min_cost": routes_to_dict(true_sols[np.argmin([o[0] for o in true_pf])]) if true_pf else [],
                                 "min_tard": routes_to_dict(true_sols[np.argmin([o[1] for o in true_pf])]) if true_pf else []},
              "drone_ratio": dr, "drone_missions": dm}
    if conn: conn.send(result)
    return result

def run_config(cfg, runs, fast=False):
    """Run all 3 algorithms on one config."""
    results = {}
    for algo_name in ["PACO-imp2", "NSGA2-imp", "PACO+ALNS-W6"]:
        print(f"\n  [{cfg['key']}] {algo_name}", flush=True)
        trials = []
        for r in range(runs):
            print(f"    Run {r+1}/{runs}...", end="", flush=True)
            res = run_algo(algo_name, cfg, r)
            trials.append(res)
            print(f" {res['time']:.1f}s, {res['n_solutions']} sols", flush=True)
        # Aggregate
        all_c = [x for t in trials for x in t["costs"]]
        all_t = [x for t in trials for x in t["tardiness"]]
        all_pf = [pt for t in trials for pt in t["pareto_front"]]
        ref = (max(p[0] for p in all_pf) * 1.1, max(p[1] for p in all_pf) * 1.1) if all_pf else (200, 200)
        def hv_func(pf):
            if not pf: return 0.0
            pts = np.array(pf)
            pts = pts[np.lexsort((pts[:,1], pts[:,0]))]
            hv, px = 0.0, ref[0]
            for i in range(len(pts)-1, -1, -1):
                x, y = pts[i]
                hv += max(0, px - x) * max(0, ref[1] - y); px = x
            return hv
        hv_val = np.mean([hv_func(t["pareto_front"]) for t in trials])
        times = [t["time"] for t in trials]
        dr = np.mean([t["drone_ratio"] for t in trials])
        results[algo_name] = {"mean_cost": float(np.mean(all_c)), "std_cost": float(np.std(all_c)),
                              "mean_tard": float(np.mean(all_t)), "std_tard": float(np.std(all_t)),
                              "mean_time": float(np.mean(times)),
                              "hypervolume": float(hv_val),
                              "drone_ratio": float(dr),
                              "trials": trials}
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", type=str, default=None,
                        help="Comma-separated, e.g. 25_RC1_medium,50_RC1_high")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--fast", action="store_true", help="Quick test (1 run, 25c only)")
    parser.add_argument("--out", type=str, default="results/comparison")
    args = parser.parse_args()

    out_dir = os.path.join(BASE, args.out)
    os.makedirs(out_dir, exist_ok=True)

    configs = CONFIGS
    if args.configs:
        keys = [k.strip() for k in args.configs.split(",")]
        configs = [c for c in configs if c["key"] in keys]
    if args.fast:
        configs = [c for c in configs if c["n"] == 25][:1]
        args.runs = 1

    json_path = os.path.join(out_dir, "raw_results.json")
    all_data = {}
    for cfg in configs:
        print(f"\n{'='*60}")
        print(f"Processing: {cfg['key']}")
        print(f"{'='*60}")
        all_data[cfg["key"]] = run_config(cfg, args.runs, args.fast)
    with open(json_path, "w") as f:
        json.dump(all_data, f, indent=2, default=str)

    json_path = os.path.join(out_dir, "raw_results.json")
    with open(json_path, "w") as f:
        json.dump(all_data, f, indent=2, default=str)
    print(f"\nRaw results saved: {json_path}")
    print("\nDone! Run compare_plot.py to generate plots.")

if __name__ == "__main__":
    mp.freeze_support()
    import platform
    if platform.python_implementation() == "PyPy":
        import tempfile
        deap_dir = os.path.join(tempfile.gettempdir(), "deap")
        if os.path.isdir(deap_dir):
            sys.path.insert(0, tempfile.gettempdir())
    main()
