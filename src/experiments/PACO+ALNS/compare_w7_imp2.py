"""Compare fixed PACO+ALNS W7 results against the imp2 baseline.

Usage:
    python compare_w7_imp2.py [--w7 results/20260802_w7_fixed/w7_results.json]
"""
import os
import sys
import json
import argparse


BASE = os.path.dirname(os.path.abspath(__file__))


def w7_key(r):
    return f"{r['n_customers']}c_RC{r['rc_type'][-1]}{r['instance_id']}_{r['n_vehicles']}V_{r['endurance_type']}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--w7", type=str,
                        default=os.path.join(BASE, "results", "20260802_w7_fixed", "w7_results.json"))
    parser.add_argument("--imp2", type=str,
                        default=os.path.join(BASE, "results", "new2", "imp2_results.json"))
    args = parser.parse_args()

    with open(args.w7, encoding="utf-8") as f:
        w7 = json.load(f)
    with open(args.imp2, encoding="utf-8") as f:
        imp2 = json.load(f)

    imp2_by_key = {r["config"]: r for r in imp2}
    rows = []
    for r in w7:
        key = w7_key(r)
        i = imp2_by_key.get(key)
        if i is None:
            continue
        rows.append((key, r, i))

    header = (f"{'config':24s} {'W7 cost':>9s} {'imp2':>8s} {'dC%':>7s} "
              f"{'W7 tard':>9s} {'imp2':>8s} {'dT%':>7s} {'W7 t':>6s} {'imp2':>6s} "
              f"{'drone':>5s} {'imp2':>5s} {'sol':>4s} {'imp2':>5s} {'miss':>5s} {'ovr':>5s}")
    print(header)
    print("-" * len(header))
    for key, r, i in rows:
        dc = (r["mean_cost"] - i["cost"]) / i["cost"] * 100
        dt = (r["mean_tardiness"] - i["tard"]) / i["tard"] * 100
        print(f"{key:24s} {r['mean_cost']:9.1f} {i['cost']:8.1f} {dc:7.1f} "
              f"{r['mean_tardiness']:9.1f} {i['tard']:8.1f} {dt:7.1f} "
              f"{r['mean_solve_time']:6.1f} {i['time']:6.1f} "
              f"{r['mean_drone_missions']:5.1f} {i['drone']:5.1f} "
              f"{r['mean_n_solutions']:4.1f} {i['sols']:5d} "
              f"{r['n_missing_solutions']:5d} {r['n_overload_solutions']:5d}")

    print(f"\nTotal configs matched: {len(rows)}")


if __name__ == "__main__":
    main()
