# Truck-Drone Routing (TDRPTW)

Compares four multi-objective algorithms for the truck-drone routing problem with time windows (TDRPTW): NSGA-II, PACO, PACO+ALNS (W8), and a pure ALNS baseline, on the Solomon and WCCI-2020 E-CVRP benchmarks.

```
src/
├── experiments/
│   ├── ALNS/                       # Pure ALNS solver and four-algorithm Solomon comparison
│   │   ├── pure_alns.py
│   │   ├── compare_pure_alns.py
│   │   └── results/
│   ├── PACO+ALNS/                  # PACO+ALNS W8, three-algorithm comparison, E-CVRP validation
│   │   ├── PACO+ALNSW8.py
│   │   ├── compare_three_algorithms.py
│   │   ├── evrp_validation.py
│   │   └── results/                # 20260809_w8, 20260812_4alg, 20260806_evrp_w8
│   ├── PACO/                       # PACO (PACO-imp2)
│   ├── NSGA2/                      # NSGA-II
│   ├── PACO_vs_NSGA2/              # shared model, Solomon data, visualization
│   ├── e-cvrp_benchmark_instances/ # WCCI-2020 E-CVRP instances
│   ├── ETRD-NL/                    # MILP + ALNS baseline
│   ├── CVRP_POMO/                  # RL baseline
│   ├── E-VRPTW/                    # GA VRPTW baseline
│   ├── py-ga-VRPTW/
│   └── report.md                   # full report
└── docs/
    └── src/                        # weekly reports
```
