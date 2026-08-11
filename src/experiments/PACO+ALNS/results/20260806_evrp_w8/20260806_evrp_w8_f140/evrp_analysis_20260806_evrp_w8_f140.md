# E-CVRP Benchmark Validation (NSGA-II / PACO-imp2 / PACO+ALNS W8)

## Mapping
- E-CVRP has no time windows: solvers receive wide [0, 1e9] windows (tardiness = 0).
- Drones disabled (capacity 0, range 0); truck fixed cost 0, variable cost 1, so cost = distance.
- Energy/recharging constraints are not part of these solvers and are not validated here.

| Instance | Opt/BKS | Algo | Best Cost | Gap % | Feasible sols | Missing | Overload | Time (s) |
|----------|---------|------|-----------|-------|---------------|---------|----------|----------|
| F-n140-k5-s5 | - | nsga2 | nan | - | 0.0 | 0 | 19 | 14.8 |
| F-n140-k5-s5 | - | imp2 | nan | - | 0.0 | 3 | 0 | 91.4 |
| F-n140-k5-s5 | - | w8 | nan | - | 0.0 | 0 | 0 | 388.1 |

| Algo | Mean gap % vs best-known | Instances at best-known |
|------|--------------------------|-------------------------|
| nsga2 | nan | 0/1 |
| imp2 | nan | 0/1 |
| w8 | nan | 0/1 |

Note: NSGA-II ignores truck capacity in its decoder; its solutions are included only when feasible after evaluation.
