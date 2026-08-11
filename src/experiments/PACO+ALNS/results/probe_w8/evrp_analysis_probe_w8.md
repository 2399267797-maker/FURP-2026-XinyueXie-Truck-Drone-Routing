# E-CVRP Benchmark Validation (NSGA-II / PACO-imp2 / PACO+ALNS W8)

## Mapping
- E-CVRP has no time windows: solvers receive wide [0, 1e9] windows (tardiness = 0).
- Drones disabled (capacity 0, range 0); truck fixed cost 0, variable cost 1, so cost = distance.
- Energy/recharging constraints are not part of these solvers and are not validated here.

| Instance | Opt/BKS | Algo | Best Cost | Gap % | Feasible sols | Missing | Overload | Time (s) |
|----------|---------|------|-----------|-------|---------------|---------|----------|----------|
| E-n29-k4-s7 | 383 | nsga2 | nan | nan | 0.0 | 0 | 12 | 0.5 |
| E-n29-k4-s7 | 383 | imp2 | 533.00 | 39.17 | 1.0 | 0 | 0 | 0.4 |
| E-n29-k4-s7 | 383 | w8 | 390.64 | 2.00 | 1.0 | 0 | 0 | 2.2 |

| Algo | Mean gap % vs best-known | Instances at best-known |
|------|--------------------------|-------------------------|
| nsga2 | nan | 0/1 |
| imp2 | 39.17 | 0/1 |
| w8 | 2.00 | 0/1 |

Note: NSGA-II ignores truck capacity in its decoder; its solutions are included only when feasible after evaluation.
