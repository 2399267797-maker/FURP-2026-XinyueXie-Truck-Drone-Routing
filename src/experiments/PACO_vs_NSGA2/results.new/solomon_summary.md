# P-ACO vs NSGA-II Results (Solomon RC Benchmark)

## Experimental Setup
- RC1: Short scheduling horizon (tight time windows)
- RC2: Long scheduling horizon (wide time windows)
- Linear scaling: Solomon [0,100] -> Urban [0,12] km
- Drone endurance: medium (4km), high (6km)
- Repetitions per experiment: 10
- Metrics: Mean Cost ± Std (averaged across all Pareto solutions from all runs)

## Results

| Config | Customers | RC Type | Vehicles | Endurance | Method | Mean Cost ± Std | Mean Tardiness ± Std |
|--------|-----------|---------|----------|-----------|--------|-----------------|----------------------|
| 100c_RC101_medium | 100 | RC1 | 10T+10D | medium | P-ACO | 1320.00 ± 33.77 | 2972.50 ± 929.38 |
| 100c_RC101_medium | 100 | RC1 | 10T+10D | medium | NSGA-II | 1802.61 ± 75.82 | 3376.60 ± 1214.36 |
| 100c_RC101_medium | 100 | RC1 | 10T+0D | - | No-Drone | 1877.00 ± 87.34 | 3896.79 ± 1297.75 |
| 100c_RC101_high | 100 | RC1 | 10T+10D | high | P-ACO | 1361.22 ± 40.73 | 2654.42 ± 770.52 |
| 100c_RC101_high | 100 | RC1 | 10T+10D | high | NSGA-II | 1803.08 ± 53.45 | 3131.42 ± 836.60 |
| 100c_RC101_high | 100 | RC1 | 10T+0D | - | No-Drone | 1859.66 ± 84.65 | 3782.15 ± 1377.08 |
| 100c_RC201_medium | 100 | RC2 | 10T+10D | medium | P-ACO | 1338.84 ± 38.61 | 7662.91 ± 2010.94 |
| 100c_RC201_medium | 100 | RC2 | 10T+10D | medium | NSGA-II | 1862.28 ± 109.02 | 5178.61 ± 2895.91 |
| 100c_RC201_medium | 100 | RC2 | 10T+0D | - | No-Drone | 1921.07 ± 115.58 | 8078.29 ± 3703.09 |
| 100c_RC201_high | 100 | RC2 | 10T+10D | high | P-ACO | 1379.76 ± 47.67 | 6341.87 ± 1873.65 |
| 100c_RC201_high | 100 | RC2 | 10T+10D | high | NSGA-II | 1848.66 ± 96.27 | 5097.16 ± 2300.23 |
| 100c_RC201_high | 100 | RC2 | 10T+0D | - | No-Drone | 1941.46 ± 121.29 | 7616.16 ± 3472.30 |
