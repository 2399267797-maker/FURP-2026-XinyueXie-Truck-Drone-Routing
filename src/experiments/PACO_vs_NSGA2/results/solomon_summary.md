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
| 25c_RC101_medium | 25 | RC1 | 2T+2D | medium | P-ACO | 282.60 ± 8.51 | 331.69 ± 153.97 |
| 25c_RC101_medium | 25 | RC1 | 2T+2D | medium | NSGA-II | 364.21 ± 26.70 | 433.88 ± 195.93 |
| 25c_RC101_medium | 25 | RC1 | 2T+0D | - | No-Drone | 372.39 ± 21.06 | 499.43 ± 171.06 |
| 25c_RC101_high | 25 | RC1 | 2T+2D | high | P-ACO | 286.25 ± 10.60 | 238.50 ± 131.13 |
| 25c_RC101_high | 25 | RC1 | 2T+2D | high | NSGA-II | 355.82 ± 28.73 | 372.09 ± 165.78 |
| 25c_RC101_high | 25 | RC1 | 2T+0D | - | No-Drone | 374.26 ± 36.95 | 533.20 ± 238.05 |
| 25c_RC201_medium | 25 | RC2 | 2T+2D | medium | P-ACO | 284.10 ± 10.55 | 1779.79 ± 906.04 |
| 25c_RC201_medium | 25 | RC2 | 2T+2D | medium | NSGA-II | 379.14 ± 39.40 | 1499.90 ± 1044.29 |
| 25c_RC201_medium | 25 | RC2 | 2T+0D | - | No-Drone | 401.70 ± 39.96 | 1967.23 ± 1066.45 |
| 25c_RC201_high | 25 | RC2 | 2T+2D | high | P-ACO | 286.20 ± 10.95 | 1762.54 ± 952.81 |
| 25c_RC201_high | 25 | RC2 | 2T+2D | high | NSGA-II | 378.74 ± 42.26 | 1376.92 ± 914.19 |
| 25c_RC201_high | 25 | RC2 | 2T+0D | - | No-Drone | 395.59 ± 36.84 | 1796.66 ± 979.91 |
| 50c_RC101_medium | 50 | RC1 | 4T+4D | medium | P-ACO | 550.90 ± 8.55 | 576.80 ± 107.01 |
| 50c_RC101_medium | 50 | RC1 | 4T+4D | medium | NSGA-II | 868.22 ± 47.95 | 1300.74 ± 322.89 |
| 50c_RC101_medium | 50 | RC1 | 4T+0D | - | No-Drone | 901.79 ± 41.01 | 1745.60 ± 333.86 |
| 50c_RC101_high | 50 | RC1 | 4T+4D | high | P-ACO | 562.25 ± 23.64 | 523.66 ± 151.64 |
| 50c_RC101_high | 50 | RC1 | 4T+4D | high | NSGA-II | 859.73 ± 38.53 | 1102.30 ± 303.86 |
| 50c_RC101_high | 50 | RC1 | 4T+0D | - | No-Drone | 932.34 ± 50.78 | 1668.02 ± 386.39 |
| 50c_RC201_medium | 50 | RC2 | 4T+4D | medium | P-ACO | 552.94 ± 22.74 | 4120.85 ± 976.51 |
| 50c_RC201_medium | 50 | RC2 | 4T+4D | medium | NSGA-II | 918.71 ± 70.69 | 3611.81 ± 1542.80 |
| 50c_RC201_medium | 50 | RC2 | 4T+0D | - | No-Drone | 959.95 ± 73.76 | 5029.40 ± 1681.49 |
| 50c_RC201_high | 50 | RC2 | 4T+4D | high | P-ACO | 557.19 ± 34.16 | 4390.09 ± 1029.82 |
| 50c_RC201_high | 50 | RC2 | 4T+4D | high | NSGA-II | 917.23 ± 66.13 | 3818.88 ± 1328.81 |
| 50c_RC201_high | 50 | RC2 | 4T+0D | - | No-Drone | 969.13 ± 71.61 | 5492.40 ± 1965.48 |
| 50c_RC101_medium | 50 | RC1 | 6T+6D | medium | P-ACO | 664.37 ± 10.78 | 342.99 ± 136.70 |
| 50c_RC101_medium | 50 | RC1 | 6T+6D | medium | NSGA-II | 1102.61 ± 52.55 | 872.61 ± 222.42 |
| 50c_RC101_medium | 50 | RC1 | 6T+0D | - | No-Drone | 1152.96 ± 55.32 | 1056.73 ± 284.70 |
| 50c_RC101_high | 50 | RC1 | 6T+6D | high | P-ACO | 667.66 ± 15.32 | 286.98 ± 106.00 |
| 50c_RC101_high | 50 | RC1 | 6T+6D | high | NSGA-II | 1117.33 ± 61.54 | 770.29 ± 226.13 |
| 50c_RC101_high | 50 | RC1 | 6T+0D | - | No-Drone | 1156.06 ± 50.97 | 917.06 ± 259.18 |
| 50c_RC201_medium | 50 | RC2 | 6T+6D | medium | P-ACO | 666.04 ± 19.72 | 3060.18 ± 983.26 |
| 50c_RC201_medium | 50 | RC2 | 6T+6D | medium | NSGA-II | 1143.87 ± 69.82 | 2970.42 ± 1315.44 |
| 50c_RC201_medium | 50 | RC2 | 6T+0D | - | No-Drone | 1175.58 ± 60.01 | 3854.10 ± 1450.02 |
| 50c_RC201_high | 50 | RC2 | 6T+6D | high | P-ACO | 669.06 ± 29.03 | 2853.70 ± 899.41 |
| 50c_RC201_high | 50 | RC2 | 6T+6D | high | NSGA-II | 1138.36 ± 75.86 | 3053.43 ± 1367.19 |
| 50c_RC201_high | 50 | RC2 | 6T+0D | - | No-Drone | 1176.98 ± 69.76 | 3804.40 ± 1250.40 |
