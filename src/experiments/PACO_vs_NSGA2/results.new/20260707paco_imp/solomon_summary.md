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
| 25c_RC101_medium | 25 | RC1 | 2T+2D | medium | P-ACO | 278.92 ± 8.09 | 513.51 ± 232.44 |
| 25c_RC101_medium | 25 | RC1 | 2T+2D | medium | NSGA-II | 325.84 ± 25.43 | 431.20 ± 368.67 |
| 25c_RC101_medium | 25 | RC1 | 2T+0D | - | No-Drone | 328.57 ± 26.60 | 468.81 ± 381.82 |
| 25c_RC101_high | 25 | RC1 | 2T+2D | high | P-ACO | 277.82 ± 6.16 | 479.95 ± 221.92 |
| 25c_RC101_high | 25 | RC1 | 2T+2D | high | NSGA-II | 309.10 ± 21.64 | 295.50 ± 251.28 |
| 25c_RC101_high | 25 | RC1 | 2T+0D | - | No-Drone | 332.39 ± 33.18 | 495.83 ± 328.65 |
| 25c_RC201_medium | 25 | RC2 | 2T+2D | medium | P-ACO | 278.21 ± 8.95 | 1726.45 ± 815.82 |
| 25c_RC201_medium | 25 | RC2 | 2T+2D | medium | NSGA-II | 325.84 ± 28.19 | 950.73 ± 942.95 |
| 25c_RC201_medium | 25 | RC2 | 2T+0D | - | No-Drone | 352.90 ± 45.95 | 1289.46 ± 1198.79 |
| 25c_RC201_high | 25 | RC2 | 2T+2D | high | P-ACO | 279.45 ± 8.36 | 1768.27 ± 924.25 |
| 25c_RC201_high | 25 | RC2 | 2T+2D | high | NSGA-II | 336.31 ± 37.93 | 694.76 ± 697.24 |
| 25c_RC201_high | 25 | RC2 | 2T+0D | - | No-Drone | 344.69 ± 41.34 | 1381.56 ± 1253.46 |
| 50c_RC101_medium | 50 | RC1 | 4T+4D | medium | P-ACO | 550.70 ± 19.01 | 1199.83 ± 358.75 |
| 50c_RC101_medium | 50 | RC1 | 4T+4D | medium | NSGA-II | 752.55 ± 40.91 | 1271.22 ± 420.49 |
| 50c_RC101_medium | 50 | RC1 | 4T+0D | - | No-Drone | 779.04 ± 42.45 | 1638.37 ± 682.22 |
| 50c_RC101_high | 50 | RC1 | 4T+4D | high | P-ACO | 552.90 ± 23.11 | 1090.53 ± 332.22 |
| 50c_RC101_high | 50 | RC1 | 4T+4D | high | NSGA-II | 747.42 ± 40.36 | 1094.91 ± 411.05 |
| 50c_RC101_high | 50 | RC1 | 4T+0D | - | No-Drone | 786.72 ± 46.30 | 1857.42 ± 634.61 |
| 50c_RC201_medium | 50 | RC2 | 4T+4D | medium | P-ACO | 549.96 ± 20.21 | 4141.39 ± 1035.60 |
| 50c_RC201_medium | 50 | RC2 | 4T+4D | medium | NSGA-II | 791.91 ± 65.97 | 2165.85 ± 1424.46 |
| 50c_RC201_medium | 50 | RC2 | 4T+0D | - | No-Drone | 856.18 ± 90.83 | 2668.52 ± 1464.72 |
| 50c_RC201_high | 50 | RC2 | 4T+4D | high | P-ACO | 554.09 ± 28.63 | 4118.06 ± 1079.65 |
| 50c_RC201_high | 50 | RC2 | 4T+4D | high | NSGA-II | 794.97 ± 81.17 | 1849.18 ± 1507.19 |
| 50c_RC201_high | 50 | RC2 | 4T+0D | - | No-Drone | 871.23 ± 100.41 | 3009.22 ± 1898.82 |
| 50c_RC101_medium | 50 | RC1 | 6T+6D | medium | P-ACO | 681.75 ± 24.96 | 825.34 ± 343.60 |
| 50c_RC101_medium | 50 | RC1 | 6T+6D | medium | NSGA-II | 975.26 ± 48.13 | 694.91 ± 350.10 |
| 50c_RC101_medium | 50 | RC1 | 6T+0D | - | No-Drone | 1039.14 ± 77.43 | 1032.05 ± 558.99 |
| 50c_RC101_high | 50 | RC1 | 6T+6D | high | P-ACO | 687.80 ± 31.19 | 810.14 ± 292.59 |
| 50c_RC101_high | 50 | RC1 | 6T+6D | high | NSGA-II | 975.88 ± 44.20 | 534.46 ± 295.68 |
| 50c_RC101_high | 50 | RC1 | 6T+0D | - | No-Drone | 1036.86 ± 60.10 | 971.43 ± 648.95 |
| 50c_RC201_medium | 50 | RC2 | 6T+6D | medium | P-ACO | 680.46 ± 27.39 | 2881.84 ± 1014.34 |
| 50c_RC201_medium | 50 | RC2 | 6T+6D | medium | NSGA-II | 997.39 ± 76.20 | 1304.70 ± 1016.41 |
| 50c_RC201_medium | 50 | RC2 | 6T+0D | - | No-Drone | 1086.60 ± 89.46 | 2280.68 ± 1681.83 |
| 50c_RC201_high | 50 | RC2 | 6T+6D | high | P-ACO | 697.69 ± 34.78 | 2950.95 ± 960.21 |
| 50c_RC201_high | 50 | RC2 | 6T+6D | high | NSGA-II | 1018.72 ± 67.91 | 1495.96 ± 1214.68 |
| 50c_RC201_high | 50 | RC2 | 6T+0D | - | No-Drone | 1075.65 ± 96.32 | 2569.90 ± 1669.19 |
