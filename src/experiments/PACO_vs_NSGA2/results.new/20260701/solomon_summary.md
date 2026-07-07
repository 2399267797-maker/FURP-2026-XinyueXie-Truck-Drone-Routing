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
| 25c_RC101_medium | 25 | RC1 | 2T+2D | medium | P-ACO | 278.48 ± 7.99 | 229.53 ± 110.83 |
| 25c_RC101_medium | 25 | RC1 | 2T+2D | medium | NSGA-II | 314.08 ± 23.34 | 142.82 ± 114.14 |
| 25c_RC101_medium | 25 | RC1 | 2T+0D | - | No-Drone | 328.71 ± 26.18 | 212.24 ± 177.99 |
| 25c_RC101_high | 25 | RC1 | 2T+2D | high | P-ACO | 279.72 ± 10.45 | 229.44 ± 114.64 |
| 25c_RC101_high | 25 | RC1 | 2T+2D | high | NSGA-II | 321.71 ± 28.14 | 190.83 ± 136.05 |
| 25c_RC101_high | 25 | RC1 | 2T+0D | - | No-Drone | 327.04 ± 27.46 | 238.52 ± 178.39 |
| 25c_RC201_medium | 25 | RC2 | 2T+2D | medium | P-ACO | 277.90 ± 7.09 | 1820.90 ± 889.06 |
| 25c_RC201_medium | 25 | RC2 | 2T+2D | medium | NSGA-II | 331.48 ± 35.85 | 979.44 ± 1017.94 |
| 25c_RC201_medium | 25 | RC2 | 2T+0D | - | No-Drone | 348.49 ± 40.56 | 1065.70 ± 1058.13 |
| 25c_RC201_high | 25 | RC2 | 2T+2D | high | P-ACO | 279.08 ± 8.42 | 1593.38 ± 726.86 |
| 25c_RC201_high | 25 | RC2 | 2T+2D | high | NSGA-II | 336.17 ± 32.65 | 817.51 ± 897.20 |
| 25c_RC201_high | 25 | RC2 | 2T+0D | - | No-Drone | 339.68 ± 33.09 | 1171.20 ± 1125.64 |
| 50c_RC101_medium | 50 | RC1 | 4T+4D | medium | P-ACO | 549.37 ± 17.20 | 592.07 ± 198.46 |
| 50c_RC101_medium | 50 | RC1 | 4T+4D | medium | NSGA-II | 739.97 ± 32.41 | 589.60 ± 255.17 |
| 50c_RC101_medium | 50 | RC1 | 4T+0D | - | No-Drone | 786.71 ± 40.81 | 1015.56 ± 372.24 |
| 50c_RC101_high | 50 | RC1 | 4T+4D | high | P-ACO | 552.84 ± 21.84 | 642.72 ± 206.24 |
| 50c_RC101_high | 50 | RC1 | 4T+4D | high | NSGA-II | 742.58 ± 29.59 | 666.30 ± 232.69 |
| 50c_RC101_high | 50 | RC1 | 4T+0D | - | No-Drone | 781.22 ± 41.94 | 847.76 ± 331.54 |
| 50c_RC201_medium | 50 | RC2 | 4T+4D | medium | P-ACO | 549.12 ± 20.40 | 3962.98 ± 1024.59 |
| 50c_RC201_medium | 50 | RC2 | 4T+4D | medium | NSGA-II | 784.99 ± 81.97 | 1955.66 ± 1285.16 |
| 50c_RC201_medium | 50 | RC2 | 4T+0D | - | No-Drone | 858.61 ± 107.00 | 2876.50 ± 1804.08 |
| 50c_RC201_high | 50 | RC2 | 4T+4D | high | P-ACO | 557.96 ± 28.57 | 3976.45 ± 1044.63 |
| 50c_RC201_high | 50 | RC2 | 4T+4D | high | NSGA-II | 806.08 ± 75.39 | 1917.31 ± 1296.91 |
| 50c_RC201_high | 50 | RC2 | 4T+0D | - | No-Drone | 876.74 ± 108.92 | 3426.82 ± 2145.76 |
| 50c_RC101_medium | 50 | RC1 | 6T+6D | medium | P-ACO | 682.13 ± 25.71 | 391.06 ± 148.44 |
| 50c_RC101_medium | 50 | RC1 | 6T+6D | medium | NSGA-II | 969.93 ± 47.15 | 387.65 ± 250.41 |
| 50c_RC101_medium | 50 | RC1 | 6T+0D | - | No-Drone | 1025.09 ± 59.54 | 511.26 ± 293.14 |
| 50c_RC101_high | 50 | RC1 | 6T+6D | high | P-ACO | 680.03 ± 26.91 | 435.32 ± 151.96 |
| 50c_RC101_high | 50 | RC1 | 6T+6D | high | NSGA-II | 986.01 ± 59.95 | 374.57 ± 198.26 |
| 50c_RC101_high | 50 | RC1 | 6T+0D | - | No-Drone | 1034.50 ± 63.21 | 510.42 ± 292.47 |
| 50c_RC201_medium | 50 | RC2 | 6T+6D | medium | P-ACO | 678.70 ± 25.00 | 3036.73 ± 1086.71 |
| 50c_RC201_medium | 50 | RC2 | 6T+6D | medium | NSGA-II | 989.07 ± 72.33 | 1230.40 ± 916.41 |
| 50c_RC201_medium | 50 | RC2 | 6T+0D | - | No-Drone | 1082.23 ± 103.00 | 2355.23 ± 1731.83 |
| 50c_RC201_high | 50 | RC2 | 6T+6D | high | P-ACO | 690.51 ± 34.27 | 3079.09 ± 1043.41 |
| 50c_RC201_high | 50 | RC2 | 6T+6D | high | NSGA-II | 992.65 ± 71.90 | 1324.24 ± 1181.50 |
| 50c_RC201_high | 50 | RC2 | 6T+0D | - | No-Drone | 1053.55 ± 93.97 | 2202.98 ± 1680.06 |
