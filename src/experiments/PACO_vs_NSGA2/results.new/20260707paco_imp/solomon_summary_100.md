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
| 100c_RC101_medium | 100 | RC1 | 10T+10D | medium | P-ACO | 1321.95 ± 45.32 | 2561.28 ± 785.35 |
| 100c_RC101_medium | 100 | RC1 | 10T+10D | medium | NSGA-II | 1782.69 ± 55.47 | 2802.19 ± 794.98 |
| 100c_RC101_medium | 100 | RC1 | 10T+0D | - | No-Drone | 1863.27 ± 89.06 | 3903.33 ± 1339.47 |
| 100c_RC101_high | 100 | RC1 | 10T+10D | high | P-ACO | 1360.54 ± 46.57 | 2551.65 ± 900.35 |
| 100c_RC101_high | 100 | RC1 | 10T+10D | high | NSGA-II | 1776.66 ± 60.64 | 2774.12 ± 977.94 |
| 100c_RC101_high | 100 | RC1 | 10T+0D | - | No-Drone | 1866.50 ± 87.00 | 3892.60 ± 1446.49 |
| 100c_RC201_medium | 100 | RC2 | 10T+10D | medium | P-ACO | 1350.88 ± 49.08 | 7013.42 ± 2019.21 |
| 100c_RC201_medium | 100 | RC2 | 10T+10D | medium | NSGA-II | 1840.88 ± 89.63 | 5409.21 ± 2671.11 |
| 100c_RC201_medium | 100 | RC2 | 10T+0D | - | No-Drone | 1913.81 ± 113.79 | 6797.48 ± 3241.92 |
| 100c_RC201_high | 100 | RC2 | 10T+10D | high | P-ACO | 1379.34 ± 49.72 | 6814.14 ± 1938.44 |
| 100c_RC201_high | 100 | RC2 | 10T+10D | high | NSGA-II | 1857.50 ± 103.88 | 5406.65 ± 2802.88 |
| 100c_RC201_high | 100 | RC2 | 10T+0D | - | No-Drone | 1948.51 ± 123.69 | 7800.22 ± 3752.06 |
