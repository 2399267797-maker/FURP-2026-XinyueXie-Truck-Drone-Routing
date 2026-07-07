# PACO-imp vs PACO+ALNS Comparison (Solomon RC Benchmark)

## Experimental Setup
- RC1: Short scheduling horizon (tight time windows)
- RC2: Long scheduling horizon (wide time windows)
- Linear scaling: Solomon [0,100] -> Urban [0,12] km
- Repetitions per experiment: 10

## Results

| Config | Customers | RC Type | Vehicles | Endurance | Method | Mean Cost ± Std | Mean Tardiness ± Std | HV | Drone Ratio |
|--------|-----------|---------|----------|-----------|--------|-----------------|----------------------|----|-------------|
| 25c_RC101_medium | 25 | RC1 | 2T+2D | medium | PACO-imp | 277.30 ± 6.86 | 287.42 ± 139.26 | 46070.3 | 71% |
| 25c_RC101_medium | 25 | RC1 | 2T+2D | medium | PACO+ALNS | 271.71 ± 7.49 | 296.69 ± 150.08 | 49775.3 | 34% |
| 25c_RC101_high | 25 | RC1 | 2T+2D | high | PACO-imp | 279.40 ± 8.48 | 234.80 ± 126.38 | 37343.0 | 75% |
| 25c_RC101_high | 25 | RC1 | 2T+2D | high | PACO+ALNS | 272.30 ± 8.24 | 250.82 ± 122.72 | 39801.7 | 28% |
| 25c_RC201_medium | 25 | RC2 | 2T+2D | medium | PACO-imp | 278.99 ± 9.31 | 1920.91 ± 919.26 | 296022.0 | 73% |
| 25c_RC201_medium | 25 | RC2 | 2T+2D | medium | PACO+ALNS | 272.45 ± 8.92 | 1943.62 ± 881.70 | 312940.7 | 26% |
| 25c_RC201_high | 25 | RC2 | 2T+2D | high | PACO-imp | 278.87 ± 8.30 | 1498.29 ± 714.12 | 335599.2 | 69% |
| 25c_RC201_high | 25 | RC2 | 2T+2D | high | PACO+ALNS | 275.74 ± 13.98 | 1532.94 ± 755.45 | 344553.1 | 32% |
| 50c_RC101_medium | 50 | RC1 | 4T+4D | medium | PACO-imp | 547.46 ± 17.92 | 646.09 ± 177.06 | 131336.5 | 100% |
| 50c_RC101_medium | 50 | RC1 | 4T+4D | medium | PACO+ALNS | 544.69 ± 25.88 | 639.60 ± 191.00 | 136686.1 | 58% |
| 50c_RC101_high | 50 | RC1 | 4T+4D | high | PACO-imp | 552.16 ± 20.88 | 577.93 ± 186.70 | 156994.8 | 93% |
| 50c_RC101_high | 50 | RC1 | 4T+4D | high | PACO+ALNS | 543.56 ± 26.72 | 660.25 ± 202.43 | 160926.6 | 59% |
| 50c_RC201_medium | 50 | RC2 | 4T+4D | medium | PACO-imp | 549.91 ± 19.24 | 3667.98 ± 912.20 | 904966.8 | 99% |
| 50c_RC201_medium | 50 | RC2 | 4T+4D | medium | PACO+ALNS | 546.56 ± 28.27 | 3891.83 ± 914.16 | 911736.5 | 67% |
| 50c_RC201_high | 50 | RC2 | 4T+4D | high | PACO-imp | 555.46 ± 26.76 | 4107.30 ± 1117.90 | 869267.6 | 98% |
| 50c_RC201_high | 50 | RC2 | 4T+4D | high | PACO+ALNS | 554.13 ± 32.72 | 3885.49 ± 983.28 | 895995.6 | 71% |
| 50c_RC101_medium | 50 | RC1 | 6T+6D | medium | PACO-imp | 678.08 ± 19.79 | 449.66 ± 155.58 | 114695.3 | 100% |
| 50c_RC101_medium | 50 | RC1 | 6T+6D | medium | PACO+ALNS | 664.97 ± 31.66 | 463.33 ± 148.31 | 128123.1 | 70% |
| 50c_RC101_high | 50 | RC1 | 6T+6D | high | PACO-imp | 683.95 ± 25.51 | 453.79 ± 156.06 | 121209.1 | 99% |
| 50c_RC101_high | 50 | RC1 | 6T+6D | high | PACO+ALNS | 666.96 ± 33.57 | 498.63 ± 150.91 | 132112.6 | 74% |
| 50c_RC201_medium | 50 | RC2 | 6T+6D | medium | PACO-imp | 683.47 ± 25.08 | 2986.76 ± 1057.25 | 877815.2 | 100% |
| 50c_RC201_medium | 50 | RC2 | 6T+6D | medium | PACO+ALNS | 672.11 ± 38.19 | 3187.20 ± 1010.41 | 906713.8 | 77% |
| 50c_RC201_high | 50 | RC2 | 6T+6D | high | PACO-imp | 688.25 ± 35.65 | 3334.78 ± 1161.64 | 1323565.2 | 100% |
| 50c_RC201_high | 50 | RC2 | 6T+6D | high | PACO+ALNS | 678.93 ± 40.77 | 3515.58 ± 1087.97 | 1339000.0 | 74% |
