# Week 8 Progress Report: W8 vs. PACO-imp2 / NSGA-II on Solomon and E-CVRP

**Date**: 2026-08-08

**Project**: Truck-Drone Cooperative Vehicle Routing Problem (TDRP), bi-objective minimization (transportation cost, time-window violation penalty)

---

## 1. Abstract

This report summarizes the work completed in weeks 1-8, focusing on two comparison experiments: **PACO+ALNS W8** versus **PACO-imp2 (PACO)** and **NSGA-II**. The first experiment is a set of 16 bi-objective runs on the Solomon RC benchmark; the second is a validation on 24 instances of the WCCI-2020 E-CVRP benchmark.

Main findings:

1. **Solomon RC (16 configurations x 10 runs)**: PACO-imp2 achieves the lowest cost (best in 12/16 configurations), NSGA-II achieves the lowest tardiness (best in 16/16 configurations), and PACO+ALNS W8 achieves the highest hypervolume (HV) in 16/16 configurations. W8 has a mean cost of 447.35 across the 16 configurations, only about 2.4% higher than PACO-imp2, and a mean tardiness of 2936.85, between PACO-imp2 and NSGA-II.
2. **W7->W8 improvement**: with solution quality roughly unchanged or slightly better, runtime decreased significantly: 25c from 54.70s to 32.55s, 50c from 246.15s to 120.46s, and 100c from 791.13s to 506.75s.
3. **E-CVRP (24 instances)**: W8 finds a feasible solution on all 24 instances (no missing customers, no overload), and its best cost is never worse than the other two algorithms on any instance. On the 4 instances with published optimal values, the mean gap is -1.01% (CVRP relaxation basis), compared with +67.44% for NSGA-II and +9.63% for PACO-imp2.

---

## 2. Problem Definition and Experimental Setup

### 2.1 Problem Definition

The problem is a truck-drone cooperative delivery problem with time windows and drone endurance constraints. Each truck is paired one-to-one with a drone. The drone can take off from the truck en route, serve a customer, and return to the truck; each drone mission serves exactly one customer. The objective functions are:

- Objective 1 (Cost): vehicle fixed cost + truck travel cost + drone flight cost;
- Objective 2 (Tardiness): weighted sum of time-window violations across all customers.

The constraints include truck capacity, drone capacity, drone endurance, time windows, and the requirement that each customer be served exactly once.

### 2.2 Solomon Experimental Setup

| Item | Value |
|------|-------|
| Dataset | Solomon RC1 (tight time windows) / RC2 (wide time windows), instance_id=1 |
| Scale | 25c (2 vehicles), 50c (4 vehicles, 6 vehicles), 100c (10 vehicles) |
| Drone endurance | medium = 4 km, high = 6 km |
| Map | 12 x 12 km, depot at the center (6.0, 6.0) |
| Repeats | 10 per configuration |
| Iteration/generation budget | 100 |
| NSGA-II population | 100 |
| PACO-imp2 | n_ants=30 |
| PACO+ALNS W8 | adaptive parameters (scaled with the number of customers) |
| Total configurations | 16 |

### 2.3 E-CVRP Mapping and Validation Budget

The WCCI-2020 E-CVRP benchmark has no time windows and no drones. To run all three algorithms uniformly, the following mapping is used:

- Time windows are set uniformly to `[0, 1e9]`, so tardiness is always 0 and the problem reduces to single-objective cost minimization;
- Drones are disabled (capacity 0, range 0), and only trucks are used;
- Truck fixed cost is 0 and variable cost is 1, so the model cost equals total distance and can be compared with the benchmark's `OPTIMAL_VALUE`;
- Energy/recharging constraints are outside the scope of these solvers and were not validated.

The validation covers 24 instances (main validation directory), spanning four families: E (7), F (3), M (4), and X (10). The computation budget varies with scale: default instances use 30 iterations x 3 runs; supplementary medium instances (E-n112, M-n163, M-n212, X-n147) use 10 iterations x 2 runs; large X instances (X-n221 and above) use 5 iterations x 1 run.

In addition, `F-n140-k5-s5` declares `VEHICLES=5`, but its total demand of 14620 exceeds 5 x 2210, so the instance is infeasible as declared. Its filename/comment is `F-n140-k7`, so the main validation runs it with 7 vehicles. A separate `k5` probe confirmed that none of the three algorithms can find a feasible solution under the declared vehicle count.

### 2.4 Metric Definitions

- Penalties are applied to every returned solution before computing Pareto fronts, HV, and means: 10000 per missing customer and 1000 per unit of overload;
- Cost is obtained from `model.evaluate_solution`, and tardiness from `calculate_pure_tardiness`;
- The HV reference point is the maximum cost and maximum tardiness over all solutions of the three algorithms in the same configuration, each multiplied by 1.1, to keep comparisons within a configuration meaningful;
- "Mean" refers to the average over the Pareto-front points of 10 runs, not the single best solution.

---

## 3. Algorithm Methods and Main W8 Changes

### 3.1 PACO+ALNS W8

PACO+ALNS uses a hybrid "construction + improvement" framework:

1. **PACO construction (outer loop)**: each ant constructs truck routes and drone missions guided by pheromone and savings heuristics, with a three-tier fallback mechanism (convert to drone, forced insertion, missing-customer penalty);
2. **ALNS improvement (inner loop)**: 5 destroy operators (surgical/random/worst/related/route) and 3 repair operators (drone-aware/greedy/regret), selected by an adaptive-weight roulette;
3. **Multi-objective archive**: non-dominated sorting plus crowding-distance pruning; each round admits only solutions that dominate the archive or are mutually non-dominated with it;
4. **Pheromone update**: only solutions in the archive deposit pheromone, supporting separate cost/tardiness dual-objective pheromone matrices.

Main changes in W8 relative to W7 (source fix log in `PACO+ALNSW8.py`):

| Fix | Description | Effect |
|-----|-------------|--------|
| Drone ID | Drone missions generated in the ALNS phase are uniformly assigned IDs as `n_trucks + truck_id`, with launch/retrieval index validation | Fixes out-of-bounds access that silently dropped missions |
| Timeline | All fallback insertions and task-restoration paths set `_timeline_dirty=True` | Eliminates timeline-calculation crashes caused by stale reads |
| SA score guard | ALNS internal scoring adds absolute penalties for missing customers and overload | Prevents simulated annealing from accepting infeasible solutions |
| Archive filter | Solutions with missing customers or overload are excluded from the archive | Prevents infeasible solutions from polluting the Pareto front |

### 3.2 PACO-imp2

PACO-imp2 is the improved P-ACO from week 4, with six improvements: MMAS pheromone bounds, adaptive pheromone normalization, nearest-neighbor fallback, drone heuristic correction, pre/post constraint validation, and modulo-based drone ID. W8 continues to iterate on top of its robust construction.

### 3.3 NSGA-II

NSGA-II uses tournament selection, crowding distance, SBX crossover, and polynomial mutation, and searches truck-drone cooperative patterns during decoding. To adapt it to E-CVRP, capacity-aware decode repair and overload penalties were added; infeasible solutions are still retained during evolution and are penalized only at evaluation.

---

## 4. Solomon RC Comparison Results (W8)

### 4.1 Mean over 16 Configurations

| Algorithm | Mean Cost | Mean Tardiness | Mean HV | Time (s) | Front solutions | Drone missions | Routes |
|-----------|-----------|----------------|---------|----------|-----------------|----------------|--------|
| NSGA-II | 980.82 | 1907.64 | 3,286,846 | 6.63 | 37.3 | 4.42 | 5.50 |
| PACO-imp2 | 436.82 | 4474.33 | 6,804,584 | 101.27 | 13.6 | 16.41 | 2.42 |
| PACO+ALNS W8 | 447.35 | 2936.85 | 8,156,083 | 195.06 | 11.8 | 10.09 | 2.69 |

Note: the table shows the arithmetic mean over the 16 configurations (25c/50c/100c mixed) and is intended for overall trends. Per-configuration details are in `results/20260807_w8/analysis_20260807_w8.md` and the JSON files in the same directory.

### 4.2 Representative Configurations

| Config | Algorithm | Cost +/- Std | Tardiness +/- Std | HV | Time (s) |
|--------|-----------|--------------|-------------------|-----|----------|
| 25c RC101 medium | NSGA-II | 305.76 +/- 20.06 | 306.52 +/- 273.22 | 302,409 | 3.7 |
| 25c RC101 medium | PACO-imp2 | 240.05 +/- 56.55 | 1117.62 +/- 378.81 | 411,083 | 8.9 |
| 25c RC101 medium | PACO+ALNS W8 | 242.40 +/- 50.14 | 864.64 +/- 664.82 | 473,337 | 29.2 |
| 25c RC201 medium | NSGA-II | 325.54 +/- 29.49 | 716.12 +/- 729.10 | 873,858 | 3.9 |
| 25c RC201 medium | PACO-imp2 | 266.40 +/- 55.23 | 2312.10 +/- 875.21 | 988,631 | 9.5 |
| 25c RC201 medium | PACO+ALNS W8 | 257.81 +/- 42.25 | 1429.58 +/- 1083.05 | 1,271,887 | 29.0 |
| 50c RC101 4V medium | NSGA-II | 749.68 +/- 37.97 | 1424.88 +/- 521.73 | 664,965 | 5.4 |
| 50c RC101 4V medium | PACO-imp2 | 363.43 +/- 54.54 | 2498.66 +/- 382.33 | 1,300,021 | 31.1 |
| 50c RC101 4V medium | PACO+ALNS W8 | 390.93 +/- 55.48 | 1465.37 +/- 784.34 | 2,097,826 | 102.3 |
| 50c RC201 6V high | NSGA-II | 1029.83 +/- 71.29 | 1519.09 +/- 1204.34 | 4,094,721 | 6.8 |
| 50c RC201 6V high | PACO-imp2 | 418.95 +/- 74.45 | 5065.49 +/- 1682.47 | 7,924,963 | 60.0 |
| 50c RC201 6V high | PACO+ALNS W8 | 419.42 +/- 61.93 | 3196.95 +/- 1664.13 | 9,670,621 | 129.6 |
| 100c RC101 10V medium | NSGA-II | 1797.40 +/- 65.06 | 3211.08 +/- 1180.37 | 4,806,539 | 9.9 |
| 100c RC101 10V medium | PACO-imp2 | 644.55 +/- 68.44 | 7535.28 +/- 1639.02 | 12,149,729 | 208.6 |
| 100c RC101 10V medium | PACO+ALNS W8 | 698.20 +/- 58.96 | 3639.25 +/- 1480.33 | 16,380,327 | 439.0 |
| 100c RC201 10V high | NSGA-II | 1858.96 +/- 90.41 | 4986.29 +/- 2224.57 | 9,413,243 | 10.7 |
| 100c RC201 10V high | PACO-imp2 | 748.45 +/- 72.99 | 10057.54 +/- 3303.21 | 22,766,246 | 390.1 |
| 100c RC201 10V high | PACO+ALNS W8 | 768.80 +/- 74.72 | 6998.58 +/- 2423.48 | 24,116,963 | 583.9 |

### 4.3 Statistical Ranking

| Algorithm | Cost best count | Tardiness best count | HV best count | Mean cost rank | Mean tardiness rank | Mean HV rank | Mean relative cost increase | Mean relative tardiness increase |
|-----------|-----------------|----------------------|---------------|----------------|--------------------|--------------|-----------------------------|---------------------------------|
| NSGA-II | 0 | 16 | 0 | 3.00 | 1.00 | 3.00 | +110.93% | 0.00% |
| PACO-imp2 | 12 | 0 | 0 | 1.25 | 3.00 | 2.00 | +0.65% | +183.23% |
| PACO+ALNS W8 | 4 | 0 | 16 | 1.75 | 2.00 | 1.00 | +3.07% | +78.94% |

Interpretation:

- **Cost**: PACO-imp2 has the best mean rank (1.25), followed by W8 (1.75), with a small gap between them; NSGA-II is clearly more expensive.
- **Tardiness**: NSGA-II has the best mean rank (1.00), but at the cost of being 110.93% more expensive.
- **Hypervolume**: W8 has the highest HV in all 16 configurations, indicating the widest Pareto-front coverage and a more complete cost-tardiness trade-off.

### 4.4 W7 to W8 Changes

| Scale | Delta Cost | Delta Tardiness | Delta HV | Delta Time (s) |
|-------|------------|-----------------|----------|----------------|
| 25c | +0.00 | +0.00 | +47,955 | -22.15 |
| 50c | -2.04 | -24.63 | +264,560 | -125.68 |
| 100c | -8.07 | -302.12 | -1,325,861 | -284.38 |

Mean runtime by scale:

| Scale | W7 (s) | W8 (s) | Speedup |
|-------|--------|--------|---------|
| 25c | 54.70 | 32.55 | about 1.7x |
| 50c | 246.15 | 120.46 | about 2.0x |
| 100c | 791.13 | 506.75 | about 1.6x |

Note: on 25c, W8 mean cost and tardiness are essentially unchanged from W7, with a small HV improvement on RC2; on 50c, cost, tardiness, and HV all improve; on 100c, cost and tardiness improve, but mean HV falls by about 1.33e6, driven mainly by `100c_RC101_10V_high` (-2,321,385) and `100c_RC201_10V_high` (-2,753,291), while `100c_RC101_10V_medium` still improves (+320,244). W8's value therefore lies mainly in robustness, generalization, and runtime, rather than in simultaneously improving front quality on all large instances.

### 4.5 Representative Pareto Fronts

| Config | Figure |
|--------|--------|
| 25c RC201 medium | ![25c RC201 medium](../../src/experiments/PACO+ALNS/results/20260807_w8/pareto_compare_25c_RC201_2V_medium.png) |
| 50c RC101 4V medium | ![50c RC101 4V medium](../../src/experiments/PACO+ALNS/results/20260807_w8/pareto_compare_50c_RC101_4V_medium.png) |
| 100c RC101 10V medium | ![100c RC101 10V medium](../../src/experiments/PACO+ALNS/results/20260807_w8/pareto_compare_100c_RC101_10V_medium.png) |

---

## 5. E-CVRP Comparison Results (W8)

### 5.1 Overall Statistics

| Algorithm | Feasible instances | Instances with missing customers | Overload instances | Mean gap on the 4 instances with known optima | Mean time (s) |
|-----------|--------------------|----------------------------------|--------------------|-----------------------------------------------|---------------|
| NSGA-II | 21/24 | 0 | 6 (33 overloaded solutions in total) | +67.44% | 7.8 |
| PACO-imp2 | 22/24 | 2 (X-n830, X-n920) | 0 | +9.63% | 30.4 |
| PACO+ALNS W8 | 24/24 | 0 | 0 | -1.01% | 1105.3 |

Notes:

- NSGA-II cannot produce a feasible solution within the given budget on X-n759, X-n830, and X-n920; on the remaining instances, a feasible solution can still be selected even when overloaded solutions appear during search;
- PACO-imp2 has missing customers on X-n830 and X-n920 and therefore no feasible solution;
- W8 finds feasible solutions on all 24 instances, and on every instance where all three algorithms have feasible solutions, its best cost is never worse than the other two;
- W8's mean time is inflated by the large X instances (X-n1006 alone takes about 7388s); on small and medium instances the time is not high.

### 5.2 The 4 Instances with Published Optimal Values

| Instance | Optimal value | NSGA-II Best | PACO-imp2 Best | W8 Best | W8 Gap |
|----------|---------------|--------------|----------------|---------|--------|
| E-n29-k4-s7 | 383 | 464.67 | 418.09 | 375.28 | -2.02% |
| E-n30-k3-s7 | 577 | 776.42 | 603.23 | 568.56 | -1.46% |
| E-n35-k3-s5 | 527 | 921.27 | 568.20 | 535.80 | +1.67% |
| F-n49-k4-s4 | 740 | 1769.07 | 865.73 | 723.54 | -2.22% |

Note: W8's negative gap is obtained under the CVRP relaxation with energy/recharging constraints removed and must not be interpreted as beating the published EVRP optimum.

### 5.3 All 24 Instances

| Instance | Opt/BKS | NSGA-II Best | PACO-imp2 Best | W8 Best | W8 Gap |
|----------|---------|--------------|----------------|---------|--------|
| E-n29-k4-s7 | 383 | 464.67 | 418.09 | 375.28 | -2.02% |
| E-n30-k3-s7 | 577 | 776.42 | 603.23 | 568.56 | -1.46% |
| E-n35-k3-s5 | 527 | 921.27 | 568.20 | 535.80 | +1.67% |
| E-n37-k4-s4 | - | 1171.92 | 907.85 | 837.67 | - |
| E-n60-k5-s9 | - | 1249.47 | 597.75 | 536.79 | - |
| E-n89-k7-s13 | - | 1965.98 | 874.67 | 697.36 | - |
| E-n112-k8-s11 | - | 3036.51 | 1203.51 | 860.86 | - |
| F-n49-k4-s4 | 740 | 1769.07 | 865.73 | 723.54 | -2.22% |
| F-n80-k4-s8 | - | 925.94 | 331.34 | 241.97 | - |
| F-n140-k5-s5 (run with k7) | - | 5077.33 | 1522.20 | 1188.58 | - |
| M-n110-k10-s9 | - | 3190.97 | 1023.96 | 820.55 | - |
| M-n126-k7-s5 | - | 4947.29 | 1350.23 | 1051.78 | - |
| M-n163-k12-s12 | - | 4512.80 | 1673.15 | 1152.06 | - |
| M-n212-k16-s12 | - | 6232.89 | 2167.03 | 1745.49 | - |
| X-n147-k7-s4 | - | 72943.80 | 24720.32 | 17825.19 | - |
| X-n221-k11-s7 | - | 49639.74 | 16796.40 | 14274.14 | - |
| X-n360-k40-s9 | - | 120812.29 | 41992.84 | 34722.12 | - |
| X-n469-k26-s10 | - | 157157.33 | 41129.19 | 30307.91 | - |
| X-n577-k30-s4 | - | 152308.60 | 66159.74 | 55795.00 | - |
| X-n698-k75-s13 | - | 350093.59 | 113181.18 | 94374.60 | - |
| X-n759-k98-s10 | - | nan | 119587.03 | 99160.74 | - |
| X-n830-k171-s11 | - | nan | nan | 297891.41 | - |
| X-n920-k207-s4 | - | nan | nan | 433017.09 | - |
| X-n1006-k43-s5 | - | 538246.83 | 118031.31 | 90816.60 | - |

### 5.4 Summary by Family

| Family | Instances | NSGA-II feasible | PACO-imp2 feasible | W8 feasible |
|--------|-----------|------------------|--------------------|-------------|
| E | 7 | 7 | 7 | 7 |
| F | 3 | 3 | 3 | 3 |
| M | 4 | 4 | 4 | 4 |
| X | 10 | 7 | 8 | 10 |

---

## 6. Discussion and Limitations

### 6.1 Method Positioning on Solomon

The three algorithms show a clear division of labor on Solomon:

- PACO-imp2 favors low-cost solutions, but with higher tardiness;
- NSGA-II favors low-tardiness solutions, but with higher cost;
- PACO+ALNS W8 sits between the two and covers the Pareto front most broadly, hence the highest HV.

This difference is a normal phenomenon in bi-objective optimization; W8's value is obtaining a more complete set of trade-off solutions at an acceptable time cost.

### 6.2 Generalization on E-CVRP

The E-CVRP validation shows that W8's robust construction is clearly better than the other two algorithms on capacity constraints, missing customers, and large instances. NSGA-II has capacity repair, but remains unstable on the large X series; PACO-imp2 produces missing customers on very large instances.

### 6.3 Limitations

1. E-CVRP energy/recharging constraints are not modeled, so a negative gap is not evidence of beating the published optimum;
2. Only 4 of the 24 instances have published optimal values, so the statistical sample is limited;
3. The computation budgets of the three algorithms are not fully consistent (W8 takes significantly longer on the large X instances);
4. F-n140 is run with 7 vehicles because the declared vehicle count is infeasible;
5. W8's HV on Solomon 100c shows mixed changes relative to W7 and does not form a consistent advantage;
6. No paired significance tests (e.g., Wilcoxon) were performed; the current conclusions rely mainly on means and rankings.

---

## 7. Conclusion

After weeks 1-8 of iteration, PACO+ALNS W8 has become the core method of this project:

1. **Solomon RC**: highest HV in 16/16 configurations, cost close to PACO-imp2, tardiness clearly lower than PACO-imp2, and runtime about 1.6-2.0x lower than W7;
2. **E-CVRP**: feasible on 24/24 instances, best cost never worse than PACO-imp2 or NSGA-II, and a mean gap of -1.01% (relaxation basis) on the 4 instances with published optimal values;
3. **Method maturity**: W8 fixed three robustness issues (drone ID, timeline stale reads, SA scoring), and added capacity-aware repair and archive filtering, providing a stable baseline for introducing energy constraints, multi-visit drones, and RL-based operator selection.

---

## 8. Data and Code Locations

| Content | Path |
|---------|------|
| W8 algorithm implementation | `src/experiments/PACO+ALNS/PACO+ALNSW8.py` |
| Three-algorithm comparison script | `src/experiments/PACO+ALNS/compare_three_algorithms.py` |
| E-CVRP validation script | `src/experiments/PACO+ALNS/evrp_validation.py` |
| Solomon W8 results and analysis | `src/experiments/PACO+ALNS/results/20260807_w8/` |
| Solomon W7 results and analysis | `src/experiments/PACO+ALNS/results/20260805/` |
| E-CVRP W8 results and analysis | `src/experiments/PACO+ALNS/results/20260806_evrp_w8/` |
| E-CVRP benchmark instances | `src/experiments/e-cvrp_benchmark_instances/` |
