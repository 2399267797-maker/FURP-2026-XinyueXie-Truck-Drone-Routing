# Week 3 Lab: Experiment Design, Evaluation, and Report Writing

## Abstract

This report presents a systematic comparative analysis of two multi-objective optimization methods for the truck-drone collaborative routing problem:

- **Collaborative P-ACO**: An ant colony optimization-based method with synchronized truck-drone joint decision-making
- **NSGA-II**: A multi-objective evolutionary algorithm based on non-dominated sorting genetic algorithm

Experiments are conducted on the Solomon RC benchmark dataset with 25 and 50 customer instances, testing both medium (4 km) and high (6 km) drone endurance configurations. A pure truck-only (No-Drone Baseline) is included as an additional baseline. All methods optimize travel cost and tardiness as dual objectives, with Hypervolume (HV) as the comprehensive performance metric.

Experimental results show that P-ACO significantly outperforms NSGA-II and the No-Drone baseline across all configurations:
- **HV Comparison**: P-ACO's HV dominates across all configurations. For 25-customer scenarios, P-ACO achieves mean HV of 742,640 vs NSGA-II's 537,866 and No-Drone's 442,597. For 50-customer 4T+4D scenarios, P-ACO achieves mean HV of 3,615,524 vs NSGA-II's 1,899,776 and No-Drone's 1,382,291.
- **Cost Advantage**: P-ACO costs are approximately 30-40% lower than NSGA-II and 25-43% lower than the No-Drone baseline.
- **Time Window Flexibility**: RC2 (wide time windows) produces much higher tardiness than RC1 (tight time windows), but P-ACO's cost advantage remains stable across both time window types.
- **Drone Utilization**: Both algorithms incorporate drone missions in their Pareto-optimal solutions. NSGA-II shows significantly higher drone utilization (67.7–98.1% of solutions contain drone missions, with 1.12–3.11 missions per solution on average) compared to P-ACO (20.9–57.4% of solutions, with 0.30–0.72 missions per solution). This suggests that NSGA-II's crossover/mutation operators more readily produce drone-enabled solutions, while P-ACO's 3D pheromone matrix is harder to converge for drone routes (detailed in §3.2).

---

## 1. Experimental Setup

### 1.1 Comparison Objectives

- **Test Method**: Collaborative P-ACO (implementation of [DOI: 10.1109/TITS.2020.2992549](https://doi.org/10.1109/TITS.2020.2992549))
- **Baselines**:
  - NSGA-II (classic multi-objective GA, sharing the same truck-drone collaborative mode)
  - No-Drone Baseline (pure truck delivery)
- **Research Questions**:
  1. Under the multi-objective (cost + tardiness) framework, does P-ACO achieve a superior Pareto front compared to NSGA-II?
  2. Can drone-assisted delivery significantly reduce total cost compared to pure trucks?
  3. How does drone endurance (4 km vs 6 km) affect solution quality?

### 1.2 Dataset and Instance Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | Solomon RC series (RC101, RC201) |
| Customer sizes | 25, 50 |
| Coordinate scaling | Solomon [0,100] → Urban [0,12] km |
| Depot location | (6.0, 6.0) |
| Time window types | RC1 (tight, scheduling horizon 120 min); RC2 (wide, scheduling horizon 240 min) |
| Truck speed | 25 km/h |
| Drone speed | 50 km/h |
| Truck capacity | 200.0 |
| Drone capacity | 40.0 |
| Drone endurance | medium = 4 km, high = 6 km |

### 1.3 Vehicle Configuration

| Customers | Trucks | Drones | Description |
|-----------|--------|--------|-------------|
| 25 | 2 | 2 | 1:1 truck-drone pairing |
| 50 | 4 | 4 | Standard configuration |
| 50 | 6 | 6 | High-density configuration |

Trucks and drones have a 1:1 pairing; each truck carries one drone.

### 1.4 Algorithm Configuration

| Parameter | P-ACO | NSGA-II |
|-----------|-------|---------|
| Population/Ants | 50 (25c) / 80 (50c) | 50 (25c) / 80 (50c) |
| Iterations | 100 | 120 |
| Repeats | 10 | 10 |
| Crossover | — | SBX (Simulated Binary Crossover) |
| Mutation | — | Polynomial Mutation |
| Selection | Pseudo-random proportional (q0=0.5) + Roulette wheel | Tournament selection + Crowding distance |
| Pheromone params | α=1.0, β=2.0, ρ=0.15, Qc=120, Qt=60 | — |

### 1.5 Evaluation Metrics

| Metric | Definition |
|--------|------------|
| **Cost** | Sum of vehicle fixed costs + truck distance × 2.0 + drone flight distance × 1.0 (fixed cost: truck 100/vehicle, drone 0) |
| **Tardiness** | $\sum \max(0, \text{arrival time} - \text{time window end}) \times \text{priority weight}$ |
| **Hypervolume (HV)** | Area covered by the Pareto front relative to reference point (170, 140), measuring multi-objective solution set quality |
| **Drone Utilization** | Number of drones actually used / total drones available, and number of customers served by drones |

### 1.6 Hardware and Environment

Experiments were run on the following environment:

| Item | Configuration |
|------|---------------|
| Processor | Intel(R) Core(TM) Ultra 5 125H (3.60 GHz) |
| RAM | 32.0 GB (31.5 GB usable) |
| Graphics | Intel(R) Arc(TM) Graphics (128 MB) |
| System type | 64-bit OS, x64-based processor | 
| OS | Windows 11 |
| Python version | 3.12 |
| Key dependencies | numpy, matplotlib, DEAP (for NSGA-II) |

---

## 2. Results

### 2.1 Summary Comparison Tables

#### 25 Customer Experiments

| Config | RC Type | Endurance | Method | Mean Cost ± Std | Mean Tardiness ± Std | HV ± Std |
|--------|---------|-----------|--------|-----------------|----------------------|----------|
| 25c_RC1_medium | RC1 | 4 km | P-ACO | 282.60 ± 8.51 | 331.69 ± 153.97 | 199,082 ± 8,655 |
| 25c_RC1_medium | RC1 | 4 km | NSGA-II | 364.21 ± 26.70 | 433.88 ± 195.93 | 113,690 ± 14,866 |
| 25c_RC1_medium | RC1 | 4 km | No-Drone | 372.39 ± 21.06 | 499.43 ± 171.06 | 91,844 ± 11,640 |
| 25c_RC1_high | RC1 | 6 km | P-ACO | 286.25 ± 10.60 | 238.50 ± 131.13 | 263,953 ± 6,372 |
| 25c_RC1_high | RC1 | 6 km | NSGA-II | 355.82 ± 28.73 | 372.09 ± 165.78 | 174,459 ± 30,878 |
| 25c_RC1_high | RC1 | 6 km | No-Drone | 374.26 ± 36.95 | 533.20 ± 238.05 | 136,866 ± 25,463 |
| 25c_RC2_medium | RC2 | 4 km | P-ACO | 284.10 ± 10.55 | 1,779.79 ± 906.04 | 1,211,666 ± 32,354 |
| 25c_RC2_medium | RC2 | 4 km | NSGA-II | 379.14 ± 39.40 | 1,499.90 ± 1,044.29 | 917,786 ± 72,947 |
| 25c_RC2_medium | RC2 | 4 km | No-Drone | 401.70 ± 39.96 | 1,967.23 ± 1,066.45 | 734,364 ± 51,053 |
| 25c_RC2_high | RC2 | 6 km | P-ACO | 286.20 ± 10.95 | 1,762.54 ± 952.81 | 1,295,860 ± 30,560 |
| 25c_RC2_high | RC2 | 6 km | NSGA-II | 378.74 ± 42.26 | 1,376.92 ± 914.19 | 945,531 ± 66,636 |
| 25c_RC2_high | RC2 | 6 km | No-Drone | 395.59 ± 36.84 | 1,796.66 ± 979.91 | 807,315 ± 75,703 |

#### 50 Customer Experiments (4T+4D)

| Config | RC Type | Endurance | Method | Mean Cost ± Std | Mean Tardiness ± Std | HV ± Std |
|--------|---------|-----------|--------|-----------------|----------------------|----------|
| 50c_RC1_medium | RC1 | 4 km | P-ACO | 550.90 ± 8.55 | 576.80 ± 107.01 | 1,499,638 ± 22,457 |
| 50c_RC1_medium | RC1 | 4 km | NSGA-II | 868.22 ± 47.95 | 1,300.74 ± 322.89 | 589,564 ± 70,155 |
| 50c_RC1_medium | RC1 | 4 km | No-Drone | 901.79 ± 41.01 | 1,745.60 ± 333.86 | 418,091 ± 51,784 |
| 50c_RC1_high | RC1 | 6 km | P-ACO | 562.25 ± 23.64 | 523.66 ± 151.64 | 1,518,684 ± 33,058 |
| 50c_RC1_high | RC1 | 6 km | NSGA-II | 859.73 ± 38.53 | 1,102.30 ± 303.86 | 625,855 ± 84,696 |
| 50c_RC1_high | RC1 | 6 km | No-Drone | 932.34 ± 50.78 | 1,668.02 ± 386.39 | 370,604 ± 64,356 |
| 50c_RC2_medium | RC2 | 4 km | P-ACO | 552.94 ± 22.74 | 4,120.85 ± 976.51 | 5,282,577 ± 160,145 |
| 50c_RC2_medium | RC2 | 4 km | NSGA-II | 918.71 ± 70.69 | 3,611.81 ± 1,542.80 | 2,986,986 ± 270,684 |
| 50c_RC2_medium | RC2 | 4 km | No-Drone | 959.95 ± 73.76 | 5,029.40 ± 1,681.49 | 2,147,509 ± 240,399 |
| 50c_RC2_high | RC2 | 6 km | P-ACO | 557.19 ± 34.16 | 4,390.09 ± 1,029.82 | 6,161,197 ± 128,466 |
| 50c_RC2_high | RC2 | 6 km | NSGA-II | 917.23 ± 66.13 | 3,818.88 ± 1,328.81 | 3,396,700 ± 183,767 |
| 50c_RC2_high | RC2 | 6 km | No-Drone | 969.13 ± 71.61 | 5,492.40 ± 1,965.48 | 2,592,960 ± 207,477 |

#### 50 Customer Experiments (6T+6D)

| Config | RC Type | Endurance | Method | Mean Cost ± Std | Mean Tardiness ± Std | HV ± Std |
|--------|---------|-----------|--------|-----------------|----------------------|----------|
| 50c_RC1_medium | RC1 | 4 km | P-ACO | 664.37 ± 10.78 | 342.99 ± 136.70 | 1,408,398 ± 33,102 |
| 50c_RC1_medium | RC1 | 4 km | NSGA-II | 1,102.61 ± 52.55 | 872.61 ± 222.42 | 480,236 ± 52,898 |
| 50c_RC1_medium | RC1 | 4 km | No-Drone | 1,152.96 ± 55.32 | 1,056.73 ± 284.70 | 378,918 ± 52,601 |
| 50c_RC1_high | RC1 | 6 km | P-ACO | 667.66 ± 15.32 | 286.98 ± 106.00 | 1,141,092 ± 23,208 |
| 50c_RC1_high | RC1 | 6 km | NSGA-II | 1,117.33 ± 61.54 | 770.29 ± 226.13 | 365,225 ± 40,288 |
| 50c_RC1_high | RC1 | 6 km | No-Drone | 1,156.06 ± 50.97 | 917.06 ± 259.18 | 282,439 ± 40,884 |
| 50c_RC2_medium | RC2 | 4 km | P-ACO | 666.04 ± 19.72 | 3,060.18 ± 983.26 | 5,239,154 ± 203,004 |
| 50c_RC2_medium | RC2 | 4 km | NSGA-II | 1,143.87 ± 69.82 | 2,970.42 ± 1,315.44 | 2,504,753 ± 145,159 |
| 50c_RC2_medium | RC2 | 4 km | No-Drone | 1,175.58 ± 60.01 | 3,854.10 ± 1,450.02 | 2,020,746 ± 206,656 |
| 50c_RC2_high | RC2 | 6 km | P-ACO | 669.06 ± 29.03 | 2,853.70 ± 899.41 | 5,260,558 ± 189,837 |
| 50c_RC2_high | RC2 | 6 km | NSGA-II | 1,138.36 ± 75.86 | 3,053.43 ± 1,367.19 | 2,385,334 ± 149,209 |
| 50c_RC2_high | RC2 | 6 km | No-Drone | 1,176.98 ± 69.76 | 3,804.40 ± 1,250.40 | 1,856,604 ± 199,511 |

### 2.2 Aggregate Statistics

| Scale | Method | Feasibility | Mean Cost | Mean Tardiness | Mean HV | Mean Runtime (s) |
|-------|--------|-------------|-----------|----------------|---------|-----------------|
| 25c | P-ACO | 100% | 284.8 | 1,028.1 | 742,640 | 37.4 |
| 25c | NSGA-II | 100% | 369.5 | 920.7 | 537,866 | 1.1 |
| 25c | No-Drone | 100% | 386.0 | 1,199.1 | 442,597 | — |
| 50c(4T) | P-ACO | 100% | 555.8 | 2,402.9 | 3,615,524 | 269.9 |
| 50c(4T) | NSGA-II | 100% | 891.0 | 2,458.4 | 1,899,776 | 2.1 |
| 50c(4T) | No-Drone | 100% | 940.8 | 3,483.9 | 1,382,291 | — |
| 50c(6T) | P-ACO | 100% | 666.8 | 1,636.0 | 3,262,300 | 248.3 |
| 50c(6T) | NSGA-II | 100% | 1,125.5 | 1,916.7 | 1,433,887 | 2.2 |
| 50c(6T) | No-Drone | 100% | 1,165.4 | 2,408.1 | 1,134,677 | — |

### 2.3 Drone Utilization Statistics

> **Note**: The table below counts DroneMission instances in the **final Pareto-optimal solutions** from all 10 runs combined. Both algorithms produce drone-containing Pareto solutions, but NSGA-II achieves significantly higher drone utilization rates.

| Config | Method | Pareto solutions | Solutions w/ drones | % | Total drone missions | Avg/solution |
|--------|--------|-----------------|--------------------|----|--------------------|-------------|
| 25c_RC1_medium | P-ACO | 144 | 67 | 46.5% | 82 | 0.57 |
| 25c_RC1_medium | NSGA-II | 214 | 192 | 89.7% | 394 | 1.84 |
| 25c_RC1_high | P-ACO | 131 | 67 | 51.1% | 92 | 0.70 |
| 25c_RC1_high | NSGA-II | 211 | 207 | 98.1% | 657 | 3.11 |
| 25c_RC2_medium | P-ACO | 156 | 66 | 42.3% | 81 | 0.52 |
| 25c_RC2_medium | NSGA-II | 205 | 164 | 80.0% | 330 | 1.61 |
| 25c_RC2_high | P-ACO | 155 | 77 | 49.7% | 102 | 0.66 |
| 25c_RC2_high | NSGA-II | 211 | 198 | 93.8% | 528 | 2.50 |
| 50c(4T)_RC1_medium | P-ACO | 68 | 39 | 57.4% | 49 | 0.72 |
| 50c(4T)_RC1_medium | NSGA-II | 295 | 268 | 90.8% | 517 | 1.75 |
| 50c(4T)_RC1_high | P-ACO | 105 | 39 | 37.1% | 64 | 0.61 |
| 50c(4T)_RC1_high | NSGA-II | 320 | 288 | 90.0% | 969 | 3.03 |
| 50c(4T)_RC2_medium | P-ACO | 185 | 96 | 51.9% | 134 | 0.72 |
| 50c(4T)_RC2_medium | NSGA-II | 320 | 228 | 71.2% | 405 | 1.27 |
| 50c(4T)_RC2_high | P-ACO | 211 | 64 | 30.3% | 128 | 0.61 |
| 50c(4T)_RC2_high | NSGA-II | 321 | 273 | 85.0% | 615 | 1.92 |
| 50c(6T)_RC1_medium | P-ACO | 137 | 42 | 30.7% | 54 | 0.39 |
| 50c(6T)_RC1_medium | NSGA-II | 314 | 244 | 77.7% | 385 | 1.23 |
| 50c(6T)_RC1_high | P-ACO | 130 | 44 | 33.8% | 56 | 0.43 |
| 50c(6T)_RC1_high | NSGA-II | 322 | 287 | 89.1% | 730 | 2.27 |
| 50c(6T)_RC2_medium | P-ACO | 187 | 39 | 20.9% | 56 | 0.30 |
| 50c(6T)_RC2_medium | NSGA-II | 325 | 220 | 67.7% | 365 | 1.12 |
| 50c(6T)_RC2_high | P-ACO | 170 | 74 | 43.5% | 114 | 0.67 |
| 50c(6T)_RC2_high | NSGA-II | 327 | 247 | 75.5% | 467 | 1.43 |

> Note: NSGA-II produces 2–5× more drone missions per Pareto solution than P-ACO on average across all configurations. However, despite higher drone usage, NSGA-II's cost is still 30–40% higher than P-ACO, indicating that the drone missions NSGA-II finds do not sufficiently reduce cost to overcome its weaker routing optimization.

### 2.4 Pareto Front Visualizations

![25c RC1 medium](FURP-2026-XinyueXie-Truck-Drone-Routing/src/experiments/PACO_vs_NSGA2/results/pareto_25c_RC101_medium.png)
![25c RC2 medium](FURP-2026-XinyueXie-Truck-Drone-Routing/src/experiments/PACO_vs_NSGA2/results/pareto_25c_RC201_medium.png)
![50c 4T+4D RC1 medium](FURP-2026-XinyueXie-Truck-Drone-Routing/src/experiments/PACO_vs_NSGA2/results/pareto_50c_RC101_medium.png)
![50c 4T+4D RC2 medium](FURP-2026-XinyueXie-Truck-Drone-Routing/src/experiments/PACO_vs_NSGA2/results/pareto_50c_RC201_medium.png)
![50c 6T+6D RC1 medium](FURP-2026-XinyueXie-Truck-Drone-Routing/src/experiments/PACO_vs_NSGA2/results/6D_pareto_50c_RC101_medium.png)
![50c 6T+6D RC2 medium](FURP-2026-XinyueXie-Truck-Drone-Routing/src/experiments/PACO_vs_NSGA2/results/6D_pareto_50c_RC201_medium.png)

- X-axis: Travel Cost, range 60–160
- Y-axis: Tardiness, range 60–140
- Green: P-ACO, Blue: NSGA-II, Orange: No-Drone Baseline
- Lines connect the joint non-dominated solutions (true Pareto front) across all 10 runs

### 2.5 Route Visualization

#### P-ACO Route Example (25c RC1 medium best solution)

![P-ACO 25c RC1 medium](file:///c:/Users/23992/FURP-2026-XinyueXie-Truck-Drone-Routing/src/experiments/PACO_vs_NSGA2/results/paco_25c_RC101_medium.png)

#### NSGA-II Route Example (25c RC1 medium best solution)

![NSGA-II 25c RC1 medium](file:///c:/Users/23992/FURP-2026-XinyueXie-Truck-Drone-Routing/src/experiments/PACO_vs_NSGA2/results/nsga2_25c_RC101_medium.png)

---

## 3. Discussion

### 3.1 Algorithm Performance Comparison

Across all 12 experimental configurations, P-ACO significantly outperforms NSGA-II and the No-Drone baseline on both **Cost** and **HV**:

- **HV Dominance**: P-ACO's HV is 1.5–3.0× that of NSGA-II and 2.0–4.0× that of the No-Drone baseline across all configurations. The gap is larger under RC2 scenarios (e.g., 4T+4D RC2 high: P-ACO HV=6,161,197 vs NSGA-II HV=3,396,700), because RC2's wider time windows create a larger tardiness range where P-ACO's pheromone guidance enables more focused exploration of low-tardiness regions.
- **Cost Difference**: P-ACO costs are approximately 60–65% of NSGA-II and 55–60% of the No-Drone baseline. This advantage is highly consistent across configurations (extremely low standard deviation), indicating strong stability in P-ACO's ant-based search.
- **Tardiness**: Under RC1 (tight time windows), P-ACO's tardiness is significantly lower than both baselines (P-ACO 238–577 vs NSGA-II 372–1301 vs No-Drone 499–1746). However, under RC2 (wide time windows), P-ACO's tardiness exceeds NSGA-II's (P-ACO 1,763–4,390 vs NSGA-II 1,377–3,818). This is because P-ACO prioritizes cost optimization in RC2, accepting higher tardiness.

### 3.2 Effectiveness of Drone Usage

Both algorithms **do** incorporate drone missions in their final Pareto-optimal solutions, but at very different rates:
- **P-ACO**: 20.9%–57.4% of Pareto solutions contain drone missions
- **NSGA-II**: 67.7%–98.1% of Pareto solutions contain drone missions

NSGA-II achieves 2–5× more drone missions per solution than P-ACO. This disparity arises from the algorithms' different search mechanisms:

**P-ACO drone search challenge**: P-ACO uses a 3D pheromone matrix (31³ ≈ 29,791 entries for 25-customer problems) to guide drone route selection. Each ant produces at most 0–1 drone missions per solution, so the 3D matrix receives far fewer updates per iteration compared to the 2D truck matrix (31² ≈ 961 entries, ~25 updates per solution). This creates a cold-start problem — the drone pheromone signals are too sparse to converge effectively within 100 iterations.

**NSGA-II drone advantage**: NSGA-II's crossover and mutation operators naturally produce drone missions in offspring solutions without needing dedicated drone-specific probability calculations. However, despite NSGA-II's higher drone utilization, its **cost is still 30–40% higher** than P-ACO. This reveals that while NSGA-II explores more drone-enabled routes, these routes do not improve cost as effectively as P-ACO's pheromone-guided truck routing.

**Why cost savings from drones are limited**: For a drone mission `i→j→k` (drone serves customer `j`, truck continues from `i` to `k`):
```
Drone cost = d_ij + d_jk  (drone flight)
Truck cost = 2 × d_ik     (truck drives from i to k and back)
Cost saved = 2 × d_ik - (d_ij + d_jk)  (drone variable cost rate = 1.0, truck rate = 2.0)
```
Given the triangle inequality `d_ij + d_jk ≥ d_ik`, the maximum distance saving is `d_ik`, which is at most the drone range (4 km). At truck cost rate 2.0, the max variable cost saving is ~8 cost units — significant but modest compared to total solution costs of 280+. This means drone usage reduces cost by at most ~3%, which explains why even NSGA-II's higher drone utilization fails to close the gap with P-ACO's superior truck routing.

**Endurance increase from 4 km to 6 km** has minimal impact on cost (<3% increase) but moderately increases drone utilization in NSGA-II (e.g., 25c RC1: 89.7% → 98.1% of solutions with drones), indicating wider endurance slightly expands drone feasibility.

### 3.3 Problem Scale Effects

- **Nonlinear cost growth with customer count**: From 25 to 50 customers (4T+4D), P-ACO cost grows from ~285 to ~556 (1.95×), while NSGA-II grows from ~370 to ~891 (2.41×). NSGA-II degrades faster, indicating its solution quality deteriorates more rapidly with problem size.
- **6T+6D vs 4T+4D**: P-ACO cost under 6T+6D is ~20% higher than 4T+4D (556→667) due to increased fixed costs from additional vehicles. However, tardiness decreases by ~32% (2,403→1,636) as more vehicles better satisfy time windows. NSGA-II shows similar trends with smaller magnitude.
- **The No-Drone baseline** maintains the same proportional relationship across both vehicle configurations, further confirming that low drone utilization has limited impact on the overall cost structure.

### 3.4 RC1 vs RC2 (Time Window Tightness)

- **Tardiness magnitude difference**: RC1 tardiness ranges 238–577, while RC2 ranges 1,377–5,492. RC2's wider time windows spread arrival times over a larger interval, substantially increasing the tardiness ceiling.
- **Algorithm adaptability**: P-ACO's cost is nearly identical between RC1 and RC2 (25c: 284 vs 285), demonstrating that its routing capability is unaffected by time window type. However, tardiness increases 5–10× in RC2, suggesting a tradeoff favoring cost over punctuality.
- **No-Drone baseline** degrades most severely under RC2 (tardiness up to 5,492), suggesting drones should theoretically add more value in wide-window scenarios — yet neither algorithm exploits this.

### 3.5 Failure Cases and Limitations

- **P-ACO's low drone utilization**: Only 20.9%–57.4% of P-ACO's Pareto solutions contain drone missions, compared to 67.7%–98.1% for NSGA-II. The root cause is P-ACO's 3D pheromone cold-start problem (detailed in §3.2) — the sparse drone pheromone matrix cannot converge effectively within 100 iterations. This is a structural limitation of extending 2D ant colony optimization to 3D drone decision spaces.
- **Limited cost impact of drones even when used**: Despite NSGA-II achieving high drone utilization, its cost remains 30–40% higher than P-ACO. This is because drone missions reduce cost by at most ~3% (max ~8 cost units at 4 km range), insufficient to compensate for NSGA-II's weaker truck routing optimization.
- **NSGA-II limitations**: Under 50-customer scenarios, NSGA-II's standard deviation is much larger than P-ACO's (cost std 38–76 vs 9–34), indicating significant solution quality variation across runs.
- **Deviation from original paper**: P-ACO's drone decision logic requires the drone to arrive first without waiting, which constrains the feasible search space. Relaxing this constraint (e.g., allowing truck wait-for-drone) could potentially increase drone utilization.

---

## 4. Conclusion

This experiment systematically compares Collaborative P-ACO and NSGA-II on the truck-drone collaborative routing problem using the Solomon RC benchmark dataset, with a pure truck baseline as control. Key findings:

1. **P-ACO leads in solution quality across all configurations**: P-ACO costs are 30–40% lower than NSGA-II, with HV 1.5–3.0× higher, and significantly better run-to-run stability. P-ACO's pheromone guidance mechanism effectively converges to high-quality routes within 100 iterations.

2. **Drone utilization exists but provides limited cost benefit**: Both algorithms produce drone-containing Pareto solutions. NSGA-II achieves higher drone utilization (67.7–98.1% of solutions) than P-ACO (20.9–57.4%), but this does not translate into a cost advantage due to drones' limited maximum cost savings (~3%). P-ACO's 3D pheromone cold-start problem further suppresses its drone exploration, while NSGA-II compensates through crossover/mutation operators that naturally generate drone missions.

3. **NSGA-II limitations**: Variance increases significantly at 50-customer scale, and solution quality degrades faster (2.41×) than P-ACO (1.95×) as problem size grows. NSGA-II shows higher tardiness under RC1 and higher cost under RC2.

4. **RC type has substantial impact**: RC2 tardiness is 5–10× that of RC1, but P-ACO's cost remains unaffected by time window type, demonstrating stronger adaptability.

Future work directions:
- **Relax the "no-wait" constraint**: Introduce truck wait-for-drone strategies (e.g., max(arr_truck, arr_drone)) and test whether drone utilization improves significantly
- **Optimize P-ACO drone pheromone updates**: Increase the initial value or deposition multiplier for the 3D pheromone matrix to mitigate the cold-start problem
- **Extend the dataset**: Test additional Solomon instances (C, R series) and different map scales

---

## Appendix: File Structure

| File / Directory | Purpose |
|------------------|---------|
| `src/experiments/PACO/` | P-ACO algorithm implementation |
| `src/experiments/NSGA2/` | NSGA-II algorithm implementation |
| `src/experiments/PACO_vs_NSGA2/` | Comparison experiment scripts and results |
| `src/experiments/PACO_vs_NSGA2/compare_paco_nsga2_solomon.py` | Main experiment runner |
| `src/experiments/PACO/data/solomon_loader.py` | Solomon dataset loader |
| `src/experiments/PACO_vs_NSGA2/models/vrp_model.py` | Problem model definition (actually used by all experiment code) |
| `src/experiments/PACO/models/vrp_model.py` | Model definition (overshadowed by cache, not used in experiments) |
| `src/experiments/NSGA2/models/vrp_model.py` | Fully commented out, placeholder only |
| `src/experiments/PACO_vs_NSGA2/utils/evaluator.py` | Unified evaluator |
| `src/experiments/PACO_vs_NSGA2/utils/visualizer.py` | Route visualization |
| `docs/src/week3-report.md` | This report |