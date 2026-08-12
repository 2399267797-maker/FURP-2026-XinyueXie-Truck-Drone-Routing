# A Comparative Study of Multi-Objective Algorithms for Truck-Drone Cooperative Delivery Routing: NSGA-II, PACO, PACO+ALNS, and Pure ALNS

## Abstract

This paper investigates the truck-drone routing problem with time windows (TDRPTW) under a unified bi-objective model that minimizes total travel cost and weighted time-window tardiness. Four multi-objective algorithms, NSGA-II, PACO, PACO+ALNS, and a pure ALNS baseline, are compared on all 54 Solomon instances of the C, R, and RC families, with customer sizes of 25, 50, and 100 and two drone endurance levels, resulting in 324 configurations, each repeated three times with 100 iterations. PACO+ALNS achieves the best overall cost and hypervolume performance, with average cost and hypervolume ranks of 1.31 and 1.10 and the best cost and hypervolume in 222/324 and 291/324 configurations, respectively. NSGA-II attains the lowest tardiness in 209/324 configurations but shows substantially higher cost and produces capacity-overloaded solutions in 82 configurations. PACO lies in between with much lower runtime. Pure ALNS, an ablation that removes the ant-colony construction and pheromone update while keeping the same ALNS parameters and outer-iteration budget as PACO+ALNS, ranks third in cost and hypervolume (2.98 and 3.22) with an average runtime of 29.7 seconds per configuration, but fails to find a feasible solution on 6 wide-time-window 100-customer C2 configurations. On the WCCI-2020 E-CVRP benchmark (24 instances), PACO+ALNS reports a mean gap of -1.01% on the four instances with published best-known values, outperforming PACO (+9.63%), Pure ALNS (+3.32%), and NSGA-II (+67.44%); PACO+ALNS and Pure ALNS both find feasible solutions on all 24 instances. The paper provides the mathematical formulation, algorithmic pseudocode, grouped statistical tables, and reproducible experimental settings.

**Keywords**: truck-drone routing; time windows; multi-objective optimization; NSGA-II; ant colony optimization; adaptive large neighborhood search; pure ALNS; Solomon benchmark

## 1 Introduction

Drones introduce new possibilities for last-mile delivery routing optimization. Trucks handle trunk transportation, while drones can be launched from moving trucks to serve customers independently; the two vehicle types complement each other in payload, speed, range, and travel cost. However, this cooperation substantially enlarges the search space of the routing problem: in addition to determining customer visit orders, one must decide whether each customer is served by a truck or a drone, select the drone launch and retrieval points, and maintain temporal synchronization between trucks and drones, while satisfying time-window, payload, and range constraints.

The problem usually involves two conflicting objectives: transportation cost and time-window violation (tardiness). Incorporating tardiness into a single objective as a penalty introduces weight-selection issues; therefore, this paper adopts a bi-objective minimization framework and evaluates Pareto-front quality using indicators such as hypervolume (HV).

At the algorithmic level, evolutionary algorithms and swarm intelligence are the mainstream approaches to this problem. NSGA-II maintains solution diversity and convergence through non-dominated sorting and crowding distance; ant colony optimization (ACO) exploits pheromone to learn high-quality route structures; adaptive large neighborhood search (ALNS) performs deep local search through adaptive combinations of destroy and repair operators. Combining the constructive ability of ACO with the neighborhood-improvement ability of ALNS is an effective way to improve solution quality.

Most existing studies are limited to a few Solomon instances (e.g., RC101 and RC201) or a single scale, lacking systematic validation over complete benchmark families. The contributions of this paper are as follows:

1. Systematic experiments on the 54 instances of the C, R, and RC families of the Solomon benchmark under a unified model and evaluation protocol, covering customer sizes of 25/50/100 and medium/high drone endurance, for a total of 324 configurations;
2. Comparison of four algorithms, NSGA-II, PACO, PACO+ALNS, and Pure ALNS, on five groups of indicators (cost, tardiness, HV, runtime, and feasibility), together with grouped statistics by family, size, and endurance; Pure ALNS serves as an ablation baseline that isolates the contribution of the ALNS component after removing the ant-colony construction and pheromone update;
3. Engineering optimization of the capacity-repair operator of PACO+ALNS (fleet-level infeasibility lower bound and incremental load maintenance), with a complete comparison against PACO and NSGA-II;
4. Cross-dataset generalization tests on the WCCI-2020 E-CVRP benchmark.

The rest of this paper is organized as follows: Section 2 reviews related work; Section 3 defines the problem and the mathematical model; Section 4 describes the four algorithms; Section 5 presents the experimental design; Section 6 reports the results and analysis; Section 7 discusses limitations; Section 8 concludes the paper.

## 2 Related Work

### 2.1 Vehicle Routing Problem with Time Windows

In 1987, Solomon proposed the classical vehicle routing problem with time windows (VRPTW) benchmark, which contains randomly distributed (R), clustered (C), and mixed random-clustered (RC) instances, as well as short time-window (type 1) and long time-window (type 2) variants. This benchmark remains the most commonly used testbed for VRPTW and its extensions. The data used in this paper come from the C, R, and RC text files of this benchmark.

### 2.2 Truck-Drone Cooperative Delivery

The truck-drone cooperative delivery problem has been studied extensively in recent years. Early work focused on single-objective variants of the traveling salesman problem with drones (TSP-D); after introducing time windows and multiple vehicles, the problem becomes a bi-objective combinatorial optimization problem that requires simultaneous determination of vehicle routes, drone missions, and synchronization schedules. Existing studies usually employ genetic algorithms, ant colony optimization, or hybrids, but most experiments cover only a small number of instances.

### 2.3 Multi-Objective Metaheuristics

NSGA-II was proposed by Deb et al. and maintains population diversity while preserving convergence through non-dominated sorting and crowding distance; ant colony optimization was proposed by Dorigo et al. and guides construction through positive-feedback pheromone; ALNS was proposed by Ropke and Pisinger and realizes neighborhood search through adaptive selection of destroy and repair operators. Combining constructive metaheuristics with local search (the memetic idea) has been shown to improve solution quality significantly. To evaluate the contribution of the ALNS component itself, this paper also implements a pure ALNS baseline whose outer-iteration budget, ALNS iteration count, acceptance criterion, and operator weights are exactly the same as the ALNS part of PACO+ALNS, with only the ant-colony construction and pheromone update removed; initial solutions are generated by randomized greedy insertion.

### 2.4 Electric Vehicle Routing Problem Benchmark

Mavrovouniotis et al. released the E-CVRP benchmark for the IEEE WCCI-2020 competition, adding battery energy capacity and recharging stations to classical CVRP instances, for a total of 24 instances (E/F/M/X families, with 29 to 1006 customer nodes). The benchmark provides optimal values or best-known upper bounds and is suitable for testing algorithm generalization on large-scale instances with different distributions. Since the model in this paper does not include energy and recharging constraints, the validation on this benchmark is equivalent to a CVRP-relaxation test.

## 3 Problem Definition and Mathematical Model

### 3.1 Notation

Let the depot be node 0, the customer set be $N=\{1,\dots,n\}$, the truck set be $K$, and the drone set be $D$. Each customer $i \in N$ has demand $q_i$, service time $s_i$, time window $[e_i, l_i]$, and priority $p_i$. The truck capacity is $Q$, the drone capacity is $Q^d$, and the maximum flight range of a single drone mission is $R^d$. The speeds of trucks and drones are $v^t$ and $v^d$; the variable costs per unit distance are $c^t$ and $c^d$; the fixed costs are $F^t$ and $F^d$. The launch preparation time and retrieval time are $t^{launch}$ and $t^{ret}$.

### 3.2 Decision Variables

- Truck main route: each truck route $r$ is a customer sequence $r=(r_1,\dots,r_m)$;
- Drone mission: a triple $(i,j,k)$ means that the drone is launched at truck position $i$, serves customer $j$, and is retrieved at truck position $k$;
- Service mode: a customer is served either by a truck or by a drone.

### 3.3 Objective Functions

Objective 1: minimize total transportation cost

$$\min \quad C = \sum_{r \in K} \Big( F^t + c^t \cdot L_r^{truck} \Big) + \sum_{d \in D} \Big( F^d \cdot z_d + c^d \cdot L_d^{drone} \Big)$$

where $L_r^{truck}$ is the truck route distance (including the return leg), $L_d^{drone}$ is the drone flight distance, and $z_d$ indicates whether drone $d$ is used.

Objective 2: minimize weighted tardiness

$$\min \quad T = \sum_{i \in N} p_i \cdot \max\big(0,\ a_i - l_i\big)$$

where $a_i$ is the actual arrival time at customer $i$; for drone-served customers, $a_i$ is computed from the drone mission timeline.

### 3.4 Constraints

- Each customer is served exactly once;
- The load of each truck route does not exceed $Q$;
- The load of each drone mission does not exceed $Q^d$, and $d(i,j)+d(j,k) \le R^d$;
- At a launch point, the truck must dwell for the launch preparation time; at a retrieval point, it must wait for the drone and consume the retrieval time;
- Drone missions are synchronized with the truck route timeline, and the drone must not arrive at the retrieval point later than the truck (exceeding the threshold is treated as infeasible);
- Routes start and end at the depot.

### 3.5 Infeasible Solution Handling

During algorithm execution and result evaluation, solutions with missing customers incur a penalty of 10000 per customer, and solutions with capacity overload incur a penalty of 1000 per unit. Pareto fronts and statistical indicators are computed from the penalized objective values, while the numbers of missing customers and the overload amounts are recorded separately to explain algorithm differences.

## 4 Solution Algorithms

### 4.1 NSGA-II

#### Encoding and Initialization

An individual is a two-layer encoding of length $2n$: the first $n$ positions are a permutation of the customer visit order, and the last $n$ positions are service-mode genes (0 for truck, 1 for drone). The initial population is generated randomly, with mode genes set to truck mode with probability 0.6.

#### Decoding

A subtractive decoding strategy of "truck first, drone peeling" is adopted: all customers are first evenly distributed into truck routes in visit order to form the truck backbone; the algorithm then scans each route and converts customers whose mode gene is drone and whose triples satisfy the range, payload, and temporal synchronization constraints into drone missions, peeling them off the truck route. Drone assignments that cannot satisfy the constraints are counted as failed penalties.

#### Genetic Operators

- Crossover: partially mapped crossover (PMX) for the permutation segment and single-point crossover for the mode segment;
- Mutation: scramble mutation for the permutation segment and random bit flips for the mode segment;
- Selection: binary tournament based on non-dominated rank and crowding distance;
- Environmental selection: controlled elitism, allocating population slots across non-dominated fronts by a geometric ratio.

**Algorithm 1 NSGA-II Main Loop**

```
Input: problem model, population size N, number of generations G
1. Initialize population P0 and evaluate all individuals
2. for g = 1 to G:
3.    Generate a mating pool M by tournament selection
4.    Apply PMX crossover and scramble mutation to M to generate offspring O
5.    Evaluate the offspring
6.    Merge P = P union O and sort by non-dominated fronts
7.    Select N individuals by controlled elitism to form the new population
8. end for
9. Return the non-dominated solutions of the final population
```

#### Complexity

The per-generation complexity is approximately $O(N \cdot n \cdot T_{\mathrm{eval}})$, where $T_{\mathrm{eval}}$ is the decoding and evaluation cost; non-dominated sorting is $O(N^2)$.

### 4.2 PACO

PACO is an improved Pareto ant colony optimization for truck-drone cooperation, whose key features include a dual pheromone structure, weighted product transition probabilities, two-stage archive update, and crowding-distance pruning (version file PACO-imp2).

#### Pheromone Structure

Two types of pheromone are maintained:

- Truck arc pheromones $\tau^c_{ij}$ and $\tau^t_{ij}$, encoding cost and tardiness preferences, respectively;
- Drone triple pheromones $\tau^c_{ijk}$ and $\tau^t_{ijk}$, encoding the cooperative mission preference of launch point $i$, target customer $j$, and retrieval point $k$.

#### Construction Process

Each ant constructs routes for multiple trucks sequentially. Candidate actions include "visit the next customer by truck" and "drone triple mission". The transition probability is:

$$p(a) \propto \tau(a)^\alpha \cdot \eta(a)^\beta$$

where $\eta$ combines distance-saving and tardiness heuristics. Drone candidates enter the candidate set only when the range, payload, time-window, and waiting-threshold constraints are satisfied. After construction, unserved customers are inserted with a minimum-insertion-cost fallback.

#### Archive Update

In each generation, new solutions are merged with the existing archive, followed by two-stage dominance sorting that keeps only non-dominated solutions; if the archive exceeds its capacity, crowding-distance pruning is applied. Pheromone is reinforced according to archive solution quality, with evaporation and lower/upper bounds.

**Algorithm 2 PACO Main Loop**

```
Input: problem model, number of ants M, number of iterations I
1. Initialize pheromone to TAU_INIT and the archive to empty
2. for t = 1 to I:
3.    for a = 1 to M:
4.        Construct a complete solution by probability and evaluate (cost, tardiness)
5.    end for
6.    Merge the current solutions with the archive and perform two-stage dominance sorting
7.    If the archive exceeds its capacity, prune by crowding distance
8.    Update pheromone with the archive solutions (evaporation + reinforcement)
9. end for
10. Return the non-dominated solutions in the archive
```

### 4.3 PACO+ALNS

PACO+ALNS integrates local search and adaptive large neighborhood search on top of the PACO construction framework; the implementation in this paper is denoted PACO+ALNS (version file PACO+ALNSW8).

#### Overall Procedure

In each generation, candidate solutions are constructed or warm-started from the archive; each candidate undergoes 2-opt intra-route optimization and inter-route relocation; elite solutions undergo ALNS local search; finally, the archive and pheromone are updated by dominance relations.

#### 2-opt and Inter-Route Relocation

2-opt enumerates reversible segments on a single truck route and uses "distance increment + tardiness penalty change" as the improvement criterion, while synchronously correcting the launch/retrieval indices of drone missions. Inter-route relocation moves or exchanges customers between routes while maintaining capacity and drone-mission consistency.

#### ALNS Destroy and Repair Operators

The destroy operators include:

- Surgical removal: remove a consecutive segment;
- Random removal;
- Worst removal: sorted by insertion-cost saving;
- Related removal: by spatial proximity;
- Route removal.

The repair operators include:

- Greedy insertion;
- Regret insertion;
- Drone-aware insertion: prefers drone cooperative triples with cost savings.

The acceptance criterion uses simulated annealing: as the temperature $T$ cools exponentially with iterations, worse solutions are accepted with probability $\exp(-\Delta/T)$. Operator weights are adaptively updated with scores for "global best / improvement / accepted" and multiplied by a decay factor each generation.

#### Capacity Repair Optimization

Capacity repair is one of the most expensive modules in ALNS. The implementation in this paper includes the following optimizations:

1. Fleet-level infeasibility lower bound: if the total demand exceeds the total capacity of the current route set, return directly to avoid meaningless 280-round repair;
2. Incremental load maintenance: maintain per-route loads in an array, updated in O(1) on relocation/exchange/drone conversion, replacing repeated inner $sum()$ calls;
3. Drone-mission set precomputation: replace per-mission linear scans of "customer belongs to a drone mission" and "position is a launch/retrieval point" with set lookups;
4. Exchange feasibility fix: after an exchange, the available capacity of the target route must first subtract the load of the customer being exchanged out.

These optimizations significantly reduce the runtime on tight-capacity and infeasible instances while preserving the repair semantics on feasible instances.

**Algorithm 3 PACO+ALNS Main Loop**

```
Input: problem model, number of ants M, number of iterations I, ALNS iterations A
1. Initialize pheromone and archive
2. for t = 1 to I:
3.    for a = 1 to M:
4.        Construct or warm-start a solution
5.        2-opt intra-route optimization; inter-route relocation
6.    end for
7.    Apply ALNS(A) to the top E elite solutions: destroy-repair-capacity repair-accept
8.    Update the archive, pheromone, and operator weights
9. end for
10. Return the non-dominated solutions in the archive
```

#### Complexity

The construction stage is the same as PACO; 2-opt on a single route is $O(n^3)$ in the worst case, and inter-route relocation is about $O(|K|^2 n^2)$. Each ALNS repair evaluates insertions in about $O(n^2)$. Therefore, the per-generation cost of PACO+ALNS is significantly higher than pure PACO, but solution quality improves markedly under the same iteration budget.

### 4.4 Pure ALNS

To evaluate the independent contribution of the ALNS component in PACO+ALNS, this paper implements a pure ALNS ablation baseline (version file `src/experiments/ALNS/pure_alns.py`). The algorithm reuses the destroy/repair operator library, adaptive operator weights, simulated-annealing acceptance criterion, capacity repair, and archive update logic of PACO+ALNS, but removes the ant-colony construction and pheromone update. Each restart first constructs an initial solution by randomized greedy insertion (sharing the same insertion machinery as the ALNS repair operators), then performs 2-opt intra-route optimization, inter-route relocation, and ALNS local search.

To keep the configuration comparable with W8, the number of restarts of Pure ALNS defaults to the outer-iteration budget `max_iter` of PACO+ALNS (100 in this paper); the ALNS iteration count uses the same scale-adaptive default, and the tardiness penalty weight `tard_penalty_truck=10` in the acceptance score is also fixed to be the same as W8. Apart from the construction method, the parameters of the two algorithms are identical.

#### Construction

Customers are shuffled randomly and inserted one by one into the best feasible insertion position among all truck routes by minimum incremental cost; drone cooperative insertions enter the candidate set only when the range, payload, time-window, and waiting-threshold constraints are satisfied. If the best insertion fails, the customer is appended to the route with the smallest current load as a fallback.

#### Local Search and Archive

The destroy operators (surgical/random/worst/related/route), repair operators (drone-aware/greedy/regret), simulated-annealing acceptance, and adaptive weight updates are identical to PACO+ALNS. After each restart, only solutions without missing customers or overload are merged into the archive by dominance; the archive is pruned by crowding distance when it exceeds its capacity.

**Algorithm 4 Pure ALNS Main Loop**

```
Input: problem model, number of restarts S, ALNS iterations A
1. Initialize operator weights and an empty archive
2. for s = 1 to S:
3.    Construct an initial solution by randomized greedy insertion
4.    2-opt intra-route optimization; inter-route relocation
5.    ALNS(A): destroy-repair-capacity repair-simulated annealing acceptance
6.    If feasible, update the archive by dominance
7. end for
8. Return the non-dominated solutions in the archive
```

#### Complexity

Greedy insertion constructs an initial solution in about $O(n^2)$; the ALNS part is the same as PACO+ALNS. Since there is no multi-ant construction or pheromone update, the total runtime is significantly lower than PACO+ALNS.

## 5 Experimental Design

### 5.1 Dataset

The Solomon benchmark contains 56 instances in total:

| Family | Type 1 | Type 2 | Total |
|--------|--------|--------|-------|
| C | C101-C109 (9) | C201-C208 (8) | 17 |
| R | R101-R112 (12) | R201-R211 (11) | 23 |
| RC | RC101-RC108 (8) | RC201-RC208 (8) | 16 |
| Total | 29 | 27 | 56 |

Each instance text file contains 100 customers (verified file by file). This experiment covers the 54 instances of the C, R, and RC families and truncates them at customer sizes of 25/50/100, forming 54 x 3 x 2 = 324 configurations.

### 5.2 Parameter Settings

| Parameter | Value |
|-----------|-------|
| Customer size | 25 / 50 / 100 |
| Drone endurance | medium / high |
| Number of vehicles | type 1: ceil(n/10); type 2: ceil(n/25), at least 2 |
| Repeats | 3 |
| Iterations/generations | 100 |
| NSGA-II population | 100 |
| PACO ants | 30 |
| PACO+ALNS ants | adaptive (20-40) |
| Pure ALNS restarts | 100 (same as the PACO+ALNS outer-iteration budget) |
| Pure ALNS ALNS iterations | same as PACO+ALNS (scale-adaptive) |
| Pure ALNS tardiness weight | tard_penalty_truck=10 (same as PACO+ALNS) |
| Random seed | fixed and reproducible for each configuration |

### 5.3 Computing Environment

Python 3.13, NumPy, Matplotlib, and DEAP; 18-core parallelism (multiprocessing, 18 workers). Results are written to JSON task by task, with resume support.

### 5.4 Evaluation Metrics

- Cost: total transportation cost of the model;
- Tardiness: priority-weighted time-window violation;
- Hypervolume (HV): computed with the shared reference point stored by the three-algorithm comparison (the maximum of each objective over the original three-algorithm solution sets, multiplied by 1.1); Pure ALNS is evaluated with the same reference point so that the HV values of the four algorithms are directly comparable;
- Runtime: mean seconds per run;
- Feasibility: number of solutions with missing customers and number with capacity overload;
- Statistical indicators: per-configuration "best counts" and average ranks (1 is best; ties are all counted as best). Means/medians in grouped statistics are based on the per-configuration `mean_cost`, `mean_tardiness`, and `mean_hv`. Configurations where Pure ALNS has no feasible solution are excluded from its statistics and rankings.

## 6 Experimental Results and Analysis

### 6.1 Overall Performance

| Algorithm | Mean Cost | Median Cost | Mean Tardiness | Median Tardiness | Cost Best Count | Tardiness Best Count | HV Best Count | Cost Rank | Tardiness Rank | HV Rank | Time (s) |
|-----------|-----------|-------------|----------------|------------------|-----------------|----------------------|---------------|-----------|----------------|---------|----------|
| NSGA-II | 10725.0 | 793.7 | 4647.6 | 793.0 | 0 | 209 | 5 | 3.98 | 1.63 | 3.66 | 13.1 |
| PACO | 409.2 | 370.9 | 4342.4 | 2196.2 | 102 | 34 | 28 | 1.71 | 3.35 | 2.01 | 275.6 |
| PACO+ALNS | 402.8 | 345.2 | 2411.7 | 1287.8 | 222 | 63 | 291 | 1.31 | 2.19 | 1.10 | 467.2 |
| Pure ALNS | 667.4 | 461.1 | 4912.6 | 1528.5 | 0 | 27 | 0 | 2.98 | 2.82 | 3.22 | 29.7 |

Note: the mean cost of NSGA-II is substantially inflated by the penalties of capacity-overloaded solutions in 82 configurations, and the median cost (793.7) better reflects its typical level. PACO and PACO+ALNS have no overloaded or missing-customer solutions in any configuration. All solutions returned by Pure ALNS are free of overload and missing customers, but no feasible solution was found in 6 100c C2 wide-time-window configurations, which are excluded from its cost/tardiness statistics and rankings. Pure ALNS ranks below the other three algorithms in cost and HV, while its runtime is about 1/16 of PACO+ALNS.

### 6.2 By Instance Family

| Family | Indicator | NSGA-II | PACO | PACO+ALNS | Pure ALNS |
|--------|-----------|---------|------|-----------|-----------|
| C | Median cost | 730.3 | 318.0 | **311.8** | 398.2 |
| C | Median tardiness | 3634.4 | 5669.0 | **2880.9** | 5588.1 |
| C | Mean time (s) | 13.0 | 291.5 | 465.2 | 24.4 |
| R | Median cost | 800.7 | 379.9 | **359.8** | 657.2 |
| R | Median tardiness | **412.8** | 1441.7 | 818.2 | 1084.7 |
| R | Mean time (s) | 13.1 | 268.3 | 468.1 | 32.0 |
| RC | Median cost | 1322.3 | 369.8 | **369.2** | 549.1 |
| RC | Median tardiness | **308.2** | 955.9 | 523.7 | 668.2 |
| RC | Mean time (s) | 14.9 | 208.5 | 385.6 | 24.9 |

PACO+ALNS achieves the lowest median cost in all three families (C, R, and RC); NSGA-II achieves the lowest tardiness in the R and RC families, and PACO+ALNS in the C family. Pure ALNS is in the middle-to-lower range in cost for all families, while its runtime is significantly lower than the PACO-type algorithms. Overall, the cost advantage of the PACO-type algorithms is consistent across all distribution types.

### 6.3 By Customer Size

| Size | Indicator | NSGA-II | PACO | PACO+ALNS | Pure ALNS |
|------|-----------|---------|------|-----------|-----------|
| 25c | Median cost | 361.6 | 200.7 | **188.5** | 340.9 |
| 25c | Median tardiness | **217.4** | 1219.2 | 524.2 | 583.6 |
| 50c | Median cost | 793.7 | 370.9 | **345.2** | 600.5 |
| 50c | Median tardiness | **884.7** | 1901.2 | 1109.1 | 1838.5 |
| 100c | Median cost | 1776.2 | **662.1** | 676.3 | 1322.8 |
| 100c | Median tardiness | **2695.3** | 3374.5 | 2802.5 | 4036.2 |

The costs of all four algorithms increase with the customer size. PACO+ALNS has a clear cost advantage at 25c and 50c; at 100c, PACO has a slightly lower median cost (662.1 vs 676.3), but PACO+ALNS still dominates in HV. The cost of Pure ALNS is close to NSGA-II at 25c, and its gap from the PACO-type algorithms widens as the size increases.

### 6.4 Family x Size

| Family-Size | Indicator | NSGA-II | PACO | PACO+ALNS | Pure ALNS |
|-------------|-----------|---------|------|-----------|-----------|
| C-25 | Median cost/tardiness | 359.6/1085.5 | 177.3/5865.0 | **166.6**/1755.1 | 340.5/1109.9 |
| C-50 | Median cost/tardiness | 730.3/4188.7 | 318.0/4891.7 | **311.8**/2176.6 | 593.6/8609.8 |
| C-100 | Median cost/tardiness | 1786.1/11786.9 | 692.9/6916.4 | **684.9**/5316.0 | 1322.8/17110.4 |
| R-25 | Median cost/tardiness | 391.1/82.8 | 208.7/807.6 | **194.1**/380.8 | 370.2/434.7 |
| R-50 | Median cost/tardiness | 800.7/412.8 | 379.9/1414.2 | **359.8**/867.3 | 657.2/1095.3 |
| R-100 | Median cost/tardiness | 1663.9/1755.2 | **653.1**/3070.0 | 674.3/1861.6 | 1299.6/2666.2 |
| RC-25 | Median cost/tardiness | 354.5/79.5 | 242.6/551.8 | 252.9/**362.7** | 334.4/189.9 |
| RC-50 | Median cost/tardiness | 4081.7/408.4 | 369.8/1037.8 | **369.2**/575.0 | 549.1/825.8 |
| RC-100 | Median cost/tardiness | 6725.7/1585.1 | **688.1**/2327.2 | 716.3/1604.0 | 1168.5/2236.1 |

The grouped results are consistent with the overall conclusions: PACO+ALNS has the lowest cost in most groups, PACO is slightly better at R-100 and RC-100, and the tardiness advantage of NSGA-II mainly comes from the small-scale instances of the R and RC families. The cost of Pure ALNS is comparable to NSGA-II at 25c, but its gap from the PACO-type algorithms clearly widens at 50c/100c.

### 6.5 By Drone Endurance

| Endurance | Indicator | NSGA-II | PACO | PACO+ALNS | Pure ALNS |
|-----------|-----------|---------|------|-----------|-----------|
| medium | Cost/tardiness/HV rank | 3.98/1.62/3.62 | 1.77/3.27/2.00 | **1.28/2.19/1.10** | 2.96/2.90/3.26 |
| medium | Cost/tardiness/HV best counts | 0/106/3 | 45/21/14 | 117/29/145 | 0/12/0 |
| high | Cost/tardiness/HV rank | 3.97/1.63/3.69 | 1.66/3.43/2.02 | **1.35/2.19/1.10** | 3.00/2.74/3.18 |
| high | Cost/tardiness/HV best counts | 0/103/2 | 57/13/14 | 105/34/146 | 0/15/0 |

The relative rankings of the four algorithms are basically the same under both endurance settings, indicating that the algorithm performance is insensitive to drone endurance and the results are robust.

### 6.6 Hypervolume Analysis

| Size | NSGA-II | PACO | PACO+ALNS | Pure ALNS | HV Best Count (W8) |
|------|---------|------|-----------|-----------|--------------------|
| 25c | 0.38M | 1.00M | 1.33M | 0.47M | 101/108 |
| 50c | 367.6M | 377.6M | 402.8M | 295.6M | 99/108 |
| 100c | 3125.2M | 5054.6M | 4960.9M | 1122.9M | 91/108 |

HV increases sharply with the instance size, so cross-size comparisons are meaningless. Within a fixed size, PACO+ALNS has the highest mean HV at 25c and 50c; at 100c its mean HV is slightly lower than PACO, but it achieves more best counts (91 vs 17), indicating a more stable front distribution. The HV of Pure ALNS is slightly higher than NSGA-II at 25c and clearly lower than the other three algorithms at 50c/100c, indicating that its front coverage is limited by the single scalarized objective.

### 6.7 Runtime and Scalability

| Size | NSGA-II | PACO | PACO+ALNS | Pure ALNS |
|------|---------|------|-----------|-----------|
| 25c | 4.9 | 21.9 | 49.1 | 7.1 |
| 50c | 9.9 | 111.3 | 214.1 | 19.6 |
| 100c | 24.4 | 693.5 | 1138.3 | 64.3 |

NSGA-II has the smallest computational cost but its solution quality is limited by the decoding structure. PACO+ALNS has the highest per-run time but significantly improves solution quality under the same iteration budget. The runtime of Pure ALNS is only about 1/7-1/18 of PACO+ALNS, at the cost of lower cost and HV, indicating that the time savings mainly come from removing the multi-ant construction and pheromone update. The engineering optimization of the capacity-repair operator reduced the runtime by 30%-62% (about 46% on average) on the earlier 16 RC configurations, with speedups on all 16.

### 6.8 Feasibility Analysis

Across the 324 configurations, PACO and PACO+ALNS have no capacity-overloaded or missing-customer solutions; NSGA-II produces capacity-overloaded solutions in 82 configurations, concentrated in large-scale and tight-capacity instances. All solutions returned by Pure ALNS are free of overload and missing customers, but no feasible solution was found in 6 100c C2 wide-time-window configurations (the 4V combinations of C201/C202/C203/C205) within the given budget. This indicates that the "even customer distribution across routes" decoding of NSGA-II needs capacity-aware repair or penalty mechanisms to serve as a fair baseline on hard-constrained problems, and that the fixed scalarized objective and greedy construction of Pure ALNS also need improvement for wide-window, high-capacity scenarios.

### 6.9 E-CVRP Generalization

The WCCI-2020 E-CVRP benchmark (24 instances of the E/F/M/X families) is used for generalization tests. Since the model does not include energy and recharging constraints, the validation is equivalent to a CVRP relaxation and the cost is measured by distance. On the four instances with published optimal values, the mean gap of each algorithm is as follows:

| Algorithm | Mean gap | Instances reaching optimum/best-known |
|-----------|----------|--------------------------------------|
| NSGA-II | +67.44% | 0/4 |
| PACO | +9.63% | 0/4 |
| PACO+ALNS | -1.01% | 3/4 |
| Pure ALNS | +3.32% | 1/4 |

On the large X instances, PACO+ALNS finds feasible solutions on all of them; PACO and NSGA-II fail to find feasible solutions on X-n759, X-n830, and X-n920 within the given budget; Pure ALNS, like PACO+ALNS, finds feasible solutions on all 24 instances. Pure ALNS uses the same outer-iteration budgets as PACO+ALNS (30 iterations x 3 runs for default instances, 10 iterations x 2 runs for supplementary medium instances, and 5 iterations x 1 run for large X instances), achieves a mean gap of +3.32% on the four instances with published optimal values (reaching the optimum on 1 of them), and has a mean runtime of 75.2 seconds, significantly lower than the 1105.3 seconds of PACO+ALNS. Negative gaps are obtained under the relaxation with energy constraints removed and must not be interpreted as beating the published EVRP optimum.

### 6.10 Comprehensive Discussion

1. **Objective trade-off**: the solutions of NSGA-II concentrate in the low-tardiness region at the cost of higher cost; PACO and PACO+ALNS achieve lower cost, and PACO+ALNS improves both objectives through ALNS, giving the best overall HV. Pure ALNS has clearly narrower front coverage because it uses a fixed scalarized objective.
2. **Feasibility**: capacity is a hard constraint in real operations. NSGA-II should be used with penalty or repair mechanisms, and feasibility statistics should be reported in direct comparisons; Pure ALNS has no feasible solution on 6 wide-time-window large-scale configurations, which should also be stated when reporting feasibility.
3. **Efficiency and quality**: the per-run time of PACO+ALNS is about 1.5-2 times that of PACO, but quality and HV are significantly higher under the same iteration budget. Pure ALNS under the same parameter configuration takes only about 1/7-1/18 of the time of PACO+ALNS but is clearly inferior in cost and HV, showing that the directed search ability provided by the PACO construction and pheromone is the key to quality improvement.
4. **Generalization**: the E-CVRP validation shows that PACO+ALNS remains feasible and competitive on larger instances with different distributions.

### 6.11 Representative Visualizations

To illustrate the differences among the algorithms, this section presents three groups of representative figures.

**Pareto-front figures**: Figs. 1-3 show the joint four-algorithm Pareto fronts for C101, R101, and RC102 at 100 customers with medium endurance. The fronts of PACO+ALNS and PACO lie in the lower-left region (lower cost), NSGA-II extends along the lower tardiness axis, Pure ALNS lies in the higher-cost region with limited coverage, and PACO+ALNS has the widest coverage.

![Fig. 1 C101 100c medium four-algorithm Pareto front](PACO+ALNS/results/20260812_4alg/pareto_compare_100c_C101_10V_medium.png)

![Fig. 2 R101 100c medium four-algorithm Pareto front](PACO+ALNS/results/20260812_4alg/pareto_compare_100c_R101_10V_medium.png)

![Fig. 3 RC102 100c medium four-algorithm Pareto front](PACO+ALNS/results/20260812_4alg/pareto_compare_100c_RC102_10V_medium.png)

**Convergence curves**: Figs. 4-5 show the best cost and HV of C101 (25 customers, medium) under iteration budgets of 5/10/20/30/50/100. PACO+ALNS reaches a lower cost within a small budget, and the HV grows steadily with iterations; NSGA-II converges slowly with a generally higher cost; PACO lies in between.

![Fig. 4 Convergence curves: best cost](PACO+ALNS/results/20260809_w8/figures/convergence_cost_25c_C101_medium.png)

![Fig. 5 Convergence curves: hypervolume](PACO+ALNS/results/20260809_w8/figures/convergence_hv_25c_C101_medium.png)

**Route visualization**: Figs. 6-11 show the min-cost and compromise routes of C101 (100 customers, medium) for the first three algorithms. Solid lines are truck main routes, dashed/dotted lines are drone launch and return legs, and gray crosses are unserved customers (none in this experiment).

![Fig. 6 NSGA-II min-cost route](PACO+ALNS/results/20260809_w8/figures/routes_nsga2_C101_100c_medium_min_cost.png)

![Fig. 7 NSGA-II compromise route](PACO+ALNS/results/20260809_w8/figures/routes_nsga2_C101_100c_medium_compromise.png)

![Fig. 8 PACO min-cost route](PACO+ALNS/results/20260809_w8/figures/routes_imp2_C101_100c_medium_min_cost.png)

![Fig. 9 PACO compromise route](PACO+ALNS/results/20260809_w8/figures/routes_imp2_C101_100c_medium_compromise.png)

![Fig. 10 PACO+ALNS min-cost route](PACO+ALNS/results/20260809_w8/figures/routes_w8_C101_100c_medium_min_cost.png)

![Fig. 11 PACO+ALNS compromise route](PACO+ALNS/results/20260809_w8/figures/routes_w8_C101_100c_medium_compromise.png)

The route figures show that NSGA-II tends to use more trucks to share customers, forming longer detours; PACO and PACO+ALNS use drone cooperative missions more actively to compress the truck main routes, and the compromise solution of PACO+ALNS achieves a more balanced cost-tardiness allocation. The route figures of Pure ALNS and all 324 four-algorithm comparison plots are available in the `PACO+ALNS/results/20260812_4alg/` directory.

## 7 Limitations and Future Work

This paper still has the following limitations:

1. The model does not include energy consumption and recharging-station constraints, so the E-CVRP validation can only serve as a CVRP-relaxation test;
2. Each configuration is repeated only 3 times, so the statistical power is limited; future work can increase the number of repeats and provide confidence intervals and significance tests;
3. The decoder of NSGA-II does not embed capacity constraints, which gives it a feasibility disadvantage as a baseline; capacity-aware repair can be introduced later;
4. Pure ALNS uses a fixed scalarized objective and fails to find feasible solutions on 6 100c C2 wide-time-window configurations, indicating that single-weight ALNS has insufficient search coverage for wide-window, high-capacity scenarios;
5. No parameter-sensitivity analysis or cross-operator ablation experiments were conducted;
6. The algorithms have not been compared end to end with commercial solvers or other state-of-the-art methods (e.g., parallel variants of large neighborhood search).

Future work includes extending equal-time-budget comparisons (especially equal-time comparisons between Pure ALNS and PACO+ALNS), introducing energy constraints and recharging stations, using machine learning to guide operator selection, and extending the methods to multi-depot and dynamic-demand scenarios.

## 8 Conclusion

This paper systematically compares four algorithms, NSGA-II, PACO, PACO+ALNS, and Pure ALNS, on the untested Solomon benchmark instances. PACO+ALNS is the best overall in cost and hypervolume; NSGA-II is the best in the tardiness objective but at a high cost and feasibility penalty; PACO provides a balanced but slightly weaker trade-off; Pure ALNS, as an ablation baseline, verifies the contribution of the PACO construction and pheromone to solution quality, but its feasibility on some wide-time-window large-scale configurations needs improvement. The complete grouped statistics, algorithmic pseudocode, and reproducible configurations provide a benchmark for multi-objective algorithm research on truck-drone cooperative delivery.

## References

[1] M. Mavrovouniotis, C. Menelaou, S. Timotheou, G. Ellinas, C. Panayiotou and M. Polycarpou, "A Benchmark Test Suite for the Electric Capacitated Vehicle Routing Problem," 2020 IEEE Congress on Evolutionary Computation (CEC), 2020, pp. 1-8.
[2] M. Mavrovouniotis et al., "Benchmark set for the IEEE WCCI-2020 competition on evolutionary computation for the electric vehicle routing problem," Dept. Elect. Comput. Eng., Univ. Cyprus, 2020. GitHub: https://github.com/Mavrovouniotis/e-cvrp_benchmark_instances
[3] M. M. Solomon, "Algorithms for the vehicle routing and scheduling problems with time window constraints," Operations Research, vol. 35, no. 2, pp. 254-265, 1987.
[4] K. Deb, A. Pratap, S. Agarwal and T. Meyarivan, "A fast and elitist multiobjective genetic algorithm: NSGA-II," IEEE Transactions on Evolutionary Computation, vol. 6, no. 2, pp. 182-197, 2002.
[5] S. Ropke and D. Pisinger, "An adaptive large neighborhood search heuristic for the pickup and delivery problem with time windows," Transportation Science, vol. 40, no. 4, pp. 455-472, 2006.
[6] M. Dorigo, V. Maniezzo and A. Colorni, "Ant system: optimization by a colony of cooperating agents," IEEE Transactions on Systems, Man, and Cybernetics, Part B, vol. 26, no. 1, pp. 29-41, 1996.
[7] D. N. Das, R. Sewani, J. Wang and M. K. Tiwari, "Synchronized truck and drone routing in package delivery logistics," IEEE Transactions on Intelligent Transportation Systems, vol. 22, no. 9, pp. 5772-5782, 2021, doi: 10.1109/TITS.2020.2992549.

## Appendix A: Reproduction Notes

Experiment scripts and results:

- Comparison script: `src/experiments/PACO+ALNS/compare_three_algorithms.py`
- Full Solomon run: `python compare_three_algorithms.py --all-solomon --runs 3 --max-iter 100 --workers 18 --outdir 20260809_w8 --w7-module w8 --no-route-plots`
- Result directory: `src/experiments/PACO+ALNS/results/20260809_w8`
- Combined result: `compare_three_20260809_w8.json` (324 configurations x 3 algorithms)
- Per-configuration results: `<configuration>_<algorithm>.json`, 972 files in total
- Pure ALNS solver: `src/experiments/ALNS/pure_alns.py`
- Four-algorithm comparison script: `src/experiments/ALNS/compare_pure_alns.py`
- Four-algorithm run: `python compare_pure_alns.py --runs 3 --max-iter 100 --workers 16 --outdir 20260812_4alg`
- Four-algorithm result directory: `src/experiments/PACO+ALNS/results/20260812_4alg`
- Four-algorithm combined result: `compare_pure_alns_20260812_4alg.json` (324 configurations x 4 algorithms)
- Four-algorithm per-configuration results: `<configuration>_<algorithm>.json`, 1296 files in total

Each per-configuration JSON contains `mean_cost`, `std_cost`, `mean_tardiness`, `std_tardiness`, `mean_hv`, `std_hv`, `mean_solve_time`, `mean_n_solutions`, `mean_drone_missions`, `mean_routes`, `n_missing_solutions`, `n_overload_solutions`, `all_costs`, `all_tardiness`, `all_pareto_fronts`, `all_hypervolumes`, `hv_reference`, and algorithm/configuration metadata.

## Appendix B: E-CVRP Validation Notes

- Data source: https://github.com/Mavrovouniotis/e-cvrp_benchmark_instances (24 `.evrp` files)
- Validation script: `src/experiments/PACO+ALNS/evrp_validation.py`
- Result directory: `src/experiments/PACO+ALNS/results/20260806_evrp_w8`
- Pure ALNS: `src/experiments/ALNS/pure_alns.py`; results are stored in the same directory as the other methods (`<instance>_alns.json`, 24 files), with budgets identical to the other methods
- Protocol: wide time windows (tardiness is always 0), drones disabled, truck fixed cost 0, variable cost 1, so cost equals distance; F-n140 is run with 7 vehicles (its filename/comment is k7 and the declared 5 vehicles are infeasible).
