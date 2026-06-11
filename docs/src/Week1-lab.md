# Week1 Lab: Solver Setup and Baseline Smoke Test
## Step 1: Choose a Starter Path
---
**OR-Tools**
## Step 2: Environment Record
---
- **operating system:**Ubuntu-22.04
- **Python version:**Python 3.10.12
- **package manager:** pip
- **solver or codebase version:** ortools==9.15.6755
- **exact install commands:**
`pip3 install ortools` 
- **hardware used for runtime:**
CPU: Intel(R) Core(TM) Ultra 5 125H (3.60 GHz)
RAM: 32.0 GB
Graphics: Intel(R) Arc(TM) integrated Graphics (no dedicated NVIDIA GPU)
Storage: 954 GB SSD
## Step 3: Smoke Test
---
- **command:**`time python3 week1_smoke_test.py`
- **instance name and size:** tiny_pdp_16nodes, total 17 nodes (1 depot node 0 + 16 customer nodes), 4 vehicles, 8 pickup-delivery pairs
- **objective value:** 226116
- **feasibility status:** Feasible
- **runtime:**0.0194s
- **route plot or textual route output:**
Route for vehicle 0:
 0 ->  13 ->  15 ->  11 ->  12 -> 0
Distance of the route: 1552m

Route for vehicle 1:
 0 ->  5 ->  2 ->  10 ->  16 ->  14 ->  9 -> 0
Distance of the route: 2192m

Route for vehicle 2:
 0 ->  4 ->  3 -> 0
Distance of the route: 1392m

Route for vehicle 3:
 0 ->  7 ->  1 ->  6 ->  8 -> 0
Distance of the route: 1780m

Total Distance of all routes: 6916m
## Step 4: Reflection
---
In this PDP pickup-and-delivery model, the maximum travel distance limit per vehicle is the easiest constraint to understand, as it directly restricts the total driving range of each vehicle. The output objective value is confusing, since it combines route length and span penalty coefficients and differs from the plain total travel distance. The baseline goal for Week 2 is to extend the basic PDP model to solve truck-drone collaborative routing, and complete construction and feasibility tests for multi-vehicle instances.
