"""Diagnostic: trace drone usage in NSGA2 decoding."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solomon_loader import SolomonLoader
import random
import numpy as np

loader = SolomonLoader()
m = loader.load_rc1_instance(25, 1, 2)

print("Drone speed:", m.get_vehicle_speed('drone'))
print("Truck speed:", m.get_vehicle_speed('truck'))
print("Drone range:", m.drone_range)
print("Drone capacity:", m.drones[0].capacity)
print("n_customers:", m.get_number_of_customers())
print("n_trucks:", m.get_number_of_trucks())
print("n_drones:", m.get_number_of_drones())
print()

# Test with absurd speed
m.drones[0].speed = 50000.0
print("After speed=50000:")
print("Drone speed:", m.get_vehicle_speed('drone'))
print()

# Simulate NSGA2 decode logic for a random individual
n = m.get_number_of_customers()
n_trucks = m.get_number_of_trucks()

for trial in range(5):
    order = list(range(n))
    random.shuffle(order)
    mode = [0 if random.random() < 0.6 else 1 for _ in range(n)]
    
    # Build routes (same as NSGA2)
    chunk_size = max(1, len(order) // n_trucks)
    routes_customers = []
    for tid in range(n_trucks):
        start = tid * chunk_size
        end = len(order) if tid == n_trucks - 1 else (tid + 1) * chunk_size
        routes_customers.append(order[start:end].copy())
    
    # Count drone attempts and successes
    drone_attempts = 0
    drone_success = 0
    drone_range_fail = 0
    drone_time_fail = 0
    
    for route_custs in routes_customers:
        if len(route_custs) < 3:
            continue
        i = 0
        while i < len(route_custs):
            if i + 2 < len(route_custs):
                target_cust = route_custs[i+1]
                target_idx = order.index(target_cust)
                is_drone = (mode[target_idx] == 1)
                
                if is_drone:
                    drone_attempts += 1
                    ci = m.customers[route_custs[i]]
                    cj = m.customers[target_cust]
                    ck = m.customers[route_custs[i+2]]
                    
                    d1 = ci.distance_to(cj)
                    d2 = cj.distance_to(ck)
                    d_truck = ci.distance_to(ck)
                    
                    range_ok = (d1 + d2) <= m.drone_range
                    time_ok = True  # speed is 50000, drone_time ~= 0
                    demand_ok = cj.demand <= m.drones[0].capacity
                    
                    if range_ok and demand_ok and time_ok:
                        drone_success += 1
                    else:
                        if not range_ok:
                            drone_range_fail += 1
                        if not time_ok:
                            drone_time_fail += 1
            i += 1
    
    print(f"Trial {trial}: drone_attempts={drone_attempts}, success={drone_success}, range_fail={drone_range_fail}, time_fail={drone_time_fail}")

# Also check: what if we look at closest triplets?
print()
print("=== Closest triplets analysis ===")
customers = m.customers
# For each customer, find the closest feasible launch and return points
feasible_count = 0
for j in range(n):
    cj = customers[j]
    best_dist = float('inf')
    best_pair = None
    for i in range(n):
        if i == j: continue
        for k in range(n):
            if k == j or k == i: continue
            d1 = customers[i].distance_to(cj)
            d2 = cj.distance_to(customers[k])
            if (d1 + d2) <= m.drone_range:
                feasible_count += 1
                if d1 + d2 < best_dist:
                    best_dist = d1 + d2
                    best_pair = (i, k)
    if best_pair:
        print(f"  Customer {j}: best d1+d2={best_dist:.2f} km (launch={best_pair[0]}, return={best_pair[1]})")
    else:
        print(f"  Customer {j}: NO feasible triplet!")

print(f"\nTotal feasible triplets: {feasible_count} / {n * (n-1) * (n-2)}")