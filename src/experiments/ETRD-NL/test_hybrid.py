# """
# ETRD-NL Hybrid Solver Test
# Test hybrid MILP + ALNS collaborative solver
# """
# import sys
# import os
# import time
# import json

# # Add project path
# script_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.insert(0, script_dir)

# print("="*60)
# print("ETRD-NL Hybrid Solver Test Pipeline")
# print("="*60)

# # Import modules
# print("\n1. Import required modules...")
# from data.instance_generator import ETRDInstanceGenerator
# from hybrid_solver.hybrid_solver import ETRD_Hybrid_Solver
# print("   ✓ All modules imported successfully")

# # Generate test instances
# print("\n2. Initialize instance generator (seed=42)")
# gen = ETRDInstanceGenerator(seed=42)

# # Instance config: (instance_name, generate_func, time_limit_sec)
# instances = [
#     ('tiny', gen.generate_tiny_instance(), 60),      # 7 customers
#     ('small', gen.generate_small_instance(), 120),   # 15 customers
#     ('medium', gen.generate_medium_instance(), 180), # 30 customers
#     ('large', gen.generate_large_instance(), 300),  # 50 customers
# ]

# # Create output dir if not exists
# os.makedirs("results", exist_ok=True)

# print("\n3. Start solving pipeline")
# print("="*60)

# results = []
# total_start_time = time.time()

# for name, raw_instance, time_limit in instances:
#     print(f"\n{'='*60}")
#     print(f"Running {name.upper()} Instance | Time Limit: {time_limit}s")
#     print(f"{'='*60}")
    
#     try:
#         # Fix critical bug: convert service_times list to dict
#         if isinstance(raw_instance.get("service_times"), list):
#             raw_instance["service_times"] = {cid + 1: t for cid, t in enumerate(raw_instance["service_times"])}
#         instance = raw_instance
        
#         # Print basic instance info
#         print(f"Customer count: {instance['n_customers']}")
#         print(f"Charging station count: {instance['n_charging_stations']}")
#         print(f"Service_times type: {type(instance['service_times'])} (converted to dict)")

#         # Initialize hybrid solver
#         solver = ETRD_Hybrid_Solver(instance, strategy='auto')
#         solution = solver.solve(time_limit=time_limit)

#         if solution:
#             solver.print_solution()
#             # Record result data
#             record = {
#                 'instance_name': name,
#                 'customer_num': instance['n_customers'],
#                 'makespan': round(solution['makespan'], 2),
#                 'used_method': solver.stats['method_used'],
#                 'solve_seconds': round(solver.stats['solve_time'], 2)
#             }
#             results.append(record)

#             # Save full solution to json
#             output_path = os.path.join("results", f"{name}_solution.json")
#             with open(output_path, "w", encoding="utf-8") as f:
#                 json.dump({
#                     "instance_info": {
#                         "name": name,
#                         "n_customers": instance["n_customers"],
#                     },
#                     "solve_stat": solver.stats,
#                     "solution_detail": solution
#                 }, f, indent=4, ensure_ascii=False)
#             print(f"\n✓ Solution saved to {output_path}")
#         else:
#             print(f"✗ {name} instance failed to find feasible solution")

#     except Exception as e:
#         print(f"\n✗ Runtime error on {name} instance: {str(e)}")
#         import traceback
#         traceback.print_exc()

# # Print final summary table
# total_elapsed = round(time.time() - total_start_time, 2)
# print("\n" + "="*70)
# print("FINAL RESULTS SUMMARY | Total Runtime: {:.2f}s".format(total_elapsed))
# print("="*70)
# header = f"{'Instance':<10}{'Customers':<12}{'Makespan':<12}{'Method Used':<14}{'Time(s)':<10}"
# print(header)
# print("-"*70)
# for item in results:
#     line = (
#         f"{item['instance_name']:<10}"
#         f"{item['customer_num']:<12}"
#         f"{item['makespan']:<12.2f}"
#         f"{item['used_method']:<14}"
#         f"{item['solve_seconds']:<10.2f}"
#     )
#     print(line)

# print("\n" + "="*60)
# print("ALL TESTS COMPLETED")
# print("="*60)
"""
ETRD-NL Hybrid Solver Test
Test hybrid MILP + ALNS collaborative solver
"""
import sys
import os
import time
import json

# Recursively convert all set objects to list for JSON serialization
def convert_set_to_list(obj):
    if isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, dict):
        return {k: convert_set_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_set_to_list(item) for item in obj]
    else:
        return obj

# Add project path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

print("="*60)
print("ETRD-NL Hybrid Solver Test Pipeline")
print("="*60)

# Import modules
print("\n1. Import required modules...")
from data.instance_generator import ETRDInstanceGenerator
from hybrid_solver.hybrid_solver import ETRD_Hybrid_Solver
print("   ✓ All modules imported successfully")

# Generate test instances
print("\n2. Initialize instance generator (seed=42)")
gen = ETRDInstanceGenerator(seed=42)

# Instance config: (instance_name, generate_func, time_limit_sec)
instances = [
    ('tiny', gen.generate_tiny_instance(), 60),      # 7 customers
    ('small', gen.generate_small_instance(), 120),   # 15 customers
    ('medium', gen.generate_medium_instance(), 180), # 30 customers
    ('large', gen.generate_large_instance(), 300),  # 60 customers
    
    ('c50', gen.generate_50_instance(), 300),    # 50  ALNS
    ('c100', gen.generate_100_instance(), 300),  # 100 ALNS
    ('c200', gen.generate_200_instance(), 300),  # 200 ALNS
]

# Create output dir if not exists
os.makedirs("results", exist_ok=True)

print("\n3. Start solving pipeline")
print("="*60)

results = []
total_start_time = time.time()

for name, raw_instance, time_limit in instances:
    print(f"\n{'='*60}")
    print(f"Running {name.upper()} Instance | Time Limit: {time_limit}s")
    print(f"{'='*60}")
    
    try:
        # Fix critical bug: convert service_times list to dict
        if isinstance(raw_instance.get("service_times"), list):
            raw_instance["service_times"] = {cid + 1: t for cid, t in enumerate(raw_instance["service_times"])}
        instance = raw_instance
        
        # Print basic instance info
        print(f"Customer count: {instance['n_customers']}")
        print(f"Charging station count: {instance['n_charging_stations']}")
        print(f"Service_times type: {type(instance['service_times'])} (converted to dict)")

        # Initialize hybrid solver
        solver = ETRD_Hybrid_Solver(instance, strategy='auto')
        solution = solver.solve(time_limit=time_limit)

        if solution:
            solver.print_solution()
            # Record result data
            record = {
                'instance_name': name,
                'customer_num': instance['n_customers'],
                'makespan': round(solution['makespan'], 2),
                'used_method': solver.stats['method_used'],
                'solve_seconds': round(solver.stats['solve_time'], 2)
            }
            results.append(record)

            # Save full solution: convert set to list before dump
            output_data = convert_set_to_list({
                "instance_info": {
                    "name": name,
                    "n_customers": instance["n_customers"],
                },
                "solve_stat": solver.stats,
                "solution_detail": solution
            })
            output_path = os.path.join("results", f"{name}_solution.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=4, ensure_ascii=False)
            print(f"\n✓ Solution saved to {output_path}")
        else:
            print(f"✗ {name} instance failed to find feasible solution")

    except Exception as e:
        print(f"\n✗ Runtime error on {name} instance: {str(e)}")
        import traceback
        traceback.print_exc()

# Print final summary table
total_elapsed = round(time.time() - total_start_time, 2)
print("\n" + "="*70)
print("FINAL RESULTS SUMMARY | Total Runtime: {:.2f}s".format(total_elapsed))
print("="*70)
header = f"{'Instance':<10}{'Customers':<12}{'Makespan':<12}{'Method Used':<14}{'Time(s)':<10}"
print(header)
print("-"*70)
for item in results:
    line = (
        f"{item['instance_name']:<10}"
        f"{item['customer_num']:<12}"
        f"{item['makespan']:<12.2f}"
        f"{item['used_method']:<14}"
        f"{item['solve_seconds']:<10.2f}"
    )
    print(line)

print("\n" + "="*60)
print("ALL TESTS COMPLETED")
print("="*60)