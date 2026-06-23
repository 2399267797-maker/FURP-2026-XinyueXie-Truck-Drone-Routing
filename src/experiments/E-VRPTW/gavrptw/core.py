# -*- coding: utf-8 -*-

'''gavrptw/core.py'''

import os
import io
import random
import numpy as np
from csv import DictWriter
from deap import base, creator, tools
from . import BASE_DIR
from .utils import make_dirs_for_file, exist, load_instance, merge_rules


def ind2route(individual, instance, energy_constraint=False, battery_capacity=None, energy_consumption=None,
              hard_time_window=False, hard_energy_constraint=False): # 【新增：硬约束开关参数】
    '''gavrptw.core.ind2route(...)'''
    route = []
    vehicle_capacity = instance['vehicle_capacity']
    depart_due_time = instance['depart']['due_time']
    # Initialize a sub-route
    sub_route = []
    vehicle_load = 0
    elapsed_time = 0
    last_customer_id = 0
    # Initialize energy if energy constraint is enabled
    remaining_energy = battery_capacity if energy_constraint else None
    
    for customer_id in individual:
        # Calculate distance to next customer
        distance_to_customer = instance['distance_matrix'][last_customer_id][customer_id]
        
        # 【修改：判断电量是否作为硬约束强制切分路线】
        is_out_of_energy = False
        if energy_constraint and hard_energy_constraint:
            # Energy needed to reach customer and return to depot
            distance_to_depot = instance['distance_matrix'][customer_id][0]
            total_energy_needed = (distance_to_customer + distance_to_depot) * energy_consumption
            
            # If not enough energy and it's a HARD constraint, trigger route split
            if remaining_energy < total_energy_needed:
                is_out_of_energy = True
        
        # Update vehicle load
        demand = instance[f'customer_{customer_id}']['demand']
        updated_vehicle_load = vehicle_load + demand
        # Update elapsed time
        service_time = instance[f'customer_{customer_id}']['service_time']
        return_time = instance['distance_matrix'][customer_id][0]
        updated_elapsed_time = elapsed_time + distance_to_customer + service_time + return_time
        
        # POMO-aligned hard TW constraint: check if late BEFORE adding to route
        is_late = False
        if hard_time_window:
            arrival_time = elapsed_time + distance_to_customer
            if arrival_time > instance[f'customer_{customer_id}']['due_time']:
                is_late = True
        
        # Validate vehicle load, elapsed time, and time window AND energy
        if (updated_vehicle_load <= vehicle_capacity) and (updated_elapsed_time <= depart_due_time) and (not is_late) and (not is_out_of_energy):
            # Add to current sub-route
            sub_route.append(customer_id)
            vehicle_load = updated_vehicle_load
            elapsed_time = updated_elapsed_time - return_time
            # Update remaining energy if constraint is enabled
            if energy_constraint:
                remaining_energy -= distance_to_customer * energy_consumption
        else:
            # Save current sub-route
            route.append(sub_route)
            # Initialize a new sub-route and add to it
            sub_route = [customer_id]
            vehicle_load = demand
            elapsed_time = instance['distance_matrix'][0][customer_id] + service_time
            # Reset energy if constraint is enabled (recharge at depot)
            if energy_constraint:
                remaining_energy = battery_capacity - instance['distance_matrix'][0][customer_id] * energy_consumption
        # Update last customer ID
        last_customer_id = customer_id
    if sub_route != []:
        # Save current sub-route before return if not empty
        route.append(sub_route)
    return route


def print_route(route, merge=False):
    '''gavrptw.core.print_route(route, merge=False)'''
    route_str = '0'
    sub_route_count = 0
    for sub_route in route:
        sub_route_count += 1
        sub_route_str = '0'
        for customer_id in sub_route:
            sub_route_str = f'{sub_route_str} - {customer_id}'
            route_str = f'{route_str} - {customer_id}'
        sub_route_str = f'{sub_route_str} - 0'
        if not merge:
            print(f'  Vehicle {sub_route_count}\'s route: {sub_route_str}')
        route_str = f'{route_str} - 0'
    if merge:
        print(route_str)


def eval_vrptw(individual, instance, unit_cost=1.0, init_cost=0, wait_cost=0, delay_cost=0, 
               energy_constraint=False, battery_capacity=None, energy_consumption=None,
               hard_time_window=False, hard_energy_constraint=False, energy_penalty_cost=5.0): # 【新增：电量软惩罚系数】
    '''gavrptw.core.eval_vrptw(...)'''
    total_cost = 0
    route = ind2route(individual, instance, energy_constraint, battery_capacity, energy_consumption,
                      hard_time_window=hard_time_window, hard_energy_constraint=hard_energy_constraint)
    total_cost = 0
    for sub_route in route:
        sub_route_time_cost = 0
        sub_route_energy_cost = 0 # 【新增：追踪子路线的缺电罚单】
        sub_route_distance = 0
        elapsed_time = 0
        last_customer_id = 0
        remaining_energy = battery_capacity if energy_constraint else None
        
        for customer_id in sub_route:
            # Calculate section distance
            distance = instance['distance_matrix'][last_customer_id][customer_id]
            # Update sub-route distance
            sub_route_distance = sub_route_distance + distance
            # Calculate time cost
            arrival_time = elapsed_time + distance
            
            # POMO-aligned hard TW constraint: infeasible if late
            if hard_time_window:
                if arrival_time > instance[f'customer_{customer_id}']['due_time']:
                    # Hard constraint violated - return infeasible
                    return (0.0,)  # Zero fitness = infeasible
            else:
                # Soft TW constraint: penalty for late/early arrival (已经包含了你的时间软惩罚逻辑)
                time_cost = wait_cost * max(instance[f'customer_{customer_id}']['ready_time'] - arrival_time, 0) + \
                            delay_cost * max(arrival_time - instance[f'customer_{customer_id}']['due_time'], 0)
                sub_route_time_cost = sub_route_time_cost + time_cost
            
            # 【新增：计算耗电与软惩罚】
            if energy_constraint:
                remaining_energy -= distance * energy_consumption
                if not hard_energy_constraint and remaining_energy < 0:
                    # 如果变成负数，累加巨额罚金 (类似 DRL 中的惩罚)
                    sub_route_energy_cost += (-remaining_energy) * energy_penalty_cost
            
            # Update elapsed time
            elapsed_time = arrival_time + instance[f'customer_{customer_id}']['service_time']
            # Update last customer ID
            last_customer_id = customer_id
            
        # Calculate transport cost (return to depot)
        return_distance = instance['distance_matrix'][last_customer_id][0]
        sub_route_distance = sub_route_distance + return_distance
        
        # 【新增：回程也要算电量，如果因为回不来导致没电，一样要扣分】
        if energy_constraint:
            remaining_energy -= return_distance * energy_consumption
            if not hard_energy_constraint and remaining_energy < 0:
                sub_route_energy_cost += (-remaining_energy) * energy_penalty_cost
                
        sub_route_transport_cost = init_cost + unit_cost * sub_route_distance
        
        # 【修改：将电量超标的罚金算进总成本里】
        sub_route_cost = sub_route_time_cost + sub_route_transport_cost + sub_route_energy_cost
        
        # Update total cost
        total_cost = total_cost + sub_route_cost
        
    fitness = 1.0 / total_cost if total_cost > 0 else 0.0
    return (fitness, )


def cx_partially_matched(ind1, ind2):
    '''gavrptw.core.cx_partially_matched(ind1, ind2)'''
    cxpoint1, cxpoint2 = sorted(random.sample(range(min(len(ind1), len(ind2))), 2))
    part1 = ind2[cxpoint1:cxpoint2+1]
    part2 = ind1[cxpoint1:cxpoint2+1]
    rule1to2 = list(zip(part1, part2))
    is_fully_merged = False
    while not is_fully_merged:
        rule1to2, is_fully_merged = merge_rules(rules=rule1to2)
    rule2to1 = {rule[1]: rule[0] for rule in rule1to2}
    rule1to2 = dict(rule1to2)
    ind1 = [gene if gene not in part2 else rule2to1[gene] for gene in ind1[:cxpoint1]] + part2 + \
        [gene if gene not in part2 else rule2to1[gene] for gene in ind1[cxpoint2+1:]]
    ind2 = [gene if gene not in part1 else rule1to2[gene] for gene in ind2[:cxpoint1]] + part1 + \
        [gene if gene not in part1 else rule1to2[gene] for gene in ind2[cxpoint2+1:]]
    return ind1, ind2


def mut_inverse_indexes(individual):
    '''gavrptw.core.mut_inverse_indexes(individual)'''
    start, stop = sorted(random.sample(range(len(individual)), 2))
    temp = individual[start:stop+1]
    temp.reverse()
    individual[start:stop+1] = temp
    return (individual, )


def run_gavrptw(instance_name, unit_cost, init_cost, wait_cost, delay_cost, ind_size, pop_size, \
    cx_pb, mut_pb, n_gen, export_csv=False, customize_data=False, energy_constraint=False, \
    battery_capacity=None, energy_consumption=None, hard_time_window=False, 
    hard_energy_constraint=False, energy_penalty_cost=5.0): # 【新增相关参数】
    '''gavrptw.core.run_gavrptw(...)'''
    if customize_data:
        json_data_dir = os.path.join(BASE_DIR, 'data', 'json_customize')
    else:
        json_data_dir = os.path.join(BASE_DIR, 'data', 'json')
    json_file = os.path.join(json_data_dir, f'{instance_name}.json')
    instance = load_instance(json_file=json_file)
    if instance is None:
        return
    
    # Calculate battery capacity if not provided and energy constraint is enabled
    if energy_constraint and battery_capacity is None:
        problem_size = ind_size  # Assuming ind_size equals number of customers
        battery_capacity = (np.sqrt(2) * problem_size / 10) * (0.8 + random.random() * 0.4)
    
    # Calculate energy consumption if not provided and energy constraint is enabled
    if energy_constraint and energy_consumption is None:
        energy_consumption = 1.0 + random.random() * 0.2
    
    # Print constraint info if enabled
    if energy_constraint:
        if hard_energy_constraint:
            print(f'\nEnergy Constraint (E) - Hard Mode:')
            print(f'  Battery Capacity: {battery_capacity:.2f}, Consumption Rate: {energy_consumption:.2f}')
            print(f'  Rule: Forces route split when energy is insufficient')
        else:
            print(f'\nEnergy Constraint (E) - Soft Mode:')
            print(f'  Battery Capacity: {battery_capacity:.2f}, Consumption Rate: {energy_consumption:.2f}')
            print(f'  Rule: Allows negative battery but adds penalty cost (weight: {energy_penalty_cost})')
            
    if hard_time_window:
        print(f'\nTime Window Constraint (TW) - Hard Mode:')
        print(f'  Late arrival = infeasible (Fitness 0)')
    else:
        print(f'\nTime Window Constraint (TW) - Soft Mode:')
        print(f'  Late arrival = penalty cost (added to total route cost)')
    
    creator.create('FitnessMax', base.Fitness, weights=(1.0, ))
    creator.create('Individual', list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    # Attribute generator
    toolbox.register('indexes', random.sample, range(1, ind_size + 1), ind_size)
    # Structure initializers
    toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.indexes)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    
    # 【修改：将新参数注册进 evaluate 函数里】
    toolbox.register('evaluate', eval_vrptw, instance=instance, unit_cost=unit_cost, \
        init_cost=init_cost, wait_cost=wait_cost, delay_cost=delay_cost, \
        energy_constraint=energy_constraint, battery_capacity=battery_capacity, \
        energy_consumption=energy_consumption, hard_time_window=hard_time_window, \
        hard_energy_constraint=hard_energy_constraint, energy_penalty_cost=energy_penalty_cost)
        
    toolbox.register('select', tools.selRoulette)
    toolbox.register('mate', cx_partially_matched)
    toolbox.register('mutate', mut_inverse_indexes)
    pop = toolbox.population(n=pop_size)
    # Results holders for exporting results to CSV file
    csv_data = []
    print('Start of evolution')
    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    print(f'  Evaluated {len(pop)} individuals')
    # Begin the evolution
    for gen in range(n_gen):
        print(f'-- Generation {gen} --')
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cx_pb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        for mutant in offspring:
            if random.random() < mut_pb:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        print(f'  Evaluated {len(invalid_ind)} individuals')
        # The population is entirely replaced by the offspring
        pop[:] = offspring
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum([x**2 for x in fits])
        std = abs(sum2 / length - mean**2)**0.5
        print(f'  Min {min(fits)}')
        print(f'  Max {max(fits)}')
        print(f'  Avg {mean}')
        print(f'  Std {std}')
        # Write data to holders for exporting results to CSV file
        if export_csv:
            csv_row = {
                'generation': gen,
                'evaluated_individuals': len(invalid_ind),
                'min_fitness': min(fits),
                'max_fitness': max(fits),
                'avg_fitness': mean,
                'std_fitness': std,
            }
            csv_data.append(csv_row)
    print('-- End of (successful) evolution --')
    best_ind = tools.selBest(pop, 1)[0]
    print(f'Best individual: {best_ind}')
    print(f'Fitness: {best_ind.fitness.values[0]}')
    
    # 【修改：打印最佳路线时传入硬约束状态】
    print_route(ind2route(best_ind, instance, energy_constraint, battery_capacity, energy_consumption, 
                          hard_time_window=hard_time_window, hard_energy_constraint=hard_energy_constraint))
                          
    print(f'Total cost: {1 / best_ind.fitness.values[0]}')
    if export_csv:
        # 【修改：将硬约束开关的状态也记录在文件名中】
        csv_file_name = f'{instance_name}_uC{unit_cost}_iC{init_cost}_wC{wait_cost}' \
            f'_dC{delay_cost}_iS{ind_size}_pS{pop_size}_cP{cx_pb}_mP{mut_pb}_nG{n_gen}' \
            f'_eC{int(energy_constraint)}_bC{battery_capacity:.1f}_eR{energy_consumption:.2f}' \
            f'_hTW{int(hard_time_window)}_hE{int(hard_energy_constraint)}.csv'
        csv_file = os.path.join(BASE_DIR, 'results', csv_file_name)
        print(f'Write to file: {csv_file}')
        make_dirs_for_file(path=csv_file)
        if not exist(path=csv_file, overwrite=True):
            with io.open(csv_file, 'wt', encoding='utf-8', newline='') as file_object:
                fieldnames = [
                    'generation',
                    'evaluated_individuals',
                    'min_fitness',
                    'max_fitness',
                    'avg_fitness',
                    'std_fitness',
                ]
                writer = DictWriter(file_object, fieldnames=fieldnames, dialect='excel')
                writer.writeheader()
                for csv_row in csv_data:
                    writer.writerow(csv_row)