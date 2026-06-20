
import torch
import numpy as np


def get_random_problems(batch_size, problem_size):

    depot_xy = torch.rand(size=(batch_size, 1, 2))
    # shape: (batch, 1, 2)

    node_xy = torch.rand(size=(batch_size, problem_size, 2))
    # shape: (batch, problem, 2)

    if problem_size == 20:
        demand_scaler = 30
    elif problem_size == 50:
        demand_scaler = 40
    elif problem_size == 100:
        demand_scaler = 50
    else:
        raise NotImplementedError

    node_demand = torch.randint(1, 10, size=(batch_size, problem_size)) / float(demand_scaler)
    # shape: (batch, problem)

    # Generate time windows (TW constraint)
    max_time = 2.0 * np.sqrt(2) * problem_size / 20  # Approximate max travel time
    node_time_windows = generate_time_windows(batch_size, problem_size, max_time)
    # shape: (batch, problem, 2) - [earliest, latest]
    
    # Generate electric vehicle parameters (E constraint)
    # Battery capacity: maximum distance the vehicle can travel on a full charge
    battery_capacity = torch.ones(batch_size) * (np.sqrt(2) * problem_size / 10) * (0.8 + torch.rand(batch_size) * 0.4)
    # shape: (batch,) - battery capacity varies between 80% to 120% of base capacity
    
    # Energy consumption rate: energy used per unit distance
    energy_consumption = torch.ones(batch_size) * (1.0 + torch.rand(batch_size) * 0.2)
    # shape: (batch,) - consumption rate varies between 1.0 to 1.2

    return depot_xy, node_xy, node_demand, node_time_windows, battery_capacity, energy_consumption


def generate_time_windows(batch_size, problem_size, max_time):
    # Generate earliest time (E) and latest time (L) for each node
    # E ~ U(0, max_time * 0.5)
    # L = E + U(max_time * 0.3, max_time * 0.8)
    
    earliest = torch.rand(size=(batch_size, problem_size)) * max_time * 0.5
    # shape: (batch, problem)
    
    window_length = torch.rand(size=(batch_size, problem_size)) * max_time * 0.5 + max_time * 0.3
    # shape: (batch, problem)
    
    latest = earliest + window_length
    # shape: (batch, problem)
    
    # Ensure latest does not exceed max_time
    latest = torch.min(latest, torch.ones_like(latest) * max_time)
    
    time_windows = torch.stack((earliest, latest), dim=2)
    # shape: (batch, problem, 2)
    
    return time_windows


def augment_xy_data_by_8_fold(xy_data):
    # xy_data.shape: (batch, N, 2)

    x = xy_data[:, :, [0]]
    y = xy_data[:, :, [1]]
    # x,y shape: (batch, N, 1)

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)

    aug_xy_data = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    # shape: (8*batch, N, 2)

    return aug_xy_data