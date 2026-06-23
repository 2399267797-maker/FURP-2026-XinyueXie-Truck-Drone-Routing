#6/20 version 1
from dataclasses import dataclass
import torch

from CVRProblemDef import get_random_problems, augment_xy_data_by_8_fold


@dataclass
class Reset_State:
    depot_xy: torch.Tensor = None
    # shape: (batch, 1, 2)
    node_xy: torch.Tensor = None
    # shape: (batch, problem, 2)
    node_demand: torch.Tensor = None
    # shape: (batch, problem)
    node_time_windows: torch.Tensor = None
    # shape: (batch, problem, 2) - [earliest, latest] (TW constraint)
    battery_capacity: torch.Tensor = None
    # shape: (batch,) - max distance on full charge (E constraint)
    energy_consumption: torch.Tensor = None
    # shape: (batch,) - energy per unit distance (E constraint)


@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor = None
    POMO_IDX: torch.Tensor = None
    # shape: (batch, pomo)
    selected_count: int = None
    load: torch.Tensor = None
    # shape: (batch, pomo)
    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    current_time: torch.Tensor = None
    # shape: (batch, pomo)
    current_energy: torch.Tensor = None
    # shape: (batch, pomo) - remaining battery energy (E constraint)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, problem+1)
    finished: torch.Tensor = None
    # shape: (batch, pomo)


class CVRPEnv:
    def __init__(self, **env_params):

        # Const @INIT
        ####################################
        self.env_params = env_params
        self.problem_size = env_params['problem_size']
        self.pomo_size = env_params['pomo_size']

        self.FLAG__use_saved_problems = False
        self.saved_depot_xy = None
        self.saved_node_xy = None
        self.saved_node_demand = None
        self.saved_node_time_windows = None
        self.saved_battery_capacity = None
        self.saved_energy_consumption = None
        self.saved_index = None

        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        # IDX.shape: (batch, pomo)
        self.depot_node_xy = None
        # shape: (batch, problem+1, 2)
        self.depot_node_demand = None
        # shape: (batch, problem+1)
        self.depot_node_time_windows = None
        # shape: (batch, problem+1, 2)
        self.battery_capacity = None
        # shape: (batch,) - max distance on full charge
        self.energy_consumption = None
        # shape: (batch,) - energy per unit distance

        # Dynamic-1
        ####################################
        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        # shape: (batch, pomo, 0~)

        # Dynamic-2
        ####################################
        self.at_the_depot = None
        # shape: (batch, pomo)
        self.load = None
        # shape: (batch, pomo)
        self.current_time = None
        # shape: (batch, pomo)
        self.current_energy = None
        # shape: (batch, pomo) - remaining battery energy
        self.visited_ninf_flag = None
        # shape: (batch, pomo, problem+1)
        self.ninf_mask = None
        # shape: (batch, pomo, problem+1)
        self.finished = None
        # shape: (batch, pomo)

        # states to return
        ####################################
        self.reset_state = Reset_State()
        self.step_state = Step_State()

    def use_saved_problems(self, filename, device):
        self.FLAG__use_saved_problems = True

        loaded_dict = torch.load(filename, map_location=device)
        self.saved_depot_xy = loaded_dict['depot_xy']
        self.saved_node_xy = loaded_dict['node_xy']
        self.saved_node_demand = loaded_dict['node_demand']
        self.saved_node_time_windows = loaded_dict.get('node_time_windows', None)
        self.saved_battery_capacity = loaded_dict.get('battery_capacity', None)
        self.saved_energy_consumption = loaded_dict.get('energy_consumption', None)
        self.saved_index = 0

    def use_saved_problems_from_dict(self, data_dict):
        """Load problems from a dictionary instead of a file"""
        self.FLAG__use_saved_problems = True
        
        self.saved_depot_xy = data_dict['depot_xy']
        self.saved_node_xy = data_dict['node_xy']
        self.saved_node_demand = data_dict['node_demand']
        self.saved_node_time_windows = data_dict.get('node_time_windows', None)
        self.saved_battery_capacity = data_dict.get('battery_capacity', None)
        self.saved_energy_consumption = data_dict.get('energy_consumption', None)
        self.saved_index = 0

    def load_problems(self, batch_size, aug_factor=1):
        self.batch_size = batch_size

        if not self.FLAG__use_saved_problems:
            depot_xy, node_xy, node_demand, node_time_windows, battery_capacity, energy_consumption = get_random_problems(batch_size, self.problem_size)
        else:
            depot_xy = self.saved_depot_xy[self.saved_index:self.saved_index+batch_size]
            node_xy = self.saved_node_xy[self.saved_index:self.saved_index+batch_size]
            node_demand = self.saved_node_demand[self.saved_index:self.saved_index+batch_size]
            node_time_windows = self.saved_node_time_windows[self.saved_index:self.saved_index+batch_size] if self.saved_node_time_windows is not None else None
            battery_capacity = self.saved_battery_capacity[self.saved_index:self.saved_index+batch_size] if self.saved_battery_capacity is not None else None
            energy_consumption = self.saved_energy_consumption[self.saved_index:self.saved_index+batch_size] if self.saved_energy_consumption is not None else None
            self.saved_index += batch_size

        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                depot_xy = augment_xy_data_by_8_fold(depot_xy)
                node_xy = augment_xy_data_by_8_fold(node_xy)
                node_demand = node_demand.repeat(8, 1)
                if node_time_windows is not None:
                    node_time_windows = node_time_windows.repeat(8, 1, 1)
                if battery_capacity is not None:
                    battery_capacity = battery_capacity.repeat(8)
                if energy_consumption is not None:
                    energy_consumption = energy_consumption.repeat(8)
            else:
                raise NotImplementedError

        self.depot_node_xy = torch.cat((depot_xy, node_xy), dim=1)
        # shape: (batch, problem+1, 2)
        depot_demand = torch.zeros(size=(self.batch_size, 1))
        # shape: (batch, 1)
        self.depot_node_demand = torch.cat((depot_demand, node_demand), dim=1)
        # shape: (batch, problem+1)
        
        # Time windows: depot has [0, inf) time window
        if node_time_windows is not None:
            depot_time_windows = torch.zeros(size=(self.batch_size, 1, 2))
            depot_time_windows[:, :, 1] = float('inf')  # depot can be visited anytime
            self.depot_node_time_windows = torch.cat((depot_time_windows, node_time_windows), dim=1)
            # shape: (batch, problem+1, 2)
        else:
            self.depot_node_time_windows = None
        
        # Electric vehicle parameters
        self.battery_capacity = battery_capacity
        self.energy_consumption = energy_consumption


        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

        self.reset_state.depot_xy = depot_xy
        self.reset_state.node_xy = node_xy
        self.reset_state.node_demand = node_demand
        self.reset_state.node_time_windows = node_time_windows
        self.reset_state.battery_capacity = battery_capacity
        self.reset_state.energy_consumption = energy_consumption

        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX = self.POMO_IDX

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long)
        # shape: (batch, pomo, 0~)
        self.previous_node = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.long)
        # shape: (batch, pomo)

        self.at_the_depot = torch.ones(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
        # shape: (batch, pomo)
        self.load = torch.ones(size=(self.batch_size, self.pomo_size))
        # shape: (batch, pomo)
        self.current_time = torch.zeros(size=(self.batch_size, self.pomo_size))
        # shape: (batch, pomo)
        
        # Initialize battery energy (E constraint)
        if self.battery_capacity is not None:
            self.current_energy = self.battery_capacity[:, None].expand(self.batch_size, self.pomo_size)
            # shape: (batch, pomo)
        else:
            self.current_energy = None
        
        self.visited_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size+1))
        # shape: (batch, pomo, problem+1)
        self.ninf_mask = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size+1))
        # shape: (batch, pomo, problem+1)
        self.finished = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
        # shape: (batch, pomo)

        reward = None
        done = False
        return self.reset_state, reward, done

    def pre_step(self):
        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.current_time = self.current_time
        self.step_state.current_energy = self.current_energy
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished

        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected):
        # selected.shape: (batch, pomo)

        # Dynamic-1
        ####################################
        self.selected_count += 1
        self.current_node = selected
        # shape: (batch, pomo)
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
        # shape: (batch, pomo, 0~)

        # Dynamic-2
        ####################################
        self.at_the_depot = (selected == 0)

        demand_list = self.depot_node_demand[:, None, :].expand(self.batch_size, self.pomo_size, -1)
        # shape: (batch, pomo, problem+1)
        gathering_index = selected[:, :, None]
        # shape: (batch, pomo, 1)
        selected_demand = demand_list.gather(dim=2, index=gathering_index).squeeze(dim=2)
        # shape: (batch, pomo)
        self.load -= selected_demand
        self.load[self.at_the_depot] = 1 # refill loaded at the depot

        # Calculate travel time and update current time
        travel_distance = torch.zeros(self.batch_size, self.pomo_size)
        if self.selected_count > 1:
            # Get previous node positions
            prev_node_pos = self.depot_node_xy[self.BATCH_IDX, self.previous_node]
            # shape: (batch, pomo, 2)
            curr_node_pos = self.depot_node_xy[self.BATCH_IDX, selected]
            # shape: (batch, pomo, 2)
            
            # Calculate travel distance (Euclidean distance)
            travel_distance = torch.norm(curr_node_pos - prev_node_pos, dim=2)
            # shape: (batch, pomo)
            
            # Update current time
            self.current_time += travel_distance
            
            # Wait until earliest time if arrived too early (TW constraint)
            if self.depot_node_time_windows is not None:
                selected_time_windows = self.depot_node_time_windows[self.BATCH_IDX, selected]
                # shape: (batch, pomo, 2)
                earliest_time = selected_time_windows[:, :, 0]
                # shape: (batch, pomo)
                self.current_time = torch.max(self.current_time, earliest_time)
        
        # Update battery energy (E constraint)
        if self.current_energy is not None and self.energy_consumption is not None:
            # Calculate energy consumption: distance * consumption rate
            energy_used = travel_distance * self.energy_consumption[:, None].expand(self.batch_size, self.pomo_size).clone()
            # Clone current_energy to avoid in-place operation on broadcasted tensor
            self.current_energy = self.current_energy.clone()
            self.current_energy -= energy_used
            
            # Refill battery at depot
            self.current_energy[self.at_the_depot] = self.battery_capacity[self.BATCH_IDX[self.at_the_depot]]
        
        self.previous_node = selected.clone()

        self.visited_ninf_flag[self.BATCH_IDX, self.POMO_IDX, selected] = float('-inf')
        # shape: (batch, pomo, problem+1)
        self.visited_ninf_flag[:, :, 0][~self.at_the_depot] = 0  # depot is considered unvisited, unless you are AT the depot

        self.ninf_mask = self.visited_ninf_flag.clone()
        round_error_epsilon = 0.00001
        demand_too_large = self.load[:, :, None] + round_error_epsilon < demand_list
        # shape: (batch, pomo, problem+1)
        self.ninf_mask[demand_too_large] = float('-inf')
        # shape: (batch, pomo, problem+1)
        
        # Time window constraint (TW): mask nodes that cannot be reached in time
        if self.depot_node_time_windows is not None:
            # Calculate arrival time to each node
            node_pos = self.depot_node_xy[:, None, :, :].expand(self.batch_size, self.pomo_size, -1, -1)
            # shape: (batch, pomo, problem+1, 2)
            curr_pos = self.depot_node_xy[self.BATCH_IDX, selected][:, :, None, :]
            # shape: (batch, pomo, 1, 2)
            dist_to_node = torch.norm(node_pos - curr_pos, dim=3)
            # shape: (batch, pomo, problem+1)
            arrival_time = self.current_time[:, :, None] + dist_to_node
            # shape: (batch, pomo, problem+1)
            
            # Get latest time for each node
            latest_time = self.depot_node_time_windows[:, None, :, 1].expand(self.batch_size, self.pomo_size, -1)
            # shape: (batch, pomo, problem+1)
            
            # Mask nodes where arrival time exceeds latest time (with NaN check)
            too_late = (arrival_time > latest_time + round_error_epsilon) & (~arrival_time.isnan()) & (~latest_time.isnan())
            self.ninf_mask[too_late] = float('-inf')
        
        # Electric vehicle constraint (E): mask nodes that cannot be reached with remaining battery
        # NOTE: E-VRPTW aligned - requires enough energy to reach node AND return to depot
        if self.current_energy is not None and self.energy_consumption is not None:
            # Calculate energy needed to reach each node
            node_pos = self.depot_node_xy[:, None, :, :].expand(self.batch_size, self.pomo_size, -1, -1)
            curr_pos = self.depot_node_xy[self.BATCH_IDX, selected][:, :, None, :]
            dist_to_node = torch.norm(node_pos - curr_pos, dim=3)
            # shape: (batch, pomo, problem+1)
            
            # E-VRPTW alignment: Energy needed = distance_to_node + distance_to_depot
            depot_pos = self.depot_node_xy[:, None, :, :].expand(self.batch_size, self.pomo_size, -1, -1)[:, :, 0:1, :]
            # Get depot position: (batch, 1, 1, 2)
            dist_to_depot = torch.norm(depot_pos - node_pos, dim=3)
            # shape: (batch, pomo, problem+1)
            total_energy_needed = (dist_to_node + dist_to_depot) * self.energy_consumption[:, None, None].expand(self.batch_size, self.pomo_size, self.problem_size+1)
            
            # Mask nodes where energy needed exceeds remaining energy (with NaN check)
            valid_energy = (~self.current_energy[:, :, None].isnan()) & (~total_energy_needed.isnan())
            not_enough_energy = (self.current_energy[:, :, None] < total_energy_needed - round_error_epsilon) & valid_energy
            self.ninf_mask[not_enough_energy] = float('-inf')

        newly_finished = (self.visited_ninf_flag == float('-inf')).all(dim=2)
        # shape: (batch, pomo)
        self.finished = self.finished + newly_finished
        # shape: (batch, pomo)

        # do not mask depot for finished episode.
        self.ninf_mask[:, :, 0][self.finished] = 0

        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.current_time = self.current_time
        self.step_state.current_energy = self.current_energy
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished

        # returning values
        done = self.finished.all()
        if done:
            reward = -self._get_travel_distance()  # note the minus sign!
        else:
            reward = None

        return self.step_state, reward, done

    def _get_travel_distance(self):
        gathering_index = self.selected_node_list[:, :, :, None].expand(-1, -1, -1, 2)
        # shape: (batch, pomo, selected_list_length, 2)
        all_xy = self.depot_node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)
        # shape: (batch, pomo, problem+1, 2)

        ordered_seq = all_xy.gather(dim=2, index=gathering_index)
        # shape: (batch, pomo, selected_list_length, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq-rolled_seq)**2).sum(3).sqrt()
        # shape: (batch, pomo, selected_list_length)

        travel_distances = segment_lengths.sum(2)
        # shape: (batch, pomo)
        return travel_distances

