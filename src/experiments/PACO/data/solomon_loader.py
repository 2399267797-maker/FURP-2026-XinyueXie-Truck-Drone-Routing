"""Solomon VRPTW Benchmark Dataset Loader with Linear Scaling for Urban Logistics.

Solomon RC series datasets (RC1, RC2) combine random and clustered customer distributions.
Original coordinates are in [0, 100] range, linearly scaled to [0, 20] km for urban logistics.
"""
import numpy as np
from typing import List, Tuple
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.vrp_model import VRPTruckDroneModel, Customer, Vehicle, Depot


class SolomonLoader:
    """Load Solomon VRPTW benchmark datasets with scaling adaptation."""
    
    # Scaling factor: Solomon [0,100] -> Urban logistics [0,20] km
    SCALE_FACTOR = 0.12  # 12km / 100 = 0.12
    
    # Time scaling: Solomon time units -> minutes
    TIME_SCALE = 1.0  # Keep as-is (already in time units)
    
    def __init__(self):
        # Adapted for urban drone logistics benchmark
        # Target cost range: 60-160, tardiness range: 60-130
        self.truck_speed = 25.0/60  # km/min
        self.truck_capacity = 200.0  # RC1 series standard capacity, 2 trucks needed for 25 customers
        self.truck_fixed_cost = 100.0
        self.truck_variable_cost = 2  # Distance-based cost
        
        self.drone_speed = 50.0/60  # km/min
        self.drone_capacity = 40#40.0  # Can carry single customer order (demand 1-40, Solomon max)
        self.drone_fixed_cost = 0.0#4.0 truck has paid this
        self.drone_variable_cost = 1#1  # Drone per-unit distance cost is lower than truck
        self.drone_range_medium = 4.0  # km
        self.drone_range_high = 6.0    # km
    
    def load_rc1_instance(self, n_customers: int = 25, instance_id: int = 1,
                          n_vehicles: int = 2, endurance_type: str = 'medium',
                          use_drones: bool = True) -> VRPTruckDroneModel:
        """Load RC1-type Solomon instance (short scheduling horizon).
        
        RC1: Mixed random-clustered distribution, tight time windows.
        Typically 5-10 customers per vehicle route.
        
        Args:
            n_customers: Number of customers (25 or 50 from Solomon 100-customer dataset)
            instance_id: Instance number (1-8 for RC1 series: RC101-RC108)
            n_vehicles: Number of trucks (1:1 truck-drone pairing)
            endurance_type: 'medium' (4km) or 'high' (6km) drone endurance
            use_drones: If False, creates pure truck model with 0 drones
        """
        filepath = self._find_solomon_file('RC1', instance_id)
        return self._load_from_solomon_file(filepath, n_customers, n_vehicles, endurance_type, use_drones)
    
    def load_rc2_instance(self, n_customers: int = 25, instance_id: int = 1,
                          n_vehicles: int = 4, endurance_type: str = 'medium',
                          use_drones: bool = True) -> VRPTruckDroneModel:
        """Load RC2-type Solomon instance (long scheduling horizon).
        
        RC2: Mixed random-clustered distribution, wide time windows.
        Typically 30+ customers per vehicle route.
        
        Args:
            n_customers: Number of customers (25 or 50)
            instance_id: Instance number (1-8 for RC2 series: RC201-RC208)
            n_vehicles: Number of trucks (1:1 truck-drone pairing)
            endurance_type: 'medium' or 'high' drone endurance
            use_drones: If False, creates pure truck model with 0 drones
        """
        filepath = self._find_solomon_file('RC2', instance_id)
        return self._load_from_solomon_file(filepath, n_customers, n_vehicles, endurance_type, use_drones)
    
    def _find_solomon_file(self, rc_type: str, instance_id: int) -> str:
        """Find Solomon dataset file in common locations."""
        filename = f"{rc_type}{instance_id:02d}.txt"
        
        # Search for Solomon data files in common locations
        search_paths = [
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                         'PACO_vs_NSGA2', 'data', 'text', filename),
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                         'E-VRPTW', 'data', 'text', filename),
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                         'py-ga-VRPTW', 'data', 'text', filename),
        ]
        
        for path in search_paths:
            if os.path.exists(path):
                return path
        
        raise FileNotFoundError(f"Solomon file {filename} not found in search paths: {search_paths}")
    
    def _load_from_solomon_file(self, filepath: str, n_customers: int, n_vehicles: int,
                               endurance_type: str, use_drones: bool = True) -> VRPTruckDroneModel:
        """Load Solomon instance from text file with drone-truck model setup."""
        model = VRPTruckDroneModel()
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Parse customer data (starts after header)
        customer_lines = []
        for i, line in enumerate(lines):
            if i >= 9:
                parts = line.strip().split()
                if len(parts) >= 7:
                    try:
                        cust_no = int(parts[0])
                        if cust_no >= 0:
                            customer_lines.append(parts)
                    except ValueError:
                        continue
        
        # Parse depot (customer 0)
        if customer_lines:
            depot_parts = customer_lines[0]
            depot_x = float(depot_parts[1]) * self.SCALE_FACTOR
            depot_y = float(depot_parts[2]) * self.SCALE_FACTOR
            model.add_depot(depot_x, depot_y)
        
        # Set drone range
        model.drone_range = self.drone_range_medium if endurance_type == 'medium' else self.drone_range_high
        
        # Set drone operation time constants (consistent with evaluation side)
        model.launch_prep_time = 0.5
        model.retrieval_time = 0.5
        
        # Add vehicles
        for i in range(n_vehicles):
            truck = Vehicle(id=i, type='truck', capacity=self.truck_capacity,
                           speed=self.truck_speed, fixed_cost=self.truck_fixed_cost,
                           variable_cost=self.truck_variable_cost)
            model.add_truck(truck)
            
            if use_drones:
                drone = Vehicle(id=n_vehicles+i, type='drone', capacity=self.drone_capacity,
                               speed=self.drone_speed, fixed_cost=self.drone_fixed_cost,
                               variable_cost=self.drone_variable_cost)
                model.add_drone(drone)
        
        # Parse customers (first n_customers after depot)
        for idx, parts in enumerate(customer_lines[1:n_customers+1]):
            cust_no = int(parts[0])
            x = float(parts[1]) * self.SCALE_FACTOR
            y = float(parts[2]) * self.SCALE_FACTOR
            demand = float(parts[3])
            ready_time = float(parts[4])
            due_time = float(parts[5])
            #service_time = float(parts[6])
            service_time = 0.0  # Ignore Solomon service time for truck-drone context
            priority = np.random.uniform(0.4, 1.0)  # Random priority for tardiness calculation
            
            customer = Customer(id=idx, x=x, y=y, demand=demand,
                               service_time=service_time,
                               time_window=(ready_time, due_time),
                               priority=priority)
            model.add_customer(customer)
        
        return model
    
    def load_from_file(self, filepath: str, n_customers: int = 25,
                       n_vehicles: int = 2, endurance_type: str = 'medium') -> VRPTruckDroneModel:
        """Load Solomon instance from text file (backward compatibility).
        
        Solomon file format:
        Line 1: VEHICLE CAPACITY
        Lines 4+: CUSTOMER DATA (CUST_NO XCOORD YCOORD DEMAND READY_TIME DUE_TIME SERVICE_TIME)
        Customer 0 is depot.
        
        Args:
            filepath: Path to Solomon .txt file
            n_customers: Number of customers to load (25 or 50 from 100-customer dataset)
            n_vehicles: Number of trucks (1:1 pairing)
            endurance_type: Drone endurance type
        """
        return self._load_from_solomon_file(filepath, n_customers, n_vehicles, endurance_type, use_drones=True)


def generate_solomon_instances(output_dir: str = './data/solomon'):
    """Generate Solomon RC-type instances for experiments."""
    loader = SolomonLoader()
    os.makedirs(output_dir, exist_ok=True)
    
    instances = {
        'RC1_25': loader.load_rc1_instance(n_customers=25, instance_id=1, n_vehicles=2),
        'RC1_50': loader.load_rc1_instance(n_customers=50, instance_id=1, n_vehicles=4),
        'RC2_25': loader.load_rc2_instance(n_customers=25, instance_id=1, n_vehicles=2),
        'RC2_50': loader.load_rc2_instance(n_customers=50, instance_id=1, n_vehicles=4),
    }
    
    for name, instance in instances.items():
        print(f"Generated {name}: {instance.get_number_of_customers()} customers, "
              f"{instance.get_number_of_trucks()} trucks, {instance.get_number_of_drones()} drones")
    
    return instances


if __name__ == '__main__':
    generate_solomon_instances()