import numpy as np
import json
from typing import Dict, List, Tuple
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.vrp_model import VRPTruckDroneModel, Customer, Vehicle, Depot


class InstanceGenerator:
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        
        self.truck_speed = 25.0
        self.truck_capacity = 100.0
        self.truck_fixed_cost = 500.0
        self.truck_variable_cost = 2
        
        self.drone_speed = 50.0
        self.drone_capacity = 15.0
        self.drone_fixed_cost = 100.0
        self.drone_variable_cost = 2.0
        self.drone_range_medium = 4.0  # Medium endurance: 4 km (round trip)
        self.drone_range_high = 6.0    # High endurance: 6 km (round trip)
        
        self.depot_location = (50.0, 50.0)
    
    def generate_instance(self, n_customers: int, n_vehicles: int = 2, 
                         endurance_type: str = 'medium',
                         area_size: float = 20.0, demand_range: Tuple[float, float] = (1, 10),
                         time_window_width: float = 100.0) -> VRPTruckDroneModel:
        """Generate instance with 1:1 truck-drone pairing.
        
        Args:
            n_customers: Number of customers
            n_vehicles: Number of trucks (equals number of drones, 1:1 pairing)
            endurance_type: 'medium' (4km) or 'high' (6km) drone endurance (round trip)
            area_size: Service area size (default 20km to match drone endurance)
        """
        model = VRPTruckDroneModel()
        # Depot at center of service area
        depot_x = area_size / 2
        depot_y = area_size / 2
        model.add_depot(depot_x, depot_y)
        
        # Set drone range based on endurance type
        if endurance_type == 'medium':
            model.drone_range = self.drone_range_medium
        elif endurance_type == 'high':
            model.drone_range = self.drone_range_high
        else:
            model.drone_range = self.drone_range_medium
        
        # Add trucks and drones (1:1 pairing)
        for i in range(n_vehicles):
            truck = Vehicle(
                id=i,
                type='truck',
                capacity=self.truck_capacity,
                speed=self.truck_speed,
                fixed_cost=self.truck_fixed_cost,
                variable_cost=self.truck_variable_cost
            )
            model.add_truck(truck)
            
            # Each truck carries one drone
            drone = Vehicle(
                id=n_vehicles + i,
                type='drone',
                capacity=self.drone_capacity,
                speed=self.drone_speed,
                fixed_cost=self.drone_fixed_cost,
                variable_cost=self.drone_variable_cost
            )
            model.add_drone(drone)
        
        for i in range(n_customers):
            x = np.random.uniform(0, area_size)
            y = np.random.uniform(0, area_size)
            demand = np.random.uniform(*demand_range)
            service_time = np.random.uniform(5, 15)
            tw_start = np.random.uniform(0, 200)
            tw_end = tw_start + time_window_width
            priority = np.random.randint(1, 4)
            
            customer = Customer(
                id=i,
                x=x,
                y=y,
                demand=demand,
                service_time=service_time,
                time_window=(tw_start, tw_end),
                priority=priority
            )
            model.add_customer(customer)
        
        return model
    
    def generate_small_instance(self, endurance_type: str = 'medium') -> VRPTruckDroneModel:
        """Generate small instance: 10 customers, 2 trucks + 2 drones."""
        return self.generate_instance(n_customers=10, n_vehicles=2, endurance_type=endurance_type)
    
    def generate_medium_instance(self, endurance_type: str = 'medium') -> VRPTruckDroneModel:
        """Generate medium instance: 25 customers, 2 trucks + 2 drones (paper config)."""
        return self.generate_instance(n_customers=25, n_vehicles=2, endurance_type=endurance_type)
    
    def generate_large_instance(self, n_vehicles: int = 4, endurance_type: str = 'medium') -> VRPTruckDroneModel:
        """Generate large instance: 50 customers.
        
        Paper tests: 2+2, 4+4, 6+6 configurations.
        Default: 4 trucks + 4 drones.
        """
        return self.generate_instance(n_customers=50, n_vehicles=n_vehicles, endurance_type=endurance_type)
    
    def generate_xlarge_instance(self, endurance_type: str = 'medium') -> VRPTruckDroneModel:
        """Generate xlarge instance: 100 customers, 6 trucks + 6 drones."""
        return self.generate_instance(n_customers=100, n_vehicles=6, endurance_type=endurance_type)
    
    def save_instance(self, model: VRPTruckDroneModel, filename: str):
        data = {
            'depot': {'x': model.depot.x, 'y': model.depot.y},
            'drone_range': model.drone_range,
            'customers': [],
            'trucks': [],
            'drones': []
        }
        
        for cust in model.customers:
            data['customers'].append({
                'id': cust.id,
                'x': cust.x,
                'y': cust.y,
                'demand': cust.demand,
                'service_time': cust.service_time,
                'time_window': list(cust.time_window),
                'priority': cust.priority
            })
        
        for truck in model.trucks:
            data['trucks'].append({
                'id': truck.id,
                'type': truck.type,
                'capacity': truck.capacity,
                'speed': truck.speed,
                'fixed_cost': truck.fixed_cost,
                'variable_cost': truck.variable_cost
            })
        
        for drone in model.drones:
            data['drones'].append({
                'id': drone.id,
                'type': drone.type,
                'capacity': drone.capacity,
                'speed': drone.speed,
                'fixed_cost': drone.fixed_cost,
                'variable_cost': drone.variable_cost
            })
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_instance(self, filename: str) -> VRPTruckDroneModel:
        with open(filename, 'r') as f:
            data = json.load(f)
        
        model = VRPTruckDroneModel()
        model.add_depot(data['depot']['x'], data['depot']['y'])
        model.drone_range = data.get('drone_range', 30.0)
        
        for cust_data in data['customers']:
            customer = Customer(
                id=cust_data['id'],
                x=cust_data['x'],
                y=cust_data['y'],
                demand=cust_data['demand'],
                service_time=cust_data['service_time'],
                time_window=tuple(cust_data['time_window']),
                priority=cust_data.get('priority', 1)
            )
            model.add_customer(customer)
        
        for truck_data in data['trucks']:
            truck = Vehicle(
                id=truck_data['id'],
                type=truck_data['type'],
                capacity=truck_data['capacity'],
                speed=truck_data['speed'],
                fixed_cost=truck_data['fixed_cost'],
                variable_cost=truck_data['variable_cost']
            )
            model.add_truck(truck)
        
        for drone_data in data['drones']:
            drone = Vehicle(
                id=drone_data['id'],
                type=drone_data['type'],
                capacity=drone_data['capacity'],
                speed=drone_data['speed'],
                fixed_cost=drone_data['fixed_cost'],
                variable_cost=drone_data['variable_cost']
            )
            model.add_drone(drone)
        
        return model


def generate_all_instances(output_dir: str = './data'):
    gen = InstanceGenerator(seed=42)
    
    instances = {
        'small': gen.generate_small_instance(),
        'medium': gen.generate_medium_instance(),
        'large': gen.generate_large_instance(),
        'xlarge': gen.generate_xlarge_instance()
    }
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    for name, instance in instances.items():
        filename = os.path.join(output_dir, f'{name}_instance.json')
        gen.save_instance(instance, filename)
        print(f"Generated {name} instance with {instance.get_number_of_customers()} customers")
    
    return instances


if __name__ == '__main__':
    generate_all_instances()
