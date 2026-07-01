import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class Customer:
    id: int
    x: float
    y: float
    demand: float
    service_time: float
    time_window: Tuple[float, float]
    priority: int = 1

    def distance_to(self, other) -> float:
        return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


@dataclass
class Depot:
    x: float
    y: float

    def distance_to(self, other) -> float:
        return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


@dataclass
class Vehicle:
    id: int
    type: str
    capacity: float
    speed: float
    fixed_cost: float
    variable_cost: float


@dataclass
class DroneMission:
    drone_id: int
    customer_ids: List[int]
    launch_point: int  # 存储在卡车 customers 列表中的索引
    return_point: int  # 存储在卡车 customers 列表中的索引
    launch_time: float = 0.0
    return_time: float = 0.0


@dataclass
class Route:
    vehicle_id: int
    vehicle_type: str
    customers: List[int] = field(default_factory=list)
    drone_missions: List[DroneMission] = field(default_factory=list)
    depot_id: int = 0

    def total_distance(self, model: 'VRPTruckDroneModel') -> float:
        if len(self.customers) == 0 and len(self.drone_missions) == 0:
            return 0.0

        dist = 0.0

        # 卡车主路线距离
        nodes = [model.depot]
        for cust_id in self.customers:
            nodes.append(model.customers[cust_id])
        nodes.append(model.depot)

        for i in range(len(nodes) - 1):
            dist += nodes[i].distance_to(nodes[i + 1])

        # 无人机协同路线距离
        if self.vehicle_type == 'truck':
            for mission in self.drone_missions:
                for cust_id in mission.customer_ids:
                    # 修复：正确将索引映射为实际客户坐标
                    if mission.launch_point >= 0 and mission.launch_point < len(self.customers):
                        l_cust_id = self.customers[mission.launch_point]
                        launch_pos = model.customers[l_cust_id]
                    else:
                        launch_pos = model.depot
                        
                    if mission.return_point >= 0 and mission.return_point < len(self.customers):
                        r_cust_id = self.customers[mission.return_point]
                        return_pos = model.customers[r_cust_id]
                    else:
                        return_pos = model.depot
                        
                    cust = model.customers[cust_id]
                    # 真实的三角飞行距离
                    dist += launch_pos.distance_to(cust) + cust.distance_to(return_pos)
        return dist

    def total_time(self, model: 'VRPTruckDroneModel') -> float:
        if len(self.customers) == 0 and len(self.drone_missions) == 0:
            return 0.0

        time = 0.0

        if self.vehicle_type == 'truck':
            prev = model.depot
            prev_time = 0.0
            customer_times = {}

            for idx, cust_id in enumerate(self.customers):
                cust = model.customers[cust_id]
                dist = prev.distance_to(cust)
                travel_time = dist / model.get_vehicle_speed('truck')
                arr_time = prev_time + travel_time
                wait_time = max(0, cust.time_window[0] - arr_time)
                time = arr_time + wait_time + cust.service_time
                prev_time = time
                prev = cust
                customer_times[idx] = time

            dist = prev.distance_to(model.depot)
            travel_time = dist / model.get_vehicle_speed('truck')
            time += travel_time

            for mission in self.drone_missions:
                launch_idx = mission.launch_point
                launch_time = customer_times.get(launch_idx, 0.0)
                
                # 修复索引映射
                if launch_idx >= 0 and launch_idx < len(self.customers):
                    l_pos = model.customers[self.customers[launch_idx]]
                else:
                    l_pos = model.depot
                    
                if mission.return_point >= 0 and mission.return_point < len(self.customers):
                    r_pos = model.customers[self.customers[mission.return_point]]
                else:
                    r_pos = model.depot

                drone_time = 0.0
                for cust_id in mission.customer_ids:
                    cust = model.customers[cust_id]
                    drone_dist = l_pos.distance_to(cust) + cust.distance_to(r_pos)
                    drone_time += drone_dist / model.get_vehicle_speed('drone')

                mission.launch_time = launch_time
                mission.return_time = launch_time + drone_time

                if mission.return_time > time:
                    time = mission.return_time

        else:
            prev = model.depot
            prev_time = 0.0

            for cust_id in self.customers:
                cust = model.customers[cust_id]
                dist = prev.distance_to(cust)
                travel_time = dist / model.get_vehicle_speed('drone')
                arr_time = prev_time + travel_time
                wait_time = max(0, cust.time_window[0] - arr_time)
                time = arr_time + wait_time + cust.service_time
                prev_time = time
                prev = cust

            dist = prev.distance_to(model.depot)
            travel_time = dist / model.get_vehicle_speed('drone')
            time += travel_time

        return time

    def total_cost(self, model: 'VRPTruckDroneModel') -> float:
        distance = self.total_distance(model)
        vehicle = model.get_vehicle(self.vehicle_id)
        if vehicle is None:
            return 0.0
        return vehicle.fixed_cost + vehicle.variable_cost * distance

    def is_feasible(self, model: 'VRPTruckDroneModel') -> bool:
        total_demand = sum(model.customers[c].demand for c in self.customers)
        vehicle = model.get_vehicle(self.vehicle_id)
        if vehicle is None:
            return False
        if total_demand > vehicle.capacity:
            return False

        if self.vehicle_type == 'truck':
            for mission in self.drone_missions:
                mission_demand = sum(model.customers[c].demand for c in mission.customer_ids)
                drone = model.get_drone(mission.drone_id)
                if drone and mission_demand > drone.capacity:
                    return False
        return True


class VRPTruckDroneModel:
    def __init__(self):
        self.depot: Depot = None
        self.customers: List[Customer] = []
        self.trucks: List[Vehicle] = []
        self.drones: List[Vehicle] = []
        self.routes: List[Route] = []
        self.drone_range: float = 0.0
        self.sync_time: float = 0.0

    def add_depot(self, x: float, y: float):
        self.depot = Depot(x, y)

    def add_customer(self, customer: Customer):
        self.customers.append(customer)

    def add_truck(self, vehicle: Vehicle):
        if vehicle.type == 'truck':
            self.trucks.append(vehicle)
        else:
            raise ValueError("Vehicle type must be 'truck'")

    def add_drone(self, vehicle: Vehicle):
        if vehicle.type == 'drone':
            self.drones.append(vehicle)
        else:
            raise ValueError("Vehicle type must be 'drone'")

    def get_vehicle(self, vehicle_id: int) -> Optional[Vehicle]:
        for t in self.trucks:
            if t.id == vehicle_id:
                return t
        for d in self.drones:
            if d.id == vehicle_id:
                return d
        return None

    def get_drone(self, drone_id: int) -> Optional[Vehicle]:
        for d in self.drones:
            if d.id == drone_id:
                return d
        return None

    def get_vehicle_speed(self, vehicle_type: str) -> float:
        if vehicle_type == 'truck' and self.trucks:
            return self.trucks[0].speed
        elif vehicle_type == 'drone' and self.drones:
            return self.drones[0].speed
        return 1.0

    def calculate_distance_matrix(self) -> np.ndarray:
        n = len(self.customers) + 1
        dist_matrix = np.zeros((n, n))
        nodes = [self.depot] + self.customers
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist_matrix[i, j] = nodes[i].distance_to(nodes[j])
        return dist_matrix

    def evaluate_solution(self, routes: List[Route]) -> Tuple[float, float]:
        total_cost = 0.0
        total_time = 0.0
        served_customers = set()

        for route in routes:
            total_cost += route.total_cost(self)
            total_time = max(total_time, route.total_time(self))
            served_customers.update(route.customers)
            for mission in route.drone_missions:
                served_customers.update(mission.customer_ids)

        unserved_penalty = (len(self.customers) - len(served_customers)) * 10000.0
        total_cost += unserved_penalty

        return total_cost, total_time

    def evaluate_multi_objective(self, routes: List[Route]) -> Tuple[float, float]:
        makespan = 0.0
        customer_satisfaction = 0.0
        served_customers = set()
        total_priority = sum(c.priority for c in self.customers)

        for route in routes:
            makespan = max(makespan, route.total_time(self))

        for route in routes:
            served_customers.update(route.customers)

            if route.vehicle_type == 'truck':
                for mission in route.drone_missions:
                    served_customers.update(mission.customer_ids)
                
                prev = self.depot
                prev_time = 0.0

                for cust_id in route.customers:
                    if cust_id >= len(self.customers):
                        continue
                    cust = self.customers[cust_id]
                    dist = prev.distance_to(cust)
                    travel_time = dist / self.get_vehicle_speed('truck')
                    arr_time = prev_time + travel_time

                    norm_time = arr_time / max(makespan, 0.001)
                    satisfaction = max(0.0, 1.0 - norm_time * 0.5)
                    customer_satisfaction += cust.priority * satisfaction

                    prev_time = arr_time + max(0, cust.time_window[0] - arr_time) + cust.service_time
                    prev = cust

                for mission in route.drone_missions:
                    for cust_id in mission.customer_ids:
                        if cust_id >= len(self.customers):
                            continue
                        cust = self.customers[cust_id]
                        
                        # 修复索引映射
                        if mission.launch_point >= 0 and mission.launch_point < len(route.customers):
                            l_cust_id = route.customers[mission.launch_point]
                            launch_pos = self.customers[l_cust_id]
                        else:
                            launch_pos = self.depot
                            
                        dist = launch_pos.distance_to(cust)
                        travel_time = dist / self.get_vehicle_speed('drone')
                        arr_time = mission.launch_time + travel_time

                        norm_time = arr_time / max(makespan, 0.001)
                        satisfaction = max(0.0, 1.0 - norm_time * 0.5)
                        customer_satisfaction += cust.priority * satisfaction
            else:
                prev = self.depot
                prev_time = 0.0

                for cust_id in route.customers:
                    if cust_id >= len(self.customers):
                        continue
                    cust = self.customers[cust_id]
                    dist = prev.distance_to(cust)
                    travel_time = dist / self.get_vehicle_speed('drone')
                    arr_time = prev_time + travel_time

                    norm_time = arr_time / max(makespan, 0.001)
                    satisfaction = max(0.0, 1.0 - norm_time * 0.5)
                    customer_satisfaction += cust.priority * satisfaction

                    prev_time = arr_time + max(0, cust.time_window[0] - arr_time) + cust.service_time
                    prev = cust

        if total_priority > 0:
            customer_satisfaction /= total_priority
        else:
            customer_satisfaction = 0.0

        customer_satisfaction = max(0.0, min(1.0, customer_satisfaction))

        unserved_count = len(self.customers) - len(served_customers)
        makespan = max(0.0, makespan + unserved_count * 10000.0)
        customer_satisfaction = max(0.0, customer_satisfaction - unserved_count * 0.2)

        return makespan, customer_satisfaction

    def is_solution_feasible(self, routes: List[Route]) -> bool:
        served_customers = set()
        for route in routes:
            if not route.is_feasible(self):
                return False
            served_customers.update(route.customers)
            for mission in route.drone_missions:
                served_customers.update(mission.customer_ids)
        return len(served_customers) == len(self.customers)

    def get_number_of_customers(self) -> int:
        return len(self.customers)

    def get_number_of_trucks(self) -> int:
        return len(self.trucks)

    def get_number_of_drones(self) -> int:
        return len(self.drones)