import numpy as np
import pulp
from typing import Dict

class ETRD_MILP_Solver_PULP:
    """MILP solver for ETRD-NL problem, implemented with PuLP + CBC solver (Fixed & Optimized)"""

    def __init__(self, instance: Dict):
        self.instance = instance
        self.model = None
        self.solver = None
        self.solution = None

        # Extract spatial data from instance
        self.depot = np.array(instance['depot'])
        self.customers = np.array(instance['customers'])
        self.charging_stations = np.array(instance['charging_stations'])
        self.service_times = instance.get('service_times', {})

        self.n_customers = instance['n_customers']
        self.n_cs = instance['n_charging_stations']

        # Node set definitions
        self.N = list(range(1, self.n_customers + 1))          # Customer nodes
        self.CS = list(range(self.n_customers + 1, self.n_customers + self.n_cs + 1))  # Charging stations
        self.end_depot = self.n_customers + self.n_cs + 1      # Virtual end depot
        self.V = [0] + self.N + self.CS + [self.end_depot]     # Full node set for truck
        self.robot_nodes = self.N + self.CS                    # Nodes accessible only to robot
        self.n_nodes = len(self.V)

        # Coordinates for all nodes (start and end depots share location)
        self.coords = np.vstack([
            self.depot.reshape(1, 2),
            self.customers,
            self.charging_stations,
            self.depot.reshape(1, 2)
        ])

        # Precompute pairwise Euclidean distance matrix
        self.distances = self._compute_distance_matrix()

        # Vehicle parameters (with safe defaults)
        self.truck_params = instance.get('truck', {'speed': 1.0, 'battery_capacity': 100, 'energy_rate': 1.0, 'charge_rate': 1.0})
        self.robot_params = instance.get('robot', {'speed': 0.5, 'max_range': 50})

        # Piecewise charging segments (Non-linear simulation)
        self.n_segments = 4
        # Assuming charging slows down as battery fills. 
        # These multipliers represent the efficiency of each segment.
        self.segment_efficiency = [1.5, 1.2, 0.9, 0.5] 

        # Global Big-M constant
        self.big_M = 10000

        # Decision variables
        self.x = None          # Truck routing arcs
        self.y = None          # Robot routing arcs
        
        self.t_truck = None      # Truck arrival time
        self.t_truck_dep = None  # Truck departure time (CRITICAL FIX: separates arrival/departure state)
        
        self.t_robot = None         # Robot arrival time at customers
        self.t_robot_return = None  # Robot return time to CS (CRITICAL FIX: prevents time looping)
        self.d_robot = None         # Cumulative robot distance traveled
        
        self.e = None          # Truck battery SOC at arrival
        self.e_dep = None      # Truck battery SOC at departure
        self.c = None          # Charged energy per segment
        self.z = None          # Charging binary indicator
        self.makespan = None   # Overall completion time

    def _compute_distance_matrix(self) -> np.ndarray:
        n = len(self.V)
        dist = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist[i, j] = np.linalg.norm(self.coords[i] - self.coords[j])
        return dist

    def build_model(self):
        self.model = pulp.LpProblem("ETRD-NL", pulp.LpMinimize)
        self._create_variables()
        self._create_objective()
        self._create_constraints()

    def _create_variables(self):
        # Routing variables
        self.x = pulp.LpVariable.dicts('x', [(i, j) for i in self.V for j in self.V if i != j], lowBound=0, upBound=1, cat='Binary')
        self.y = pulp.LpVariable.dicts('y', [(i, j) for i in self.robot_nodes for j in self.robot_nodes if i != j], lowBound=0, upBound=1, cat='Binary')

        # Time tracking
        self.t_truck = pulp.LpVariable.dicts('t_truck', self.V, lowBound=0, cat='Continuous')
        self.t_truck_dep = pulp.LpVariable.dicts('t_truck_dep', self.V, lowBound=0, cat='Continuous')
        
        self.t_robot = pulp.LpVariable.dicts('t_robot', self.N, lowBound=0, cat='Continuous')
        self.t_robot_return = pulp.LpVariable.dicts('t_robot_return', self.CS, lowBound=0, cat='Continuous')
        
        # Energy and distance tracking
        cap = self.truck_params['battery_capacity']
        self.e = pulp.LpVariable.dicts('e', self.V, lowBound=0, upBound=cap, cat='Continuous')
        self.e_dep = pulp.LpVariable.dicts('e_dep', self.V, lowBound=0, upBound=cap, cat='Continuous')
        self.c = pulp.LpVariable.dicts('c', [(i, s) for i in self.CS for s in range(self.n_segments)], lowBound=0, cat='Continuous')
        self.z = pulp.LpVariable.dicts('z', self.CS, lowBound=0, upBound=1, cat='Binary')
        
        self.d_robot = pulp.LpVariable.dicts('d_robot', self.N, lowBound=0, cat='Continuous')

        # MTZ variables
        self.u = pulp.LpVariable.dicts('u', self.V, lowBound=1, upBound=self.n_nodes, cat='Continuous')
        self.u_robot = pulp.LpVariable.dicts('u_robot', self.N, lowBound=1, upBound=self.n_customers, cat='Continuous')

        self.makespan = pulp.LpVariable('makespan', lowBound=0, cat='Continuous')

    def _create_objective(self):
        self.model += self.makespan, "Minimize_Makespan"

    def _create_constraints(self):
        self._create_truck_route_constraints()
        self._create_robot_delivery_constraints()
        self._create_time_constraints()
        self._create_energy_constraints()
        self._create_makespan_constraints()

    def _create_truck_route_constraints(self):
        m = self.model
        end = self.end_depot

        # Start and End logic
        m += pulp.lpSum([self.x[0, j] for j in self.V if j != 0]) == 1
        m += pulp.lpSum([self.x[j, 0] for j in self.V if j != 0]) == 0
        m += pulp.lpSum([self.x[i, end] for i in self.V if i != end]) == 1
        m += pulp.lpSum([self.x[end, j] for j in self.V if j != end]) == 0

        # Flow conservation
        for i in self.N + self.CS:
            inflow = pulp.lpSum([self.x[j, i] for j in self.V if j != i])
            outflow = pulp.lpSum([self.x[i, j] for j in self.V if j != i])
            m += inflow == outflow
            m += inflow <= 1 # Max 1 visit per node

        # MTZ for truck
        for i in self.N + self.CS:
            for j in self.N + self.CS:
                if i != j:
                    m += self.u[i] - self.u[j] + self.n_nodes * self.x[i, j] <= self.n_nodes - 1

    def _create_robot_delivery_constraints(self):
        m = self.model
        robot_max_range = self.robot_params.get('max_range', float('inf'))

        # 1. Customer Coverage (Truck OR Robot)
        for i in self.N:
            truck_serve = pulp.lpSum([self.x[j, i] for j in self.V if j != i])
            robot_serve = pulp.lpSum([self.y[j, i] for j in self.robot_nodes if j != i])
            m += truck_serve + robot_serve == 1

            # 2. Robot flow conservation AT CUSTOMERS
            robot_out = pulp.lpSum([self.y[i, k] for k in self.robot_nodes if k != i])
            m += robot_serve == robot_out

        # 3. Truck-Robot Coupling at Charging Stations
        for cs in self.CS:
            truck_visit = pulp.lpSum([self.x[j, cs] for j in self.V if j != cs])
            robots_launched = pulp.lpSum([self.y[cs, j] for j in self.N])
            robots_returned = pulp.lpSum([self.y[i, cs] for i in self.N])
            
            # Can only launch or retrieve if truck visits the station
            m += robots_launched <= self.n_customers * truck_visit
            m += robots_returned <= self.n_customers * truck_visit

        # 4. Robot MTZ & Dynamic Range Constraints
        for i in self.N:
            for j in self.N:
                if i != j:
                    m += self.u_robot[i] - self.u_robot[j] + self.n_customers * self.y[i, j] <= self.n_customers - 1
                    
                    if robot_max_range != float('inf'):
                        m += self.d_robot[j] >= self.d_robot[i] + self.distances[i, j] - self.big_M * (1 - self.y[i, j])

        # Range tracking from CS and to CS
        if robot_max_range != float('inf'):
            for i in self.CS:
                for j in self.N:
                    m += self.d_robot[j] >= self.distances[i, j] - self.big_M * (1 - self.y[i, j])
            
            for i in self.N:
                for j in self.CS:
                    m += self.d_robot[i] + self.distances[i, j] <= robot_max_range + self.big_M * (1 - self.y[i, j])

    def _create_time_constraints(self):
        m = self.model
        M = self.big_M
        
        # --- Truck Time ---
        m += self.t_truck[0] == 0
        m += self.t_truck_dep[0] == 0

        # Travel propagation
        for i in self.V:
            for j in self.V:
                if i != j:
                    travel = self.distances[i, j] / self.truck_params['speed']
                    m += self.t_truck[j] >= self.t_truck_dep[i] + travel - M * (1 - self.x[i, j])

        # Service / Wait time mapping
        for i in self.N:
            service = self.service_times.get(i, 0)
            m += self.t_truck_dep[i] >= self.t_truck[i] + service

        for cs in self.CS:
            base_rate = self.truck_params.get('charge_rate', 1.0)
            # Calculate non-linear charge time based on segments
            charge_time = pulp.lpSum([self.c[cs, s] / (base_rate * self.segment_efficiency[s]) for s in range(self.n_segments)])
            
            # Truck leaves after arriving + charging
            m += self.t_truck_dep[cs] >= self.t_truck[cs] + charge_time
            # CRITICAL: Truck MUST wait for all dispatched robots to return to this CS
            m += self.t_truck_dep[cs] >= self.t_robot_return[cs]

        # --- Robot Time ---
        for j in self.N:
            for i in self.CS:
                travel = self.distances[i, j] / self.robot_params['speed']
                # Robot launches AFTER truck arrives at CS
                m += self.t_robot[j] >= self.t_truck[i] + travel - M * (1 - self.y[i, j])

            for i in self.N:
                if i != j:
                    travel = self.distances[i, j] / self.robot_params['speed']
                    service = self.service_times.get(i, 0)
                    m += self.t_robot[j] >= self.t_robot[i] + service + travel - M * (1 - self.y[i, j])

        # Robot return to CS
        for i in self.N:
            for cs in self.CS:
                travel = self.distances[i, cs] / self.robot_params['speed']
                service = self.service_times.get(i, 0)
                m += self.t_robot_return[cs] >= self.t_robot[i] + service + travel - M * (1 - self.y[i, cs])

    def _create_energy_constraints(self):
        m = self.model
        cap = self.truck_params['battery_capacity']
        rate = self.truck_params['energy_rate']
        M = cap

        m += self.e[0] == cap
        m += self.e_dep[0] == cap

        # Energy propagation over distance
        for i in self.V:
            for j in self.V:
                if i != j:
                    consume = self.distances[i, j] * rate
                    m += self.e[j] <= self.e_dep[i] - consume + M * (1 - self.x[i, j])

        # Customer nodes: no charging
        for i in self.N:
            m += self.e_dep[i] <= self.e[i]

        # Charging Station logic (Piecewise)
        seg_cap = cap / self.n_segments
        for cs in self.CS:
            total_charge = pulp.lpSum([self.c[cs, s] for s in range(self.n_segments)])
            
            # State transfer: departure = arrival + total charge
            m += self.e_dep[cs] <= self.e[cs] + total_charge
            m += self.e_dep[cs] <= cap
            
            # Charging Indicator
            m += total_charge <= cap * self.z[cs]
            
            # Segment capacities
            for s in range(self.n_segments):
                m += self.c[cs, s] <= seg_cap

        # 10% safety buffer at all arrival points
        for i in self.V:
            m += self.e[i] >= 0.1 * cap

    def _create_makespan_constraints(self):
        m = self.model
        m += self.makespan >= self.t_truck[self.end_depot]

    def solve(self, time_limit=60):
        self.build_model()
        print(f"开始求解，时间限制: {time_limit}秒")
        print(f"变量数: {len(self.model.variables())}")
        print(f"约束数: {len(self.model.constraints)}")

        self.solver = pulp.PULP_CBC_CMD(timeLimit=time_limit, msg=True)
        
        import time
        start = time.time()
        status = self.model.solve(self.solver)
        solve_time = time.time() - start

        if status == pulp.LpStatusOptimal:
            print("✓ 找到最优解")
            return self._extract_solution(solve_time, 'optimal')
        elif status != pulp.LpStatusInfeasible:
            print("✓ 找到可行解")
            return self._extract_solution(solve_time, 'feasible')
        else:
            print("✗ 未找到可行解")
            return None

    def _extract_solution(self, solve_time, status):
        result = {
            'makespan': pulp.value(self.makespan),
            'truck_route': [],
            'robot_route': [],
            'truck_energy': {},
            'solve_time': solve_time,
            'status': status
        }

        # Reconstruct truck route
        current = 0
        end = self.end_depot
        while current != end:
            found = False
            for j in self.V:
                if current != j and pulp.value(self.x[current, j]) and pulp.value(self.x[current, j]) > 0.5:
                    result['truck_route'].append((current, j))
                    current = j
                    found = True
                    break
            if not found:
                break

        # Reconstruct robot arcs
        for i in self.robot_nodes:
            for j in self.robot_nodes:
                if i != j and pulp.value(self.y[i, j]) and pulp.value(self.y[i, j]) > 0.5:
                    result['robot_route'].append((i, j))

        for i in self.V:
            if pulp.value(self.e[i]) is not None:
                result['truck_energy'][i] = pulp.value(self.e[i])

        return result