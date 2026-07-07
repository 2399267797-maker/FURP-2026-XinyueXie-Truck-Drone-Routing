"""Enhanced Solomon VRPTW Benchmark Dataset Loader with Multi-Scale Support.

Extends the original SolomonLoader with:
- 25/50/100 customer scale support
- Full dataset family support (R, C, RC series, type 1 and 2)
- Automatic vehicle count calculation based on problem scale
- Instance validation and metadata generation
- Batch loading and automated test suites
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import sys
import os
import json
import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.vrp_model import VRPTruckDroneModel, Customer, Vehicle, Depot
from data.solomon_loader import SolomonLoader


# Dataset family metadata
SOLOMON_FAMILIES = {
    'R': {'description': 'Random distribution', 'instances': list(range(1, 13))},   # R101-R112
    'C': {'description': 'Clustered distribution', 'instances': list(range(1, 10))}, # C101-C109
    'RC': {'description': 'Mixed random-clustered distribution', 'instances': list(range(1, 9))},  # RC101-RC108
}

# Type-1: short scheduling horizon, narrow time windows
# Type-2: long scheduling horizon, wide time windows
SOLOMON_TYPES = {
    1: {'label': 'short horizon', 'customers_per_truck': 10,
        'default_priority_range': (0.8, 2.0),
        'window_width': 30, 'scheduling_cycle': 120},
    2: {'label': 'long horizon', 'customers_per_truck': 25,
        'default_priority_range': (0.4, 1.0),
        'window_width': 60, 'scheduling_cycle': 240},
}


class SolomonLoaderImp(SolomonLoader):
    """Enhanced Solomon dataset loader with multi-scale and full-series support."""
    
    def __init__(self):
        super().__init__()
        # Extended search paths covering all dataset families
        self.data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'PACO_vs_NSGA2', 'data', 'text'
        )
    
    # ──────────────────────────────────────────────
    #  Flexible instance loading
    # ──────────────────────────────────────────────
    
    def load_instance(self, family: str = 'RC', typ: int = 1, instance_id: int = 1,
                      n_customers: int = 25, n_vehicles: Optional[int] = None,
                      endurance_type: str = 'medium', use_drones: bool = True) -> VRPTruckDroneModel:
        """Load any Solomon instance by family, type, and instance ID.
        
        Args:
            family: Dataset family ('R', 'C', 'RC')
            typ: Type (1 = short horizon, 2 = long horizon)
            instance_id: Instance number (1-based, within family range)
            n_customers: Number of customers to load (25/50/100)
            n_vehicles: Number of trucks (auto-calculated if None)
            endurance_type: 'medium' (4km) or 'high' (6km)
            use_drones: If False, creates pure truck model
        """
        filepath = self._find_file(family, typ, instance_id)
        if n_vehicles is None:
            n_vehicles = self._auto_n_vehicles(typ, n_customers)
        return self._load_from_solomon_file(filepath, n_customers, n_vehicles,
                                            endurance_type, use_drones)
    
    def load_rc1_instance(self, n_customers: int = 25, instance_id: int = 1,
                          n_vehicles: int = 2, endurance_type: str = 'medium',
                          use_drones: bool = True) -> VRPTruckDroneModel:
        """Load RC1 (short horizon) instance (backward-compatible wrapper)."""
        return self.load_instance('RC', 1, instance_id, n_customers,
                                  n_vehicles, endurance_type, use_drones)
    
    def load_rc2_instance(self, n_customers: int = 25, instance_id: int = 1,
                          n_vehicles: int = 4, endurance_type: str = 'medium',
                          use_drones: bool = True) -> VRPTruckDroneModel:
        """Load RC2 (long horizon) instance (backward-compatible wrapper)."""
        return self.load_instance('RC', 2, instance_id, n_customers,
                                  n_vehicles, endurance_type, use_drones)
    
    def load_r1_instance(self, n_customers: int = 25, instance_id: int = 1,
                         n_vehicles: Optional[int] = None,
                         endurance_type: str = 'medium',
                         use_drones: bool = True) -> VRPTruckDroneModel:
        """Load R1 (random, short horizon) instance."""
        return self.load_instance('R', 1, instance_id, n_customers,
                                  n_vehicles, endurance_type, use_drones)
    
    def load_r2_instance(self, n_customers: int = 25, instance_id: int = 1,
                         n_vehicles: Optional[int] = None,
                         endurance_type: str = 'medium',
                         use_drones: bool = True) -> VRPTruckDroneModel:
        """Load R2 (random, long horizon) instance."""
        return self.load_instance('R', 2, instance_id, n_customers,
                                  n_vehicles, endurance_type, use_drones)
    
    def load_c1_instance(self, n_customers: int = 25, instance_id: int = 1,
                         n_vehicles: Optional[int] = None,
                         endurance_type: str = 'medium',
                         use_drones: bool = True) -> VRPTruckDroneModel:
        """Load C1 (clustered, short horizon) instance."""
        return self.load_instance('C', 1, instance_id, n_customers,
                                  n_vehicles, endurance_type, use_drones)
    
    def load_c2_instance(self, n_customers: int = 25, instance_id: int = 1,
                         n_vehicles: Optional[int] = None,
                         endurance_type: str = 'medium',
                         use_drones: bool = True) -> VRPTruckDroneModel:
        """Load C2 (clustered, long horizon) instance."""
        return self.load_instance('C', 2, instance_id, n_customers,
                                  n_vehicles, endurance_type, use_drones)
    
    # ──────────────────────────────────────────────
    #  Batch loading
    # ──────────────────────────────────────────────
    
    def batch_load(self, family: str = 'RC', typ: int = 1,
                   instance_ids: Optional[List[int]] = None,
                   n_customers: int = 25, n_vehicles: Optional[int] = None,
                   endurance_type: str = 'medium',
                   use_drones: bool = True) -> Dict[int, VRPTruckDroneModel]:
        """Load multiple instances of the same family/type at once.
        
        Returns dict mapping instance_id -> model.
        """
        if instance_ids is None:
            instance_ids = SOLOMON_FAMILIES[family]['instances']
        models = {}
        for iid in instance_ids:
            models[iid] = self.load_instance(family, typ, iid, n_customers,
                                             n_vehicles, endurance_type, use_drones)
        return models
    
    def batch_load_all_families(self, n_customers: int = 25,
                                endurance_type: str = 'medium',
                                use_drones: bool = True) -> Dict[str, VRPTruckDroneModel]:
        """Load all available Solomon instances across all families and types.
        
        Returns dict mapping 'R1_101_25' -> model (family + type + id + scale).
        """
        models = {}
        for family in ['R', 'C', 'RC']:
            for typ in [1, 2]:
                instances = SOLOMON_FAMILIES[family]['instances']
                n_vehicles = self._auto_n_vehicles(typ, n_customers)
                loaded = self.batch_load(family, typ, instances, n_customers,
                                         n_vehicles, endurance_type, use_drones)
                for iid, model in loaded.items():
                    key = f"{family}{typ}_{iid:02d}_{n_customers}"
                    models[key] = model
        return models
    
    # ──────────────────────────────────────────────
    #  Instance validation
    # ──────────────────────────────────────────────
    
    def validate_instance(self, model: VRPTruckDroneModel) -> Dict:
        """Validate a loaded instance and return diagnostics."""
        diagnostics = {
            'n_customers': len(model.customers),
            'n_trucks': len(model.trucks),
            'n_drones': len(model.drones),
            'depot': (model.depot.x, model.depot.y),
            'drone_range': model.drone_range,
            'valid': True,
            'issues': [],
            'statistics': {},
        }
        
        if model.depot is None:
            diagnostics['valid'] = False
            diagnostics['issues'].append('Depot not set')
            return diagnostics
        
        # Check depot position
        depot_x, depot_y = model.depot.x, model.depot.y
        if not (0 <= depot_x <= 12 and 0 <= depot_y <= 12):
            diagnostics['issues'].append(f'Depot ({depot_x:.2f}, {depot_y:.2f}) outside [0,12] km')
        
        # Check each customer
        for i, c in enumerate(model.customers):
            if not (0 <= c.x <= 12 and 0 <= c.y <= 12):
                diagnostics['issues'].append(f'Customer {c.id} ({c.x:.2f}, {c.y:.2f}) outside [0,12] km')
            if c.demand < 0:
                diagnostics['issues'].append(f'Customer {c.id} has negative demand {c.demand}')
            if c.demand > self.drone_capacity:
                diagnostics['issues'].append(
                    f'Customer {c.id} demand {c.demand} > drone capacity {self.drone_capacity}')
            tw_start, tw_end = c.time_window
            if tw_start > tw_end:
                diagnostics['issues'].append(f'Customer {c.id} has invalid time window ({tw_start}, {tw_end})')
        
        # Statistics
        if model.customers:
            demands = [c.demand for c in model.customers]
            tw_starts = [c.time_window[0] for c in model.customers]
            tw_ends = [c.time_window[1] for c in model.customers]
            
            distances_from_depot = [
                np.sqrt((c.x - depot_x)**2 + (c.y - depot_y)**2)
                for c in model.customers
            ]
            
            diagnostics['statistics'] = {
                'demand': {'min': float(min(demands)), 'max': float(max(demands)),
                           'mean': float(np.mean(demands)), 'total': float(sum(demands))},
                'time_window_start': {'min': float(min(tw_starts)), 'max': float(max(tw_starts))},
                'time_window_end': {'min': float(min(tw_ends)), 'max': float(max(tw_ends))},
                'distance_from_depot_km': {
                    'min': float(min(distances_from_depot)),
                    'max': float(max(distances_from_depot)),
                    'mean': float(np.mean(distances_from_depot)),
                },
                'max_drone_round_trip_km': model.drone_range * 2,
            }
            
            # Drone coverage estimate: customers within drone_range of depot
            within_range = sum(1 for d in distances_from_depot if d <= model.drone_range)
            diagnostics['statistics']['drone_depot_coverage'] = {
                'within_range': within_range,
                'total': len(model.customers),
                'ratio': within_range / len(model.customers),
            }
        
        if diagnostics['issues']:
            diagnostics['valid'] = False
        
        return diagnostics
    
    # ──────────────────────────────────────────────
    #  Metadata generation
    # ──────────────────────────────────────────────
    
    def generate_metadata(self, family: str = 'RC', typ: int = 1,
                           n_customers: int = 25,
                           endurance_type: str = 'medium') -> Dict:
        """Generate comprehensive metadata for a set of Solomon instances."""
        instances = SOLOMON_FAMILIES[family]['instances']
        n_vehicles = self._auto_n_vehicles(typ, n_customers)
        
        metadata = {
            'generated_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'dataset': f'{family}{typ}',
            'family': family,
            'type': typ,
            'type_description': SOLOMON_TYPES[typ]['label'],
            'n_customers': n_customers,
            'n_vehicles': n_vehicles,
            'n_instances': len(instances),
            'instance_ids': instances,
            'endurance_type': endurance_type,
            'drone_range_km': self.drone_range_medium if endurance_type == 'medium' else self.drone_range_high,
            'scale': {
                'map_size_km': 12,
                'depot_location': (6.0, 6.0),
                'scale_factor': self.SCALE_FACTOR,
            },
            'vehicle_config': {
                'truck_speed_km_per_min': self.truck_speed,
                'truck_capacity': self.truck_capacity,
                'drone_speed_km_per_min': self.drone_speed,
                'drone_capacity': self.drone_capacity,
                'drone_range_km': self.drone_range_medium if endurance_type == 'medium' else self.drone_range_high,
            },
            'instances': {},
        }
        
        for iid in instances:
            try:
                model = self.load_instance(family, typ, iid, n_customers,
                                           n_vehicles, endurance_type)
                diag = self.validate_instance(model)
                metadata['instances'][f'{family}{typ}{iid:02d}'] = {
                    'file': f'{family}{typ}{iid:02d}.txt',
                    'diagnostics': diag,
                }
            except Exception as e:
                metadata['instances'][f'{family}{typ}{iid:02d}'] = {
                    'file': f'{family}{typ}{iid:02d}.txt',
                    'error': str(e),
                }
        
        return metadata
    
    def generate_all_metadata(self, n_customers_list: Optional[List[int]] = None,
                               endurance_types: Optional[List[str]] = None) -> Dict:
        """Generate metadata for all family/type/scale combinations."""
        if n_customers_list is None:
            n_customers_list = [25, 50, 100]
        if endurance_types is None:
            endurance_types = ['medium', 'high']
        
        all_meta = {}
        for family in ['R', 'C', 'RC']:
            for typ in [1, 2]:
                for nc in n_customers_list:
                    for et in endurance_types:
                        key = f'{family}{typ}_{nc}_{et}'
                        try:
                            all_meta[key] = self.generate_metadata(family, typ, nc, et)
                        except Exception as e:
                            all_meta[key] = {'error': str(e)}
        return all_meta
    
    def save_metadata_json(self, output_path: str, metadata: Dict):
        """Save metadata dictionary to JSON file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        print(f"Metadata saved to {output_path}")
    
    # ──────────────────────────────────────────────
    #  Automated test suite
    # ──────────────────────────────────────────────
    
    def test_suite(self, n_customers_list: Optional[List[int]] = None,
                   families: Optional[List[str]] = None) -> Dict:
        """Run an automated test suite across multiple configurations.
        
        Returns dict of test results.
        """
        if n_customers_list is None:
            n_customers_list = [25, 50, 100]
        if families is None:
            families = ['R', 'RC']
        
        results = {}
        for family in families:
            for typ in [1, 2]:
                for nc in n_customers_list:
                    key = f'{family}{typ}_n{nc}'
                    try:
                        n_veh = self._auto_n_vehicles(typ, nc)
                        model = self.load_instance(family, typ, 1, nc, n_veh)
                        diag = self.validate_instance(model)
                        results[key] = {
                            'status': 'PASS' if diag['valid'] else 'WARN',
                            'n_vehicles_auto': n_veh,
                            'diagnostics': diag,
                        }
                    except Exception as e:
                        results[key] = {'status': 'FAIL', 'error': str(e)}
        return results
    
    # ──────────────────────────────────────────────
    #  Internal helpers
    # ──────────────────────────────────────────────
    
    def _find_file(self, family: str, typ: int, instance_id: int) -> str:
        """Find Solomon data file for any family/type/instance."""
        filename = f"{family}{typ}{instance_id:02d}.txt"
        
        search_paths = [
            os.path.join(self.data_dir, filename),
            os.path.join(os.path.dirname(self.data_dir), 'text', filename),
        ]
        
        # Add original search paths for backward compatibility
        base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        search_paths.extend([
            os.path.join(base, 'PACO_vs_NSGA2', 'data', 'text', filename),
            os.path.join(base, 'E-VRPTW', 'data', 'text', filename),
            os.path.join(base, 'py-ga-VRPTW', 'data', 'text', filename),
        ])
        
        for path in search_paths:
            if os.path.exists(path):
                return path
        
        raise FileNotFoundError(
            f"Solomon file {filename} not found. "
            f"Searched: {search_paths}")
    
    @staticmethod
    def _auto_n_vehicles(typ: int, n_customers: int) -> int:
        """Automatically determine vehicle count based on problem scale.
        
        Type 1 (short horizon): ~10 customers per truck
        Type 2 (long horizon): ~25 customers per truck
        
        Minimum 2 vehicles, rounded up.
        """
        if typ not in SOLOMON_TYPES:
            typ = 1
        cpt = SOLOMON_TYPES[typ]['customers_per_truck']
        n = int(np.ceil(n_customers / cpt))
        return max(2, n)
    
    def _load_from_solomon_file(self, filepath: str, n_customers: int, n_vehicles: int,
                               endurance_type: str, use_drones: bool = True) -> VRPTruckDroneModel:
        """Enhanced file loader with RC-type adaptive priority weights."""
        model = VRPTruckDroneModel()
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Parse customer data (starts after header, line 9)
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
        
        if not customer_lines:
            raise ValueError(f"No customer data found in {filepath}")
        
        # Parse depot (customer 0)
        depot_parts = customer_lines[0]
        depot_x = float(depot_parts[1]) * self.SCALE_FACTOR
        depot_y = float(depot_parts[2]) * self.SCALE_FACTOR
        model.add_depot(depot_x, depot_y)
        
        # Set drone range
        model.drone_range = (
            self.drone_range_medium if endurance_type == 'medium'
            else self.drone_range_high
        )
        
        # Set drone operation time constants
        model.launch_prep_time = 0.5
        model.retrieval_time = 0.5
        
        # Detect RC type from filename for adaptive priority
        filename = os.path.basename(filepath)
        is_type2 = '2' in filename and not filename.startswith('C1') and not filename.startswith('R1') and not filename.startswith('RC1')
        # More robust detection: check if type indicator char is '2'
        typ_char = filename[2] if len(filename) > 2 else '1'
        is_type2 = typ_char == '2'
        
        if is_type2:
            priority_range = SOLOMON_TYPES[2]['default_priority_range']
        else:
            priority_range = SOLOMON_TYPES[1]['default_priority_range']
        
        # Add vehicles
        for i in range(n_vehicles):
            truck = Vehicle(id=i, type='truck', capacity=self.truck_capacity,
                           speed=self.truck_speed, fixed_cost=self.truck_fixed_cost,
                           variable_cost=self.truck_variable_cost)
            model.add_truck(truck)
            
            if use_drones:
                drone = Vehicle(id=n_vehicles + i, type='drone',
                               capacity=self.drone_capacity,
                               speed=self.drone_speed,
                               fixed_cost=self.drone_fixed_cost,
                               variable_cost=self.drone_variable_cost)
                model.add_drone(drone)
        
        # Limit customers to requested count
        available_customers = customer_lines[1:]
        if n_customers > len(available_customers):
            print(f"Warning: Requested {n_customers} customers but only "
                  f"{len(available_customers)} available in {filename}. "
                  f"Using all available.")
            n_customers = len(available_customers)
        
        # Parse customers
        for idx, parts in enumerate(available_customers[:n_customers]):
            cust_no = int(parts[0])
            x = float(parts[1]) * self.SCALE_FACTOR
            y = float(parts[2]) * self.SCALE_FACTOR
            demand = float(parts[3])
            ready_time = float(parts[4])
            due_time = float(parts[5])
            service_time = 0.0  # Ignore Solomon service time for truck-drone context
            priority = np.random.uniform(priority_range[0], priority_range[1])
            
            customer = Customer(id=idx, x=x, y=y, demand=demand,
                               service_time=service_time,
                               time_window=(ready_time, due_time),
                               priority=priority)
            model.add_customer(customer)
        
        return model


# Backward-compatible alias
create_solomon_loader = SolomonLoaderImp


# ──────────────────────────────────────────────
#  CLI entry points
# ──────────────────────────────────────────────

def run_test_suite():
    """Run automated test suite and print results."""
    loader = SolomonLoaderImp()
    results = loader.test_suite()
    
    print("=" * 60)
    print("SolomonLoaderImp Test Suite Results")
    print("=" * 60)
    
    passed = sum(1 for r in results.values() if r['status'] == 'PASS')
    warned = sum(1 for r in results.values() if r['status'] == 'WARN')
    failed = sum(1 for r in results.values() if r['status'] == 'FAIL')
    
    for key, result in results.items():
        status_icon = {'PASS': 'PASS', 'WARN': 'WARN', 'FAIL': 'FAIL'}[result['status']]
        if result['status'] == 'FAIL':
            print(f"  [{status_icon}] {key}: {result['error']}")
        else:
            diag = result['diagnostics']
            issues = diag.get('issues', [])
            stats = diag.get('statistics', {})
            print(f"  [{status_icon}] {key}: {diag['n_customers']} customers, "
                  f"{diag['n_trucks']} trucks, {diag['n_drones']} drones")
            if issues:
                for iss in issues[:3]:
                    print(f"         Issue: {iss}")
            if stats:
                dd = stats.get('distance_from_depot_km', {})
                if dd:
                    print(f"         Dist from depot: {dd['mean']:.2f} km avg "
                          f"({dd['min']:.2f}-{dd['max']:.2f})")
    
    print(f"\nSummary: {passed} passed, {warned} warnings, {failed} failed")


def generate_instance_report(output_dir: str = None):
    """Generate comprehensive metadata report for all instance configurations."""
    loader = SolomonLoaderImp()
    
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                  'data', 'reports')
    
    # Generate metadata for key configurations
    meta = loader.generate_all_metadata(n_customers_list=[25, 50, 100],
                                         endurance_types=['medium', 'high'])
    
    report_path = os.path.join(output_dir, 'solomon_instances_metadata.json')
    loader.save_metadata_json(report_path, meta)
    
    # Also generate a condensed text summary
    summary_path = os.path.join(output_dir, 'solomon_instances_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("Solomon Instance Metadata Summary\n")
        f.write("=" * 60 + "\n\n")
        for key, data in sorted(meta.items()):
            if 'error' in data:
                f.write(f"{key}: ERROR - {data['error']}\n")
                continue
            f.write(f"Dataset: {data['dataset']} ({data['type_description']})\n")
            f.write(f"  Customers: {data['n_customers']}, "
                    f"Vehicles: {data['n_vehicles']}\n")
            f.write(f"  Instances: {data['n_instances']} "
                    f"({data['instance_ids']})\n")
            f.write(f"  Drone range: {data['drone_range_km']} km\n")
            for inst_name, inst_data in data['instances'].items():
                if 'error' in inst_data:
                    f.write(f"    {inst_name}: ERROR\n")
                    continue
                diag = inst_data['diagnostics']
                f.write(f"    {inst_name}: {diag['n_customers']} customers, "
                        f"{'valid' if diag['valid'] else 'ISSUES'}\n")
            f.write("\n")
    
    print(f"Summary saved to {summary_path}")
    return meta


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='SolomonLoaderImp - Enhanced Solomon Dataset Loader')
    parser.add_argument('--test', action='store_true', help='Run test suite')
    parser.add_argument('--report', action='store_true', help='Generate instance report')
    parser.add_argument('--metadata', type=str, help='Generate metadata JSON for specific config, '
                                                      'e.g. RC1_25_medium')
    args = parser.parse_args()
    
    if args.test:
        run_test_suite()
    elif args.report:
        generate_instance_report()
    elif args.metadata:
        loader = SolomonLoaderImp()
        parts = args.metadata.split('_')
        if len(parts) >= 3:
            family_typ = parts[0]
            family = family_typ[:-1]
            typ = int(family_typ[-1])
            nc = int(parts[1])
            et = parts[2]
            meta = loader.generate_metadata(family, typ, nc, et)
            print(json.dumps(meta, indent=2, default=str))
        else:
            print("Invalid metadata format. Use: FAMILYTYPE_CUSTOMERS_ENDURANCE, e.g. RC1_25_medium")
    else:
        # Default: load and display a single instance
        loader = SolomonLoaderImp()
        model = loader.load_instance('RC', 1, 1, 25)
        diag = loader.validate_instance(model)
        print(f"Loaded: {diag['n_customers']} customers, "
              f"{diag['n_trucks']} trucks, {diag['n_drones']} drones")
        print(f"Depot: {diag['depot']}")
        print(f"Valid: {diag['valid']}")
        if diag['statistics']:
            dd = diag['statistics'].get('distance_from_depot_km', {})
            if dd:
                print(f"Avg dist from depot: {dd['mean']:.2f} km")