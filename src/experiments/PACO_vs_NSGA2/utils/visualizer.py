import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple
import sys
import os

# Adjust import path: go up 3 levels to reach src root
# Modify depth if your directory structure differs
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models.vrp_model import VRPTruckDroneModel, Route, Customer


class Visualizer:
    def __init__(self, model: VRPTruckDroneModel):
        self.model = model
    
    def plot_routes(self, routes: List[Route], title: str = "Vehicle Routes", 
                    save_path: str = None, show_all_nodes: bool = True):
        """
        Plot vehicle routes with truck paths and collaborative drone missions.
        Truck: solid line
        Drone launch: dashed line
        Drone return: dotted line
        """
        plt.figure(figsize=(12, 10))
        
        # Plot depot
        plt.scatter(self.model.depot.x, self.model.depot.y, c='red', marker='s', s=150, 
                    label='Depot', zorder=10, edgecolors='black', linewidths=1)
        
        served_customers = set()
        truck_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        drone_colors = ['#17becf', '#bcbd22', '#e377c2', '#7f7f7f', '#bc80bd', '#17becf']
        
        # Phase 1: Plot truck main routes
        for route in routes:
            if route.vehicle_type != 'truck':
                continue
            
            # Assign color by vehicle ID for consistency
            vid = route.vehicle_id
            color = truck_colors[vid % len(truck_colors)]
            served_customers.update(route.customers)
            
            if route.customers:
                x_coords = [self.model.depot.x]
                y_coords = [self.model.depot.y]
                
                for cust_id in route.customers:
                    cust = self.model.customers[cust_id]
                    x_coords.append(cust.x)
                    y_coords.append(cust.y)
                    plt.annotate(f'C{cust_id}', (cust.x, cust.y), 
                                 textcoords="offset points", xytext=(5, 5), 
                                 fontsize=8, ha='center')
                
                x_coords.append(self.model.depot.x)
                y_coords.append(self.model.depot.y)
                
                # Solid line for truck path
                plt.plot(x_coords, y_coords, color=color, linestyle='-', 
                         marker='o', markersize=8, linewidth=2, 
                         label=f'Truck {vid}', zorder=3)
                
                for cust_id in route.customers:
                    cust = self.model.customers[cust_id]
                    plt.scatter(cust.x, cust.y, c=color, marker='o', s=60, 
                               zorder=5, edgecolors='white', linewidths=1)
        
        # Phase 2: Plot drone collaborative missions (launch -> serve -> return)
        drone_labeled = set()
        for route in routes:
            if route.vehicle_type != 'truck' or not route.drone_missions:
                continue
            
            for mission in route.drone_missions:
                served_customers.add(mission.customer_ids[0])
                drone_color = drone_colors[mission.drone_id % len(drone_colors)]
                
                # Get launch point coordinates
                launch_idx = mission.launch_point
                if launch_idx == -1:
                    lx, ly = self.model.depot.x, self.model.depot.y
                else:
                    launch_cust = self.model.customers[route.customers[launch_idx]]
                    lx, ly = launch_cust.x, launch_cust.y
                        
                # Get return point coordinates
                return_idx = mission.return_point
                if return_idx == -1 or return_idx >= len(route.customers):
                    rx, ry = self.model.depot.x, self.model.depot.y
                else:
                    return_cust = self.model.customers[route.customers[return_idx]]
                    rx, ry = return_cust.x, return_cust.y
                
                # Plot drone-served customer
                cust = self.model.customers[mission.customer_ids[0]]
                label = f'Drone {mission.drone_id}' if mission.drone_id not in drone_labeled else None
                drone_labeled.add(mission.drone_id)
                plt.scatter(cust.x, cust.y, c=drone_color, marker='^', s=100, 
                           zorder=6, edgecolors='black', linewidths=1.5,
                           label=label)
                plt.annotate(f'D-C{mission.customer_ids[0]}', (cust.x, cust.y), 
                             textcoords="offset points", xytext=(5, -12), 
                             fontsize=8, ha='center', fontweight='bold')
                
                # Launch flight path (dashed line)
                plt.plot([lx, cust.x], [ly, cust.y], 
                        color=drone_color, linestyle='--', linewidth=2, 
                        alpha=0.8, zorder=4)
                # Return flight path (dotted line)
                plt.plot([cust.x, rx], [cust.y, ry], 
                        color=drone_color, linestyle=':', linewidth=2, 
                        alpha=0.8, zorder=4)
        
        # Mark unserved customers
        if show_all_nodes:
            for idx, cust in enumerate(self.model.customers):
                if idx not in served_customers:
                    plt.scatter(cust.x, cust.y, c='gray', marker='x', s=60, 
                               label='Unserved', zorder=4, alpha=0.5)
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('X Coordinate (km)', fontsize=12)
        plt.ylabel('Y Coordinate (km)', fontsize=12)
        
        config_text = f"Trucks: {self.model.get_number_of_trucks()} | Drones: {self.model.get_number_of_drones()}\n"
        config_text += f"Drone Endurance: {self.model.drone_range} km\n"
        config_text += f"-- Dashed: Launch | : Dotted: Return"
        
        plt.annotate(config_text, xy=(0.02, 0.02), xycoords='axes fraction',
                     fontsize=10, verticalalignment='bottom', 
                     bbox=dict(boxstyle='round', facecolor='#f8f9fa', alpha=0.9, edgecolor='gray'))
        
        # Remove duplicate legend entries
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='upper right', 
                   bbox_to_anchor=(1.25, 1), fontsize=10, borderaxespad=0.)
        
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Route plotted and saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_pareto_front(self, pareto_fronts: List[Tuple[List[Tuple[float, float]], str]],
                          title: str = "Pareto Front", save_path: str = None):
        """
        Plot multi-algorithm Pareto fronts (both minimization objectives).
        X: Travel Cost (smaller = better)
        Y: Tardiness Penalty (smaller = better)
        """
        plt.figure(figsize=(8, 6))
        
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#17becf']
        
        for idx, (front, label) in enumerate(pareto_fronts):
            if not front:
                continue
            
            costs = [obj[0] for obj in front]
            tardiness = [obj[1] for obj in front]
            
            marker = markers[idx % len(markers)]
            color = colors[idx % len(colors)]
            
            plt.scatter(costs, tardiness, marker=marker, facecolors='none', edgecolors=color, 
                        label=label, alpha=0.8, s=60, linewidths=1.5)
        
        plt.xlabel('Travel Cost', fontsize=11)
        plt.ylabel('Tardiness Penalty', fontsize=11)
        plt.title(title, fontsize=13, fontweight='bold')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()