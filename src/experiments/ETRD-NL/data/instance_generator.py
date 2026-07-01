"""
ETRD-NL Instance Generator
生成电动卡车+地面机器人配送问题的测试算例
包含：客户节点、充电站、非线性充电参数
"""
import numpy as np
import json
from typing import Dict, List, Tuple

class ETRDInstanceGenerator:
    """ETRD-NL算例生成器"""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        
        # 论文参数（Table 1）
        self.truck_speed = 25.0  # km/h
        self.robot_speed = 15.0  # km/h
        self.truck_energy_rate = 0.2  # kWh/km
        self.robot_energy_rate = 0.1  # kWh/km
        
        # 非线性充电分段参数（论文Fig.7）
        # 格式: [(SOC区间, 充电功率kW), ...]
        self.charging_segments = {
            'truck': [
                (0.0, 0.3, 50.0),   # 0-30% SOC: 50kW
                (0.3, 0.6, 40.0),   # 30-60% SOC: 40kW
                (0.6, 0.8, 30.0),   # 60-80% SOC: 30kW
                (0.8, 1.0, 20.0),   # 80-100% SOC: 20kW
            ],
            'robot': [
                (0.0, 0.4, 10.0),   # 0-40% SOC: 10kW
                (0.4, 0.7, 8.0),    # 40-70% SOC: 8kW
                (0.7, 0.9, 5.0),    # 70-90% SOC: 5kW
                (0.9, 1.0, 3.0),    # 90-100% SOC: 3kW
            ]
        }
    
    def generate_tiny_instance(self) -> Dict:
        """生成tiny算例（7个客户）"""
        return self._generate_instance(n_customers=7, n_charging_stations=2, 
                                       truck_battery=60, robot_battery=3)
    
    def generate_small_instance(self) -> Dict:
        """生成small算例（15个客户）"""
        return self._generate_instance(n_customers=15, n_charging_stations=3,
                                       truck_battery=80, robot_battery=4)
    
    def generate_medium_instance(self) -> Dict:
        """生成medium算例（30个客户）"""
        return self._generate_instance(n_customers=30, n_charging_stations=5,
                                       truck_battery=100, robot_battery=5)
    
    def generate_large_instance(self) -> Dict:
        """生成large算例（60个客户）"""
        return self._generate_instance(n_customers=60, n_charging_stations=8,
                                       truck_battery=120, robot_battery=6)

    # ========== 新增 50 / 100 / 200 客户函数 ==========
    def generate_50_instance(self) -> Dict:
        """50客户 中等规模"""
        return self._generate_instance(n_customers=50, n_charging_stations=7,
                                       truck_battery=120, robot_battery=6)

    def generate_100_instance(self) -> Dict:
        """100客户 大规模"""
        return self._generate_instance(n_customers=100, n_charging_stations=12,
                                       truck_battery=150, robot_battery=7)

    def generate_200_instance(self) -> Dict:
        """200客户 超大规模"""
        return self._generate_instance(n_customers=200, n_charging_stations=20,
                                       truck_battery=180, robot_battery=8)
    # =================================================

    def _generate_instance(self, n_customers: int, n_charging_stations: int,
                          truck_battery: float, robot_battery: float) -> Dict:
        """生成完整算例"""
        
        # 生成节点坐标（100x100区域）
        depot = np.array([50.0, 50.0])  # 仓库在中心
        customers = np.random.rand(n_customers, 2) * 100
        charging_stations = np.random.rand(n_charging_stations, 2) * 100
        
        # 服务时间（5-15分钟）
        service_times = np.random.uniform(5, 15, n_customers)
        
        # 时间窗 - 基于距离计算合理的时间窗
        time_windows = {}
        for i in range(n_customers):
            dist_to_depot = np.linalg.norm(customers[i] - depot)
            # 卡车行驶到客户的时间（距离/速度，转换为分钟）
            truck_time = dist_to_depot / self.truck_speed * 60
            # 时间窗设置：ready_time=0, due_time=2-4倍卡车行驶时间
            ready_time = 0.0
            due_time = truck_time * np.random.uniform(2.0, 4.0)
            time_windows[i + 1] = [ready_time, due_time]
        
        instance = {
            'name': f'ETRD-NL-{n_customers}',
            'depot': depot.tolist(),
            'customers': customers.tolist(),
            'charging_stations': charging_stations.tolist(),
            'service_times': service_times.tolist(),
            'time_windows': time_windows,
            'truck': {
                'speed': self.truck_speed,
                'energy_rate': self.truck_energy_rate,
                'battery_capacity': truck_battery,  # kWh
                'charging_segments': self.charging_segments['truck']
            },
            'robot': {
                'speed': self.robot_speed,
                'energy_rate': self.robot_energy_rate,
                'battery_capacity': robot_battery,  # kWh
                'charging_segments': self.charging_segments['robot']
            },
            'n_customers': n_customers,
            'n_charging_stations': n_charging_stations
        }
        
        return instance
    
    def save_instance(self, instance: Dict, filename: str):
        """保存算例到JSON文件"""
        with open(filename, 'w') as f:
            json.dump(instance, f, indent=2)
        print(f"Instance saved to {filename}")
    
    def load_instance(self, filename: str) -> Dict:
        """从JSON文件加载算例"""
        with open(filename, 'r') as f:
            return json.load(f)


def generate_all_instances():
    """生成所有规模的算例"""
    gen = ETRDInstanceGenerator(seed=42)
    
    instances = {
        'tiny': gen.generate_tiny_instance(),
        'small': gen.generate_small_instance(),
        'medium': gen.generate_medium_instance(),
        'large': gen.generate_large_instance(),
        'c50': gen.generate_50_instance(),
        'c100': gen.generate_100_instance(),
        'c200': gen.generate_200_instance()
    }
    
    for name, instance in instances.items():
        filename = f'../data/{name}_instance.json'
        gen.save_instance(instance, filename)
    
    return instances


if __name__ == '__main__':
    generate_all_instances()
    print("All instances generated successfully!")