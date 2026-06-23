"""
ETRD-NL Instance Converter
将E-VRPTW/CVRP实例转换为ETRD-NL格式
支持三个方法在同一实例上比较
"""
import json
import numpy as np
from typing import Dict, Optional
from pathlib import Path


class ETRDInstanceConverter:
    """E-VRPTW/CVRP实例转换为ETRD-NL格式"""

    # ETRD-NL非线性充电分段参数
    CHARGING_SEGMENTS = {
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

    # 车辆默认参数
    DEFAULT_TRUCK = {
        'speed': 25.0,           # km/h
        'energy_rate': 0.2,      # kWh/km
        'battery_capacity': 120.0,  # kWh
    }

    DEFAULT_ROBOT = {
        'speed': 15.0,           # km/h
        'energy_rate': 0.1,      # kWh/km
        'battery_capacity': 5.0,  # kWh
    }

    def __init__(self):
        pass

    def convert_evrptw_to_etrdnl(self, evrptw_json_path: str,
                                  n_charging_stations: int = 3,
                                  truck_battery: float = 120.0,
                                  robot_battery: float = 5.0) -> Dict:
        """
        将E-VRPTW JSON实例转换为ETRD-NL格式

        Args:
            evrptw_json_path: E-VRPTW实例文件路径
            n_charging_stations: 充电站数量
            truck_battery: 卡车电池容量(kWh)
            robot_battery: 机器人电池容量(kWh)

        Returns:
            ETRD-NL格式的实例字典
        """
        with open(evrptw_json_path, 'r') as f:
            data = json.load(f)

        # 提取仓库坐标 (E-VRPTW使用'depart'键)
        depot_x = data['depart']['coordinates']['x']
        depot_y = data['depart']['coordinates']['y']
        depot = [depot_x, depot_y]

        # 提取客户数据
        customers = []
        service_times = []
        time_windows = {}
        demands = []

        # 收集所有客户（按编号排序）
        customer_keys = sorted([k for k in data.keys() if k.startswith('customer_')],
                              key=lambda x: int(x.split('_')[1]))

        for key in customer_keys:
            c = data[key]
            customers.append([c['coordinates']['x'], c['coordinates']['y']])
            service_times.append(c.get('service_time', 10.0))
            time_windows[len(customers)] = [c['ready_time'], c['due_time']]
            demands.append(c['demand'])

        n_customers = len(customers)

        # 在客户区域附近生成充电站
        charging_stations = self._generate_charging_stations(
            customers, depot, n_charging_stations
        )

        # 计算服务时间数组（索引从1开始，与ETRD-NL一致）
        service_time_array = {i + 1: service_times[i] for i in range(n_customers)}

        instance = {
            'name': f'ETRD-NL-{Path(evrptw_json_path).stem}',
            'source': evrptw_json_path,
            'depot': depot,
            'customers': customers,
            'charging_stations': charging_stations,
            'service_times': service_time_array,
            'time_windows': time_windows,
            'demands': {i + 1: demands[i] for i in range(n_customers)},
            'truck': {
                'speed': self.DEFAULT_TRUCK['speed'],
                'energy_rate': self.DEFAULT_TRUCK['energy_rate'],
                'battery_capacity': truck_battery,
                'charging_segments': self.CHARGING_SEGMENTS['truck']
            },
            'robot': {
                'speed': self.DEFAULT_ROBOT['speed'],
                'energy_rate': self.DEFAULT_ROBOT['energy_rate'],
                'battery_capacity': robot_battery,
                'charging_segments': self.CHARGING_SEGMENTS['robot']
            },
            'n_customers': n_customers,
            'n_charging_stations': n_charging_stations,
            'vehicle_capacity': data.get('vehicle_capacity', 200.0)
        }

        return instance

    def _generate_charging_stations(self, customers: list, depot: list,
                                    n_stations: int) -> list:
        """在客户区域附近生成充电站"""
        # 计算客户区域边界
        customers_array = np.array(customers)
        min_x, max_x = customers_array[:, 0].min(), customers_array[:, 0].max()
        min_y, max_y = customers_array[:, 1].min(), customers_array[:, 1].max()

        # 扩展边界
        margin = 5.0
        min_x, max_x = max(0, min_x - margin), min(100, max_x + margin)
        min_y, max_y = max(0, min_y - margin), min(100, max_y + margin)

        # 在区域内随机生成充电站
        charging_stations = []
        np.random.seed(42)  # 固定种子保证可重复性

        for _ in range(n_stations):
            x = np.random.uniform(min_x, max_x)
            y = np.random.uniform(min_y, max_y)
            charging_stations.append([x, y])

        return charging_stations

    def convert_cvrp_to_etrdnl(self, cvrp_json_path: str,
                               n_charging_stations: int = 3,
                               truck_battery: float = 120.0,
                               robot_battery: float = 5.0) -> Dict:
        """
        将CVRP JSON实例转换为ETRD-NL格式

        Args:
            cvrp_json_path: CVRP实例文件路径
            n_charging_stations: 充电站数量
            truck_battery: 卡车电池容量(kWh)
            robot_battery: 机器人电池容量(kWh)

        Returns:
            ETRD-NL格式的实例字典
        """
        with open(cvrp_json_path, 'r') as f:
            data = json.load(f)

        # 提取仓库坐标 (E-VRPTW使用'depart'键)
        depot_x = data['depart']['coordinates']['x']
        depot_y = data['depart']['coordinates']['y']
        depot = [depot_x, depot_y]

        # 提取客户数据
        customers = []
        service_times = []
        demands = []

        # 收集所有客户
        customer_keys = sorted([k for k in data.keys() if k.startswith('customer_')],
                              key=lambda x: int(x.split('_')[1]))

        for key in customer_keys:
            c = data[key]
            customers.append([c['coordinates']['x'], c['coordinates']['y']])
            service_times.append(c.get('service_time', 10.0))
            demands.append(c['demand'])

        n_customers = len(customers)

        # 生成充电站
        charging_stations = self._generate_charging_stations(
            customers, depot, n_charging_stations
        )

        # 服务时间数组
        service_time_array = {i + 1: service_times[i] for i in range(n_customers)}

        instance = {
            'name': f'ETRD-NL-{Path(cvrp_json_path).stem}',
            'source': cvrp_json_path,
            'depot': depot,
            'customers': customers,
            'charging_stations': charging_stations,
            'service_times': service_time_array,
            'demands': {i + 1: demands[i] for i in range(n_customers)},
            'truck': {
                'speed': self.DEFAULT_TRUCK['speed'],
                'energy_rate': self.DEFAULT_TRUCK['energy_rate'],
                'battery_capacity': truck_battery,
                'charging_segments': self.CHARGING_SEGMENTS['truck']
            },
            'robot': {
                'speed': self.DEFAULT_ROBOT['speed'],
                'energy_rate': self.DEFAULT_ROBOT['energy_rate'],
                'battery_capacity': robot_battery,
                'charging_segments': self.CHARGING_SEGMENTS['robot']
            },
            'n_customers': n_customers,
            'n_charging_stations': n_charging_stations,
            'vehicle_capacity': data.get('vehicle_capacity', 200.0)
        }

        return instance

    def save_instance(self, instance: Dict, output_path: str):
        """保存转换后的实例"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(instance, f, indent=2, ensure_ascii=False)
        print(f"实例已保存到: {output_path}")


def load_evrptw_instance(instance_name: str,
                         base_path: str = None) -> Dict:
    """
    加载E-VRPTW实例并转换为ETRD-NL格式

    Args:
        instance_name: 实例名称，如'R101', 'C101'等
        base_path: E-VRPTW数据目录路径

    Returns:
        ETRD-NL格式的实例
    """
    if base_path is None:
        base_path = r'C:\Users\23992\FURP-2026-XinyueXie-Truck-Drone-Routing\src\experiments\E-VRPTW\data\json'

    evrptw_path = f'{base_path}/{instance_name}.json'

    converter = ETRDInstanceConverter()
    instance = converter.convert_evrptw_to_etrdnl(evrptw_path)

    return instance


if __name__ == '__main__':
    # 测试转换
    print("="*60)
    print("E-VRPTW/CVRP -> ETRD-NL 实例转换测试")
    print("="*60)

    converter = ETRDInstanceConverter()

    # 转换R101实例
    print("\n1. 转换R101实例...")
    r101_path = r'C:\Users\23992\FURP-2026-XinyueXie-Truck-Drone-Routing\src\experiments\E-VRPTW\data\json\R101.json'

    try:
        instance = converter.convert_evrptw_to_etrdnl(r101_path)
        print(f"   客户数: {instance['n_customers']}")
        print(f"   充电站数: {instance['n_charging_stations']}")
        print(f"   仓库: {instance['depot']}")
        print(f"   第一个客户: {instance['customers'][0]}")
        print(f"   卡车电池: {instance['truck']['battery_capacity']} kWh")
        print(f"   机器人电池: {instance['robot']['battery_capacity']} kWh")
        print("   ✓ 转换成功!")

        # 保存到ETRD-NL目录
        output_path = r'C:\Users\23992\FURP-2026-XinyueXie-Truck-Drone-Routing\src\experiments\ETRD-NL\data\json\R101_etrdnl.json'
        converter.save_instance(instance, output_path)

    except FileNotFoundError:
        print(f"   ✗ 文件未找到: {r101_path}")
    except Exception as e:
        print(f"   ✗ 错误: {e}")

    print("\n" + "="*60)
    print("转换测试完成!")
    print("="*60)