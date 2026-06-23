#6/21 version5 - Direct Solomon Instance Comparison
##########################################################################################
# E-VRPTW and CVRP-POMO Comparison Tool


DEBUG_MODE = False
USE_CUDA = False
CUDA_DEVICE_NUM = 0

SOLOMON_INSTANCE = 'R101'  # 可选: 'R101', 'C101', 'RC101', 'R102', 'C102' 等

##########################################################################################
import os
import sys
import json
import torch
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")
sys.path.insert(0, "../..")


json_path = f'../../../E-VRPTW/data/json/{SOLOMON_INSTANCE}.json'
print(f"=== Solomon Instance Comparison Mode ===")
print(f"实例: {SOLOMON_INSTANCE}")
print(f"文件: {json_path}")
print(f"约束: E (能量) + TW (时间窗口)")
print("✅ 与E-VRPTW使用完全相同的Solomon实例！")
print("=" * 50)

with open(json_path, 'r') as f:
    data = json.load(f)

# 提取数据
depot_x = data['depart']['coordinates']['x']
depot_y = data['depart']['coordinates']['y']
customers = []
demands = []
time_windows = []

for i in range(1, 101):
    key = f'customer_{i}'
    if key in data:
        c = data[key]
        customers.append([c['coordinates']['x'], c['coordinates']['y']])
        demands.append(c['demand'])
        # 提取时间窗 [ready_time, due_time] (Solomon实例的字段名)
        ready_time = c.get('ready_time', 0)
        due_time = c.get('due_time', 1000)
        time_windows.append([ready_time, due_time])

# 归一化坐标到 [0, 1] 范围（与训练数据一致）
all_coords = [[depot_x, depot_y]] + customers
x_coords = [c[0] for c in all_coords]
y_coords = [c[1] for c in all_coords]
x_min, x_max = min(x_coords), max(x_coords)
y_min, y_max = min(y_coords), max(y_coords)

# 归一化坐标
depot_xy_norm = [[(depot_x - x_min) / (x_max - x_min), (depot_y - y_min) / (y_max - y_min)]]
customers_norm = [[(c[0] - x_min) / (x_max - x_min), (c[1] - y_min) / (y_max - y_min)] for c in customers]

print(f"✓ 坐标归一化: depot [{depot_x:.1f}, {depot_y:.1f}] -> [{depot_xy_norm[0][0]:.3f}, {depot_xy_norm[0][1]:.3f}]")

# 归一化时间窗到训练数据的范围（与训练数据一致）
# 训练时: max_time = 2.0 * sqrt(2) * problem_size / 20 ≈ 14.14 (对于100客户)
# 训练时: earliest ~ U(0, max_time * 0.5), latest = earliest + U(max_time * 0.3, max_time * 0.8)
max_time_train = 2.0 * np.sqrt(2) * 100 / 20  # ≈ 14.14

# Solomon实例的时间窗原始值（如 [0, 1000]）
# 归一化到 [0, max_time_train] 范围，保持与训练数据一致的分布
tw_max_original = max(tw[1] for tw in time_windows)
tw_min_original = min(tw[0] for tw in time_windows)
tw_range = tw_max_original - tw_min_original if tw_max_original != tw_min_original else 1.0

time_windows_normalized = []
for tw in time_windows:
    # 归一化到训练数据的范围
    # earliest: 归一化到 [0, max_time_train * 0.5]
    tw_start_norm = (tw[0] - tw_min_original) / tw_range * max_time_train * 0.5
    # latest: earliest + window_length, window_length ~ [max_time * 0.3, max_time * 0.8]
    window_length = (tw[1] - tw[0]) / tw_range * max_time_train
    # 确保window_length在训练数据的范围内
    window_length = np.clip(window_length, max_time_train * 0.3, max_time_train * 0.8)
    tw_end_norm = tw_start_norm + window_length
    # 确保latest不超过max_time_train
    tw_end_norm = min(tw_end_norm, max_time_train)
    time_windows_normalized.append([tw_start_norm, tw_end_norm])

tw_array = torch.tensor(time_windows_normalized, dtype=torch.float32)
print(f"✓ 时间窗归一化: [{time_windows[0][0]}, {time_windows[0][1]}] -> [{time_windows_normalized[0][0]:.2f}, {time_windows_normalized[0][1]:.2f}]")

# 训练时的能量约束参数（基于归一化坐标）
# 训练时: battery_capacity = sqrt(2) * problem_size / 10 * (0.8~1.2) ≈ 14.14 * (0.8~1.2)
# 训练时: energy_consumption = 1.0~1.2
battery_cap_train = np.sqrt(2) * 100 / 10 * 1.0  # ≈ 14.14（取中间值）
energy_consumption_train = 1.0

SOL_DATA = {
    'depot_xy': torch.tensor([depot_xy_norm], dtype=torch.float32),
    'node_xy': torch.tensor([customers_norm], dtype=torch.float32),
    'node_demand': torch.tensor([demands], dtype=torch.float32) / 50.0,  # 与训练时的demand_scaler一致
    'node_time_windows': None,  # 禁用时间窗约束
    'battery_capacity': None,  # 禁用能量约束
    'energy_consumption': None
}

##########################################################################################
import logging
from utils.utils import create_logger, copy_all_src
from CVRPTester import CVRPTester as Tester

env_params = {'problem_size': 100, 'pomo_size': 100}

model_params = {
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',
}

tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'model_load': {'path': './result/20260622_160010_train_cvrp_n100_with_instNorm', 'epoch': 5},  # 使用训练了5个epoch的模型
    'test_episodes': 1,
    'test_batch_size': 1,
    'augmentation_enable': False,
    'aug_factor': 8,
    'aug_batch_size': 400,
    'test_data_load': {'enable': False}  # 不加载预保存文件
}

logger_params = {'log_file': {'desc': 'test_cvrp100', 'filename': 'log.txt'}}

##########################################################################################
def main():
    create_logger(**logger_params)
    
    print("开始创建 Tester...")
    # 创建tester
    tester = Tester(env_params=env_params, model_params=model_params, tester_params=tester_params)
    print("✓ Tester 创建成功")
    
    # 直接设置Solomon实例数据
    print("设置环境数据...")
    env = tester.env
    env.FLAG__use_saved_problems = True
    env.saved_depot_xy = SOL_DATA['depot_xy']
    env.saved_node_xy = SOL_DATA['node_xy']
    env.saved_node_demand = SOL_DATA['node_demand']
    env.saved_node_time_windows = SOL_DATA['node_time_windows']
    env.saved_battery_capacity = SOL_DATA['battery_capacity']
    env.saved_energy_consumption = SOL_DATA['energy_consumption']
    env.saved_index = 0
    print("✓ 环境数据设置完成")
    
    print("开始测试...")
    #copy_all_src(tester.result_folder)
    tester.run()
    print("✓ 测试完成")

if __name__ == "__main__":
    main()