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
        # 提取时间窗 [tw_start, tw_end]
        tw = c.get('time_window', {'start': 0, 'end': 1000})
        time_windows.append([tw['start'], tw['end']])


max_dist = max((c[0]**2 + c[1]**2)**0.5 for c in customers)
battery_cap = max_dist * 1.5

# 时间窗归一化 (0-1范围)
tw_array = torch.tensor(time_windows, dtype=torch.float32)
tw_max = tw_array[:, 1].max()
tw_min = tw_array[:, 0].min()
tw_range = tw_max - tw_min if tw_max != tw_min else 1.0
tw_normalized = (tw_array - tw_min) / tw_range

SOL_DATA = {
    'depot_xy': torch.tensor([[[depot_x, depot_y]]], dtype=torch.float32),
    'node_xy': torch.tensor([customers], dtype=torch.float32),
    'node_demand': torch.tensor([demands], dtype=torch.float32) / data.get('vehicle_capacity', 200.0),
    'node_time_windows': tw_normalized.unsqueeze(0),  # 归一化时间窗
    'battery_capacity': torch.tensor([battery_cap], dtype=torch.float32),
    'energy_consumption': torch.tensor([1.0], dtype=torch.float32)
}
# print(f"✓ 时间窗数据已加载: {time_windows[0]} -> {SOL_DATA['node_time_windows'][0, 0].item():.3f}")
tw_norm = SOL_DATA['node_time_windows'][0, 0]
print(f"✓ 时间窗数据已加载: {time_windows[0]} -> 归一化 [{tw_norm[0].item():.3f}, {tw_norm[1].item():.3f}]")
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
    'model_load': {'path': './result/20260622_160010_train_cvrp_n100_with_instNorm', 'epoch': 5},  # 新的5维模型
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
    copy_all_src(tester.result_folder)
    tester.run()
    print("✓ 测试完成")

if __name__ == "__main__":
    main()