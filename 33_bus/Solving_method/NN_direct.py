"""
神经网络模型验证脚本（监督学习 NN 直接输出，无投影）
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from System_data.system_config import case, Y_bus_matrix, PV_bus_define
from NN_Model.powerflow_env import check_feasibility, run_powerflow_numpy_single
import torch
import numpy as np
import matplotlib.pyplot as plt
import time

from NN_Model.model import DERDispatchNet

plt.rcParams.update({'font.size': 14})
plt.rc('font', family='Times New Roman')

# ==============================================================================
# 配置参数
# ==============================================================================
N_VALIDATE_SAMPLES = None  # 设为 None 验证全部，或设为整数如 20 验证前20个

# ==============================================================================
# 1. 加载系统参数和数据集
# ==============================================================================
print("="*70)
print("神经网络模型验证（监督学习 NN 直接输出）")
print("="*70)

SYSTEM_DATA = case()
branch = np.array(SYSTEM_DATA['branch'])
bus = np.array(SYSTEM_DATA['bus'])
PV_bus, PV_capacity = PV_bus_define()
R_ij_matrix, X_ij_matrix, r_x_ratio, branch_max = Y_bus_matrix()
R_vector = branch[:, 2] * r_x_ratio

n_bus = bus.shape[0]
n_branches = branch.shape[0]
n_pv = len(PV_bus)

print(f"系统配置: {n_bus} 节点, {n_branches} 支路, {n_pv} 个 PV")

# 加载数据集
data = np.load("Data_generation/dataset_split.npz")
scaler = np.load("Data_generation/scaler_params.npz")

X_test_norm = data['X_val_norm']
Y_test_norm = data['Y_val_norm']
X_test_raw = data['X_val_raw']
Y_test_raw = data['Y_val_raw']

X_mean, X_scale = scaler['X_mean'], scaler['X_scale']
Y_mean, Y_scale = scaler['Y_mean'], scaler['Y_scale']

n_test_total = X_test_norm.shape[0]

if N_VALIDATE_SAMPLES is not None and N_VALIDATE_SAMPLES > 0 and N_VALIDATE_SAMPLES < n_test_total:
    n_test = N_VALIDATE_SAMPLES
    X_test_norm = X_test_norm[:n_test]
    Y_test_norm = Y_test_norm[:n_test]
    X_test_raw = X_test_raw[:n_test]
    Y_test_raw = Y_test_raw[:n_test]
    print(f"测试样本总数: {n_test_total}")
    print(f"验证样本数: {n_test} (前 {n_test} 个样本)")
else:
    n_test = n_test_total
    print(f"测试样本数: {n_test} (全部样本)")

# ==============================================================================
# 2. 加载神经网络模型
# ==============================================================================
print("\n" + "="*70)
print("加载神经网络模型（监督学习）")
print("="*70)
HIDDEN_DIMS = [16,16]       # 隐藏层维度
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DERDispatchNet(n_bus, n_pv, hidden_dims=HIDDEN_DIMS).to(device)

checkpoint = torch.load(f"NN_parameter/supervised_trained_model_{HIDDEN_DIMS}.pth", map_location=device, weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"模型加载成功!")

# ==============================================================================
# 3. 准备系统参数
# ==============================================================================
if np.isscalar(PV_capacity):
    PV_capacity_array = np.full(n_pv, PV_capacity, dtype=np.float32)
else:
    PV_capacity_array = np.array(PV_capacity, dtype=np.float32)

# 转换为 tensor
X_mean_tensor = torch.tensor(X_mean, dtype=torch.float32, device=device)
X_scale_tensor = torch.tensor(X_scale, dtype=torch.float32, device=device)
Y_mean_tensor = torch.tensor(Y_mean, dtype=torch.float32, device=device)
Y_scale_tensor = torch.tensor(Y_scale, dtype=torch.float32, device=device)

# ==============================================================================
# 4. 逐个样本处理
# ==============================================================================
print("\n" + "="*70)
print("逐个样本处理（监督学习 NN 直接输出）")
print("="*70)

# 确保是 numpy 数组
if torch.is_tensor(X_test_raw):
    X_test_raw = X_test_raw.cpu().numpy()
if torch.is_tensor(Y_test_raw):
    Y_test_raw = Y_test_raw.cpu().numpy()

def compute_objective_single(l_sq, pv_p, p_available):
    """计算单个样本的目标函数"""
    pi_p = 1.0
    curtailment = np.sum(p_available - pv_p)
    loss_curtailment = curtailment * pi_p
    loss_network = np.sum(l_sq * R_vector)
    return loss_curtailment + loss_network

# 存储结果
Y_pred_list = []           # NN预测解
decision_times = []        # 每个样本的总决策时间
nn_times = []              # NN前向传播时间
projection_times = []      # 投影时间（对于直接输出方法为0）
feasible_list = []         # 可行性
objective_list = []        # 目标函数值

print(f"开始处理 {n_test} 个样本...")

for i in range(n_test):
    x_norm = X_test_norm[i:i+1]  # (1, n_features)
    x_raw = X_test_raw[i]        # (n_features,)
    
    # 提取输入参数
    p_load = x_raw[:n_bus]
    q_load = x_raw[n_bus:2*n_bus]
    p_available = x_raw[2*n_bus:2*n_bus+n_pv]
    
    # -----------------------------------------
    # NN前向传播
    # -----------------------------------------
    x_tensor = torch.tensor(x_norm, dtype=torch.float32, device=device)
    
    start_nn = time.time()
    with torch.no_grad():
        y_pred = model(x_tensor)
        y_pred_np = y_pred.cpu().numpy().flatten()
    nn_time = time.time() - start_nn
    
    # -----------------------------------------
    # 潮流计算和可行性检查
    # -----------------------------------------
    pv_p_nn = y_pred_np[:n_pv] * p_available  # 得到实际出力
    pv_q_nn = y_pred_np[n_pv:]
    
    V_sq, l_sq, _, _ = run_powerflow_numpy_single(
        p_load, q_load, 
        pv_p_nn, pv_q_nn, PV_bus
    )
    
    feasible = check_feasibility(V_sq, l_sq, pv_p_nn, pv_q_nn, 
                                  p_available, PV_capacity_array)
    
    # 计算目标函数
    obj_value = compute_objective_single(l_sq, pv_p_nn, p_available)
    
    # 记录结果（直接输出方法没有投影时间）
    decision_times.append(nn_time)
    nn_times.append(nn_time)
    projection_times.append(0.0)  # 无投影
    feasible_list.append(feasible)
    objective_list.append(obj_value)
    Y_pred_list.append(y_pred_np)
    
    if (i + 1) % 100 == 0 or i == n_test - 1:
        print(f"  已处理 {i+1}/{n_test} 个样本...")

# 转换为numpy数组
Y_pred_array = np.array(Y_pred_list)
feasible_array = np.array(feasible_list)
objective_values = np.array(objective_list)

# ==============================================================================
# 5. 统计结果
# ==============================================================================


# 保存结果
results = {
    'method': 'USL_NN_Penalty',
    'decision_times': decision_times,
    'nn_times': nn_times,
    'projection_times': projection_times,
    'objective_values': objective_values,
    'feasible': feasible_array,
    'avg_decision_time': np.mean(decision_times),
    'feasibility_rate': np.mean(feasible_array),
    'avg_objective_value': np.mean(objective_values),
}
np.savez("Test_result/results_V_NN.npz", **results)
print(f"\n结果已保存到: Test_result/results_V_NN.npz")

# 计算 label 的最优目标函数值
print("\n计算 label 的最优目标函数值...")

p_avail_np = X_test_raw[:n_test, 2*n_bus:2*n_bus+n_pv]
pv_p_label = Y_test_raw[:n_test, :n_pv]*p_avail_np
pv_q_label = Y_test_raw[:n_test, n_pv:]

# 逐个样本计算label的潮流和目标函数
l_sq_label_list = []
label_objective_values_list = []
for j in range(n_test):
    _, l_sq_j, _, _ = run_powerflow_numpy_single(
        X_test_raw[j, :n_bus],
        X_test_raw[j, n_bus:2*n_bus],
        pv_p_label[j], pv_q_label[j], PV_bus
    )
    l_sq_label_list.append(l_sq_j)
    label_objective_values_list.append(
        compute_objective_single(l_sq_j, pv_p_label[j], p_avail_np[j])
    )
l_sq_label = np.array(l_sq_label_list)
label_objective_values = np.array(label_objective_values_list)
print(f"\n" + "="*70)
print("统计结果")
print("="*70)

print(f"平均每个样本决策时间 (NN推理) : {np.mean(decision_times):.4f} s")
print(f"测试集物理可行率: {np.mean(feasible_array) * 100:.2f}%")

print(f"\n目标函数对比:")
print(f"  Label 目标函数均值: {np.mean(label_objective_values):.6f}")
print(f"  NN 目标函数均值:    {np.mean(objective_values):.6f}")
print(f"  平均相对差距:       {(np.mean(objective_values) - np.mean(label_objective_values)) / np.mean(label_objective_values) * 100:.3f}%")