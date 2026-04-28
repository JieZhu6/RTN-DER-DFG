"""
神经网络模型验证脚本（NN + 二分法投影）
基于 USL_NN_penalty_oproj.py，使用二分法投影代替IPOPT投影
逐个样本处理以计算准确的平均时间

使用方法:
    python USL_NN_bisection.py
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
# N_VALIDATE_SAMPLES = 20  # 设为 None 验证全部，或设为整数如 20 验证前20个
# ==============================================================================
# 1. 加载系统参数和数据集
# ==============================================================================
print("="*70)
print("神经网络模型验证（NN + 二分法投影）")
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
# 检验的时候使用validation的数据集
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
# 2. 加载神经网络模型（二分法训练）
# ==============================================================================
print("\n" + "="*70)
print("加载神经网络模型（二分法训练）")
print("="*70)
# HIDDEN_DIMS = [64, 64]       # 隐藏层维度
HIDDEN_DIMS = [32, 32] 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DERDispatchNet(n_bus, n_pv, hidden_dims=HIDDEN_DIMS).to(device)
LEARNING_RATE = 1e-3          # 学习率
BATCH_SIZE = 64   
checkpoint = torch.load(f"NN_parameter/penalty_model_{HIDDEN_DIMS}_{BATCH_SIZE}_{LEARNING_RATE}.pth", map_location=device, weights_only=True)
# checkpoint = torch.load(f"NN_parameter/supervised_trained_model_{HIDDEN_DIMS}.pth", map_location=device, weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"模型加载成功!")

# ==============================================================================
# 3. 准备系统参数和仿射系数
# ==============================================================================
if np.isscalar(PV_capacity):
    PV_capacity_array = np.full(n_pv, PV_capacity, dtype=np.float32)
else:
    PV_capacity_array = np.array(PV_capacity, dtype=np.float32)

# 加载仿射系数（用于计算内点）
affine_coef = np.load('System_data/robust_affine_coefficients.npz')
M_yPVq = affine_coef['M_yPVq']
m_yPVq = affine_coef['m_yPVq']
M_yPVp = affine_coef['M_yPVp']
m_yPVp = affine_coef['m_yPVp']

W_IP = np.vstack([M_yPVp, M_yPVq])
b_IP = np.hstack([m_yPVp, m_yPVq])

# 转换为 tensor（用于前向传播）
X_mean_tensor = torch.tensor(X_mean, dtype=torch.float32, device=device)
X_scale_tensor = torch.tensor(X_scale, dtype=torch.float32, device=device)
Y_mean_tensor = torch.tensor(Y_mean, dtype=torch.float32, device=device)
Y_scale_tensor = torch.tensor(Y_scale, dtype=torch.float32, device=device)

# ==============================================================================
# 4. 二分法投影函数（基于numpy，无梯度计算）
# ==============================================================================
def bisection_projection(f_NN, f_IP, p_load, q_load, p_available, feasible_before,
                          bisection_tol=1e-3, max_iter=50):
    """
    二分法投影（numpy实现，无梯度计算）
    
    参数:
        f_NN: np.ndarray, shape (2*n_pv,) - NN输出
        f_IP: np.ndarray, shape (2*n_pv,) - 内点
        p_load: np.ndarray, shape (n_bus,) - 有功负荷
        q_load: np.ndarray, shape (n_bus,) - 无功负荷
        p_available: np.ndarray, shape (n_pv,) - PV可用功率
        feasible_before: bool - NN输出是否可行
        bisection_tol: float - 二分法收敛容差
        max_iter: int - 最大迭代次数
    
    返回:
        f_proj: np.ndarray, shape (2*n_pv,) - 投影后的决策变量
        kappa: float - 投影系数
        n_iter: int - 实际迭代次数
    """
    if feasible_before:
        # 已经可行，直接返回
        return f_NN, 1.0, 0
    
    # 需要二分法投影
    k_l = 0.0
    k_u = 1.0
    
    n_iter = 0
    while (k_u - k_l) >= bisection_tol and n_iter < max_iter:
        n_iter += 1
        k_mid = (k_l + k_u) / 2.0
        
        # 计算测试点
        f_test = k_mid * (f_NN - f_IP) + f_IP
        pv_p_test = f_test[:n_pv]
        pv_q_test = f_test[n_pv:]
        
        # 潮流计算
        V_sq_mid, l_sq_mid, _, _ = run_powerflow_numpy_single(
            p_load, q_load,
            pv_p_test, pv_q_test, PV_bus
        )
        
        # 检查可行性
        feasible_mid = check_feasibility(V_sq_mid, l_sq_mid, pv_p_test, pv_q_test,
                                          p_available, PV_capacity_array)
        
        if feasible_mid:
            k_l = k_mid
        else:
            k_u = k_mid
    
    # 最终投影点（取k_l，因为k_l对应可行解）
    kappa = k_l
    f_proj = kappa * (f_NN - f_IP) + f_IP
    
    return f_proj, kappa, n_iter


def compute_inner_point(x_input):
    """
    计算内点
    
    参数:
        x_input: np.ndarray, shape (n_features,) - 输入特征
    
    返回:
        f_IP: np.ndarray, shape (2*n_pv,) - 内点
    """
    # 提取除去首节点的负荷
    p_load_except_root = x_input[1:n_bus]
    q_load_except_root = x_input[n_bus+1:2*n_bus]
    p_avail = x_input[2*n_bus:2*n_bus+n_pv]
    
    # 构建内点计算输入
    x_IP_input = np.concatenate([p_load_except_root, q_load_except_root, p_avail])
    
    # 计算内点
    f_IP = W_IP @ x_IP_input + b_IP
    
    return f_IP


def compute_objective_single(l_sq, pv_p, p_available):
    """
    计算单个样本的目标函数
    
    参数:
        l_sq: np.ndarray, shape (n_branch,) - 支路电流平方
        pv_p: np.ndarray, shape (n_pv,) - PV有功出力
        p_available: np.ndarray, shape (n_pv,) - PV可用功率
    
    返回:
        obj: float - 目标函数值
    """
    pi_p = 1.0
    curtailment = np.sum(p_available - pv_p)
    loss_curtailment = curtailment * pi_p
    loss_network = np.sum(l_sq * R_vector)
    
    return loss_curtailment + loss_network


# ==============================================================================
# 5. 逐个样本处理（计算每个样本的平均时间）
# ==============================================================================
print("\n" + "="*70)
print("逐个样本处理（NN + 二分法投影）")
print("="*70)

# 确保 X_test_raw 是 numpy 数组
if torch.is_tensor(X_test_raw):
    X_test_raw = X_test_raw.cpu().numpy()
if torch.is_tensor(Y_test_raw):
    Y_test_raw = Y_test_raw.cpu().numpy()

# 存储结果
Y_proj_list = []           # 投影后的解
decision_times = []        # 每个样本的总决策时间
nn_times = []              # NN前向传播时间
projection_times = []      # 二分法投影时间（所有样本，可行点为0）
projection_iters = []      # 二分法迭代次数
feasible_before_list = []  # NN输出是否可行（是否需要投影）
feasible_list = []         # 投影后可行性
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
    # 步骤1: NN前向传播
    # -----------------------------------------
    x_tensor = torch.tensor(x_norm, dtype=torch.float32, device=device)
    
    start_nn = time.time()
    with torch.no_grad():
        y_pred = model(x_tensor)
        f_NN = y_pred.cpu().numpy().flatten()
    nn_time = time.time() - start_nn
    
    # -----------------------------------------
    # 步骤2: 检查NN输出可行性
    # -----------------------------------------
    pv_p_nn = f_NN[:n_pv] * p_available
    pv_q_nn = f_NN[n_pv:]
    V_sq_nn, l_sq_nn, _, _ = run_powerflow_numpy_single(
        p_load, q_load, 
        pv_p_nn, pv_q_nn, PV_bus
    )
    feasible_before = check_feasibility(V_sq_nn, l_sq_nn, pv_p_nn, pv_q_nn, 
                                         p_available, PV_capacity_array)
    
    # -----------------------------------------
    # 步骤3: 计算内点
    # -----------------------------------------
    f_IP = compute_inner_point(x_raw)
    
    # 将 NN 输出（比例空间）转换为实际出力空间，与内点保持一致
    f_NN_actual = f_NN.copy()
    f_NN_actual[:n_pv] = f_NN_actual[:n_pv] * p_available
    
    # -----------------------------------------
    # 步骤4: 二分法投影
    # -----------------------------------------
    start_proj = time.time()
    f_proj, kappa, n_iter = bisection_projection(f_NN_actual, f_IP, p_load, q_load, p_available, feasible_before)
    proj_time = time.time() - start_proj
    
    # -----------------------------------------
    # 步骤4: 计算潮流和目标函数
    # -----------------------------------------
    pv_p_proj = f_proj[:n_pv]
    pv_q_proj = f_proj[n_pv:]
    
    V_sq, l_sq, _, _ = run_powerflow_numpy_single(
        p_load, q_load,
        pv_p_proj, pv_q_proj, PV_bus
    )
    
    # 检查投影后的可行性
    feasible_after = check_feasibility(V_sq, l_sq, pv_p_proj, pv_q_proj,
                                        p_available, PV_capacity_array)
    
    # 计算目标函数
    obj_value = compute_objective_single(l_sq, pv_p_proj, p_available)
    
    # 记录结果
    total_time = nn_time + proj_time
    decision_times.append(total_time)
    nn_times.append(nn_time)
    projection_times.append(proj_time)
    projection_iters.append(n_iter)
    feasible_before_list.append(feasible_before)  # NN原始输出是否可行
    feasible_list.append(feasible_after)          # 投影后是否可行
    objective_list.append(obj_value)
    Y_proj_list.append(f_proj)
    
    if (i + 1) % 10 == 0 or i == n_test - 1:
        print(f"  已处理 {i+1}/{n_test} 个样本...")

# 转换为numpy数组
Y_proj_array = np.array(Y_proj_list)
feasible_before_array = np.array(feasible_before_list)  # NN原始输出可行率
feasible_array = np.array(feasible_list)                # 投影后可行率
objective_values = np.array(objective_list)

# ==============================================================================
# 6. 统计结果
# ==============================================================================
print(f"\n" + "="*70)
print("统计结果")
print("="*70)

# 统计可行点和不可行点的投影时间
n_infeasible = np.sum(~feasible_before_array)  # 需要投影的样本数
n_feasible = np.sum(feasible_before_array)     # 不需要投影的样本数

avg_proj_time_all = np.mean(projection_times)  # 全体样本平均（包含0）
avg_proj_time_infeasible = np.mean(np.array(projection_times)[~feasible_before_array]) if n_infeasible > 0 else 0

print(f"平均每个样本总时间 (NN推理 + 二分法投影) : {np.mean(decision_times):.4f} s")
print(f"  - 平均NN推理时间: {np.mean(nn_times):.4f} s")
print(f"  - 平均二分法投影时间 (全体样本): {avg_proj_time_all:.4f} s")
print(f"  - 平均二分法投影时间 (仅不可行点): {avg_proj_time_infeasible:.4f} s")
print(f"\n样本统计:")
print(f"  NN原始可行 (无需投影): {n_feasible}/{n_test} ({n_feasible/n_test*100:.2f}%)")
print(f"  NN原始不可行 (需要投影): {n_infeasible}/{n_test} ({n_infeasible/n_test*100:.2f}%)")
print(f"  平均二分法迭代次数: {np.mean(projection_iters):.2f}")
print(f"\n投影后可行率: {np.mean(feasible_array) * 100:.2f}%")
print(f"\n目标函数统计:")
print(f"  均值: {np.mean(objective_values):.6f}")
print(f"  标准差: {np.std(objective_values):.6f}")

# 保存结果
results = {
    'method': 'NN_Bisection',
    'decision_times': decision_times,
    'nn_times': nn_times,
    'projection_times': projection_times,
    'projection_iters': projection_iters,
    'feasible_before': feasible_before_array,  # NN原始输出可行性
    'feasible': feasible_array,                # 投影后可行性
    'objective_values': objective_values,
    'avg_decision_time': np.mean(decision_times),
    'feasibility_rate': np.mean(feasible_array),
    'avg_objective_value': np.mean(objective_values),
    'nn_feasible_rate': np.mean(feasible_before_array),  # NN原始可行率
    'avg_proj_time_all': avg_proj_time_all,
    'avg_proj_time_infeasible': avg_proj_time_infeasible,
    'n_infeasible': n_infeasible,
    'n_feasible': n_feasible,
}
np.savez("Test_result/results_B_NN.npz", **results)
print(f"\n结果已保存到: Test_result/results_B_NN.npz")

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

# 目标函数对比
obj_gap = objective_values - label_objective_values
obj_gap_relative = obj_gap / (np.abs(label_objective_values) + 1e-8) * 100

print(f"\n目标函数对比:")
print(f"  Label 目标函数均值: {np.mean(label_objective_values):.6f}")
print(f"  NN 目标函数均值:    {np.mean(objective_values):.6f}")
print(f"  平均相对差距:       {(np.mean(objective_values) - np.mean(label_objective_values)) / np.mean(label_objective_values) * 100:.3f}%")
