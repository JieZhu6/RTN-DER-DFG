"""
神经网络模型验证脚本（USL惩罚训练 NN + IPOPT投影）
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

# Pyomo 相关导入
from pyomo.environ import *
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition

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
print("神经网络模型验证（USL惩罚训练 NN + IPOPT投影）")
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
# 使用validation数据集进行测试
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
print("加载神经网络模型")
print("="*70)
# HIDDEN_DIMS = [64, 64, 64]       # 隐藏层维度
HIDDEN_DIMS = [32,32] 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DERDispatchNet(n_bus, n_pv, hidden_dims=HIDDEN_DIMS).to(device)

checkpoint = torch.load(f"NN_parameter/penalty_model_{HIDDEN_DIMS}.pth", map_location=device, weights_only=True)
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

V_MAX, V_MIN = 1.05, 0.95
ENABLE_CURTAILMENT = True

# 转换为 tensor
X_mean_tensor = torch.tensor(X_mean, dtype=torch.float32, device=device)
X_scale_tensor = torch.tensor(X_scale, dtype=torch.float32, device=device)
Y_mean_tensor = torch.tensor(Y_mean, dtype=torch.float32, device=device)
Y_scale_tensor = torch.tensor(Y_scale, dtype=torch.float32, device=device)

# 预计算拓扑结构信息
branch_from = {i: int(branch[i, 0]) - 1 for i in range(branch.shape[0])}
branch_to = {i: int(branch[i, 1]) - 1 for i in range(branch.shape[0])}

def get_parent_child_relations():
    parent_of = {i: [] for i in range(bus.shape[0])}
    children_of = {i: [] for i in range(bus.shape[0])}
    branch_to_parent = {}
    branch_from_parent = {}
    
    for b in range(branch.shape[0]):
        f, t = branch_from[b], branch_to[b]
        parent_of[t].append(f)
        children_of[f].append(t)
        branch_to_parent[t] = b
        branch_from_parent[f] = branch_from_parent.get(f, []) + [b]
    
    return parent_of, children_of, branch_to_parent, branch_from_parent

parent_of, children_of, branch_to_parent, branch_from_parent = get_parent_child_relations()
pv_bus_indices = {i: PV_bus[i] - 1 for i in range(len(PV_bus))}
bus_to_pv = {PV_bus[i] - 1: i for i in range(len(PV_bus))}

# ==============================================================================
# 4. IPOPT投影优化函数
# ==============================================================================
def ipopt_projection(pv_p_raw, pv_q_raw, p_load, q_load, p_available):
        
    # 求解
    start_time = time.time()
    """
    使用IPOPT将NN输出的解投影到满足Distflow约束的可行域
    目标: 最小化 ||投影解 - 原始NN解||^2
    """
    model = ConcreteModel()
    
    # 集合定义
    model.BUSES = RangeSet(0, bus.shape[0] - 1)
    model.BRANCHES = RangeSet(0, branch.shape[0] - 1)
    model.PV_UNITS = RangeSet(0, len(PV_bus) - 1)
    
    # 决策变量
    model.PV_q_power = Var(model.PV_UNITS, initialize=lambda m, i: pv_q_raw[i])
    
    if ENABLE_CURTAILMENT:
        model.PV_p_actual = Var(model.PV_UNITS, bounds=(0, None), 
                                 initialize=lambda m, i: pv_p_raw[i])
    else:
        model.PV_p_actual = Param(model.PV_UNITS, initialize=lambda m, i: p_available[i])
    
    model.MG_Power = Var(RangeSet(0, 1), initialize=0.0)
    model.branch_current = Var(model.BRANCHES, bounds=(0, branch_max**2), initialize=0.01)
    model.P_ij = Var(model.BRANCHES, initialize=0.0)
    model.Q_ij = Var(model.BRANCHES, initialize=0.0)
    model.Bus_V = Var(model.BUSES, bounds=(V_MIN**2, V_MAX**2), initialize=1.0)
    model.Bus_P_inj = Var(model.BUSES, initialize=0.0)
    model.Bus_Q_inj = Var(model.BUSES, initialize=0.0)
    
    # 约束定义
    def ref_voltage_rule(m):
        return m.Bus_V[0] == 1.0
    model.ref_voltage = Constraint(rule=ref_voltage_rule)
    
    def pv_apparent_power_rule(m, i):
        if ENABLE_CURTAILMENT:
            return m.PV_q_power[i]**2 + m.PV_p_actual[i]**2 <= PV_capacity**2
        else:
            return m.PV_q_power[i]**2 + p_available[i]**2 <= PV_capacity**2
    model.pv_apparent_power = Constraint(model.PV_UNITS, rule=pv_apparent_power_rule)
    
    if ENABLE_CURTAILMENT:
        def pv_curtailment_upper_rule(m, i):
            return m.PV_p_actual[i] <= p_available[i]
        model.pv_curtailment_upper = Constraint(model.PV_UNITS, rule=pv_curtailment_upper_rule)
    
    def bus_p_balance_rule(m, i):
        inflow = 0
        if i in branch_to_parent:
            b_in = branch_to_parent[i]
            inflow += m.P_ij[b_in]
            parent = parent_of[i][0] if parent_of[i] else 0
            r_ij = R_ij_matrix[i, parent] if parent < bus.shape[0] else 0
            inflow -= r_ij * m.branch_current[b_in]
        
        outflow = 0
        if i in branch_from_parent:
            for b_out in branch_from_parent[i]:
                outflow += m.P_ij[b_out]
        
        return m.Bus_P_inj[i] == outflow - inflow
    model.bus_p_balance = Constraint(model.BUSES, rule=bus_p_balance_rule)
    
    def bus_q_balance_rule(m, i):
        inflow = 0
        if i in branch_to_parent:
            b_in = branch_to_parent[i]
            inflow += m.Q_ij[b_in]
            parent = parent_of[i][0] if parent_of[i] else 0
            x_ij = X_ij_matrix[i, parent] if parent < bus.shape[0] else 0
            inflow -= x_ij * m.branch_current[b_in]
        
        outflow = 0
        if i in branch_from_parent:
            for b_out in branch_from_parent[i]:
                outflow += m.Q_ij[b_out]
        
        return m.Bus_Q_inj[i] == outflow - inflow
    model.bus_q_balance = Constraint(model.BUSES, rule=bus_q_balance_rule)
    
    model.voltage_drop = ConstraintList()
    for i in range(bus.shape[0]):
        if i in branch_from_parent:
            for b_out in branch_from_parent[i]:
                j = branch_to[b_out]
                r_ij = R_ij_matrix[i, j]
                x_ij = X_ij_matrix[i, j]
                
                drop_part1 = 2 * (r_ij * model.P_ij[b_out] + x_ij * model.Q_ij[b_out])
                drop_part2 = (r_ij**2 + x_ij**2) * model.branch_current[b_out]
                
                model.voltage_drop.add(model.Bus_V[j] == model.Bus_V[i] - drop_part1 + drop_part2)
    
    def distflow_exact_rule(m, b):
        i = branch_from[b]
        return m.branch_current[b] * m.Bus_V[i] == m.P_ij[b]**2 + m.Q_ij[b]**2
    model.distflow_exact = Constraint(model.BRANCHES, rule=distflow_exact_rule)
    
    def bus_p_injection_rule(m, i):
        if i == 0:
            return m.Bus_P_inj[i] == m.MG_Power[0] - p_load[i]
        elif i in bus_to_pv:
            pv_idx = bus_to_pv[i]
            if ENABLE_CURTAILMENT:
                return m.Bus_P_inj[i] == m.PV_p_actual[pv_idx] - p_load[i]
            else:
                return m.Bus_P_inj[i] == p_available[pv_idx] - p_load[i]
        else:
            return m.Bus_P_inj[i] == -p_load[i]
    model.bus_p_injection = Constraint(model.BUSES, rule=bus_p_injection_rule)
    
    def bus_q_injection_rule(m, i):
        if i == 0:
            return m.Bus_Q_inj[i] == m.MG_Power[1] - q_load[i]
        elif i in bus_to_pv:
            pv_idx = bus_to_pv[i]
            return m.Bus_Q_inj[i] == m.PV_q_power[pv_idx] - q_load[i]
        else:
            return m.Bus_Q_inj[i] == -q_load[i]
    model.bus_q_injection = Constraint(model.BUSES, rule=bus_q_injection_rule)
    
    # 投影目标函数：最小化投影解与原始NN解的距离
    def projection_objective_rule(m):
        dist_sq = 0
        for i in m.PV_UNITS:
            if ENABLE_CURTAILMENT:
                dist_sq += (m.PV_p_actual[i] - pv_p_raw[i])**2
            dist_sq += (m.PV_q_power[i] - pv_q_raw[i])**2
        return dist_sq
    
    model.objective = Objective(rule=projection_objective_rule, sense=minimize)
    
    # 配置 IPOPT 求解器
    solver = SolverFactory('ipopt', executable='D:/anaconda/envs/py3.10/Library/bin/ipopt.exe')
    # solver.options['print_level'] = 0
    # solver.options['tol'] = 1e-6
    # solver.options['max_iter'] = 300

    results = solver.solve(model, tee=False)
    solve_time = time.time() - start_time
    
    # 提取结果
    pv_p_proj = np.array([value(model.PV_p_actual[i]) for i in model.PV_UNITS]) if ENABLE_CURTAILMENT else p_available
    pv_q_proj = np.array([value(model.PV_q_power[i]) for i in model.PV_UNITS])
    
    return pv_p_proj, pv_q_proj, solve_time


def compute_objective_single(l_sq, pv_p, p_available):
    """计算单个样本的目标函数"""
    pi_p = 1.0
    curtailment = np.sum(p_available - pv_p)
    loss_curtailment = curtailment * pi_p
    loss_network = np.sum(l_sq * R_vector)
    return loss_curtailment + loss_network


# ==============================================================================
# 5. 逐个样本处理（NN + IPOPT投影）
# ==============================================================================
print("\n" + "="*70)
print("逐个样本处理（NN + IPOPT投影）")
print("="*70)

# 确保是 numpy 数组
if torch.is_tensor(X_test_raw):
    X_test_raw = X_test_raw.cpu().numpy()
if torch.is_tensor(Y_test_raw):
    Y_test_raw = Y_test_raw.cpu().numpy()

# 存储结果
Y_proj_list = []           # 投影后的解
decision_times = []        # 每个样本的总决策时间
nn_times = []              # NN前向传播时间
projection_times = []      # IPOPT投影时间
feasible_before_list = []  # NN原始输出可行性
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
    # NN前向传播
    # -----------------------------------------
    x_tensor = torch.tensor(x_norm, dtype=torch.float32, device=device)
    
    start_nn = time.time()
    with torch.no_grad():
        y_pred = model(x_tensor)
        y_pred_np = y_pred.cpu().numpy().flatten()
    nn_time = time.time() - start_nn
    
    # -----------------------------------------
    # 检查NN原始输出可行性
    # -----------------------------------------
    pv_p_nn = y_pred_np[:n_pv] * p_available  # 得到实际出力
    pv_q_nn = y_pred_np[n_pv:]
    
    V_sq_nn, l_sq_nn, _, _ = run_powerflow_numpy_single(
        p_load, q_load, 
        pv_p_nn, pv_q_nn, PV_bus
    )
    
    feasible_before = check_feasibility(V_sq_nn, l_sq_nn, 
                                         pv_p_nn, pv_q_nn, 
                                         p_available, PV_capacity_array)
    
    # -----------------------------------------
    # IPOPT投影（仅对不可行样本）
    # -----------------------------------------
    proj_time = 0.0
    if not feasible_before:
        pv_p_proj, pv_q_proj, proj_time = ipopt_projection(pv_p_nn, pv_q_nn, p_load, q_load, p_available)
    else:
        pv_p_proj = pv_p_nn
        pv_q_proj = pv_q_nn
    
    # -----------------------------------------
    # 计算潮流和目标函数
    # -----------------------------------------
    V_sq, l_sq, _, _ = run_powerflow_numpy_single(
        p_load, q_load,
        pv_p_proj, pv_q_proj, PV_bus
    )
    
    feasible_after = check_feasibility(V_sq, l_sq, 
                                        pv_p_proj, pv_q_proj,
                                        p_available, PV_capacity_array)
    
    obj_value = compute_objective_single(l_sq, pv_p_proj, p_available)
    
    # 记录结果
    total_time = nn_time + proj_time
    decision_times.append(total_time)
    nn_times.append(nn_time)
    projection_times.append(proj_time)
    feasible_before_list.append(feasible_before)
    feasible_list.append(feasible_after)
    objective_list.append(obj_value)
    Y_proj_list.append(np.concatenate([pv_p_proj, pv_q_proj]))
    
    if (i + 1) % 100 == 0 or i == n_test - 1:
        print(f"  已处理 {i+1}/{n_test} 个样本...")

# 转换为numpy数组
Y_proj_array = np.array(Y_proj_list)
feasible_before_array = np.array(feasible_before_list)
feasible_array = np.array(feasible_list)
objective_values = np.array(objective_list)

# ==============================================================================
# 6. 统计结果
# ==============================================================================
print(f"\n" + "="*70)
print("统计结果")
print("="*70)

# 统计可行点和不可行点的投影时间
n_infeasible = np.sum(~feasible_before_array)
n_feasible = np.sum(feasible_before_array)

avg_proj_time_all = np.mean(projection_times)
avg_proj_time_infeasible = np.mean(np.array(projection_times)[~feasible_before_array]) if n_infeasible > 0 else 0
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

# 保存结果
results = {
    'method': 'NN_IPOPT_Projection',
    'decision_times': decision_times,
    'nn_times': nn_times,
    'projection_times': projection_times,
    'feasible_before': feasible_before_array,
    'feasible': feasible_array,
    'objective_values': objective_values,
    'label_objective_values': label_objective_values,
    'avg_decision_time': np.mean(decision_times),
    'feasibility_rate': np.mean(feasible_array),
    'avg_objective_value': np.mean(objective_values),
    'nn_feasible_rate': np.mean(feasible_before_array),
    'avg_proj_time_all': avg_proj_time_all,
    'avg_proj_time_infeasible': avg_proj_time_infeasible,
    'n_infeasible': int(n_infeasible),
    'n_feasible': int(n_feasible),
}
np.savez("Test_result/results_O_NN.npz", **results)
print(f"\n结果已保存到: Test_result/results_O_NN.npz")




print(f"平均每个样本总时间 (NN推理 + IPOPT投影) : {np.mean(decision_times):.4f} s")
print(f"  - 平均NN推理时间: {np.mean(nn_times):.4f} s")
print(f"  - 平均IPOPT投影时间 (全体样本): {avg_proj_time_all:.4f} s")
print(f"  - 平均IPOPT投影时间 (仅不可行点): {avg_proj_time_infeasible:.4f} s")
print(f"\n样本统计:")
print(f"  NN原始可行 (无需投影): {n_feasible}/{n_test} ({n_feasible/n_test*100:.2f}%)")
print(f"  NN原始不可行 (需要投影): {n_infeasible}/{n_test} ({n_infeasible/n_test*100:.2f}%)")
print(f"\n投影后可行率: {np.mean(feasible_array) * 100:.2f}%")
print(f"\n目标函数统计:")
print(f"  均值: {np.mean(objective_values):.6f}")
print(f"  标准差: {np.std(objective_values):.6f}")
print(f"\n目标函数对比:")
print(f"  Label 目标函数均值: {np.mean(label_objective_values):.6f}")
print(f"  NN 目标函数均值:    {np.mean(objective_values):.6f}")
print(f"  平均相对差距:       {(np.mean(objective_values) - np.mean(label_objective_values)) / np.mean(label_objective_values) * 100:.3f}%")