# 计算内点的仿射系数

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import warnings
# 屏蔽 Gurobi 链式矩阵乘法警告
warnings.filterwarnings("ignore", message="Chained matrix multiplications of MVars is inefficient")
from System_data.system_config import case, Y_bus_matrix, PV_bus_define
from gurobipy import Model, GRB, MVar
import numpy as np
import time
import cvxpy as cp
# ==============================================================================
# 1. LOAD DATA & SYSTEM CONFIGURATION (Adaptive Dimensions)
# ==============================================================================
np.random.seed(2)

# 获取系统基础拓扑和配置
SYSTEM_DATA = case()
branch = np.array(SYSTEM_DATA['branch'])
bus = np.array(SYSTEM_DATA['bus'])
PV_bus, PV_capacity = PV_bus_define()
R_ij_matrix, X_ij_matrix, r_x_ratio, branch_max = Y_bus_matrix()


# --- 自适应推导网络维度 ---
n_bus = bus.shape[0]           # 节点总数
n_branches = branch.shape[0]   # 支路总数
n_pv = len(PV_bus)             # PV 数量
print(f"System loaded adaptively: {n_bus} Buses, {n_branches} Branches, {n_pv} PVs")

# f_nodes[k] 表示第k条支路的起始节点 (0-based，0表示根节点)
f_nodes = branch[:, 0].astype(int) - 1  # 转换为0-based索引

# branch_from_matrix[k, j] = 1 如果支路k的起始节点是j+1 (即非根节点j)
branch_from_matrix = np.zeros((n_branches, n_branches))
for k in range(n_branches):
    f_node = f_nodes[k]
    if f_node > 0:
        branch_from_matrix[k, f_node - 1] = 1.0

# 标记哪些支路起始于根节点
is_from_root = (f_nodes == 0).astype(float)

V_MAX, V_MIN = 1.05, 0.95
S_BASE_MVA = 10

# 读取测试集数据
TEST_DATA = np.load("Data_generation/dataset.npy")
N_TEST_SAMPLES = TEST_DATA.shape[0]

ACTIVE_LOAD_ALL = TEST_DATA[:, :n_bus].T
REACTIVE_LOAD_ALL = TEST_DATA[:, n_bus:2*n_bus].T
PV_P_POWER_ALL = TEST_DATA[:, 2*n_bus:]

# 统一处理支路电流上限
branch_max_array = np.ones(n_branches) * branch_max if isinstance(branch_max, (int, float)) else branch_max

# ==============================================================================
# 2. CONSTRUCT NETWORK MATRICES
# ==============================================================================
B = np.zeros((n_bus, n_branches))
branch_map = {}
for k in range(n_branches):
    f_node = int(branch[k, 0]) - 1
    t_node = int(branch[k, 1]) - 1
    B[f_node, k] = 1
    B[t_node, k] = 1
    branch_map[(f_node, t_node)] = k

A = np.zeros((n_branches, n_branches))
for k in range(n_branches):
    t_node_k = int(branch[k, 1]) - 1
    for m in range(n_branches):
        if int(branch[m, 0]) - 1 == t_node_k:
            A[k, m] = 1

I_n = np.eye(n_branches)
C = np.linalg.inv(I_n - A)

r_vec = branch[:, 2] * r_x_ratio
x_vec = branch[:, 3] * r_x_ratio
z2_vec = r_vec**2 + x_vec**2
R, X, Z2 = np.diag(r_vec), np.diag(x_vec), np.diag(z2_vec)

Mp = 2 * (C.T @ R @ C)
Mq = 2 * (C.T @ X @ C)
D_R = C @ A @ R
D_X = C @ A @ X
H = C.T @ (2 * R @ D_R + 2 * X @ D_X + Z2)

D_X_plus, D_X_minus = np.maximum(D_X, 0), np.minimum(D_X, 0)
H_plus, H_minus = np.maximum(H, 0), np.minimum(H, 0)

# Baseline Taylor Expansion Point
baseline = np.load('System_data/baseline_operating_point.npz')
V0_mag = baseline['Bus_V']
P0_flow = baseline['P_ij']
Q0_flow = baseline['Q_ij']
l0 = baseline['branch_current']

J_P = 2 * P0_flow / (V0_mag[1:]**2)
J_Q = 2 * Q0_flow / (V0_mag[1:]**2)
J_V = -1 * (P0_flow**2 + Q0_flow**2) / (V0_mag[1:]**2)

J_P_plus, J_P_minus = np.maximum(J_P, 0), np.minimum(J_P, 0)
J_Q_plus, J_Q_minus = np.maximum(J_Q, 0), np.minimum(J_Q, 0)
J_V_plus, J_V_minus = np.maximum(J_V, 0), np.minimum(J_V, 0)

# PV Mapping Matrix
M_PV = np.zeros((n_branches, n_pv))
for k, bus_idx in enumerate(PV_bus):
    map_idx = bus_idx - 1 - 1
    if map_idx >= 0:
        M_PV[map_idx, k] = 1.0

C_MPV = C @ M_PV
Mp_MPV = Mp @ M_PV
Mq_MPV = Mq @ M_PV

# ==============================================================================
# 3. DEFINE UNCERTAINTY SET (Data-Driven Box Constraints)
# ==============================================================================
P_min = np.min(ACTIVE_LOAD_ALL[1:, :], axis=1)
P_max = np.max(ACTIVE_LOAD_ALL[1:, :], axis=1)

Q_min = np.min(REACTIVE_LOAD_ALL[1:, :], axis=1)
Q_max = np.max(REACTIVE_LOAD_ALL[1:, :], axis=1)

PV_min = np.min(PV_P_POWER_ALL, axis=0)
PV_max = np.max(PV_P_POWER_ALL, axis=0)

# 组合边界向量
x_min_data = np.concatenate([P_min, Q_min, PV_min])
x_max_data = np.concatenate([P_max, Q_max, PV_max])

# 稍微放大以确保测试集完全被包裹
margin = 0.03
x_lb = x_min_data - margin * np.abs(x_min_data)
x_ub = x_max_data + margin * np.abs(x_max_data)

N_X = len(x_ub)
N_Y = 2 * n_pv + 2 * n_branches
V0_vec = np.ones(n_branches) * (1.0**2)


# ==============================================================================
# 4. ROBUST MASTER PROBLEM (CVXPY 建模 + Gurobi 求解)
# ==============================================================================

def add_robust_linear_constr_cvxpy(constraints, M_expr, m_expr, x_lb, x_ub):
    """
    将鲁棒线性约束: M_expr * x + m_expr <= 0, ∀x ∈ [x_lb, x_ub]
    利用强对偶转化为静态确定性线性约束。
    M_expr: 形状为 (N_constr, N_X) 的矩阵或 cvxpy Expression
    m_expr: 形状为 (N_constr,) 的向量或 cvxpy Expression
    """
    N_constr, N_x = M_expr.shape
    
    # 引入对偶变量 \lambda^+ 和 \lambda^-
    lam_plus = cp.Variable((N_constr, N_x), nonneg=True)
    lam_minus = cp.Variable((N_constr, N_x), nonneg=True)

    # 对偶等式约束: \lambda^+ - \lambda^- = M_expr
    constraints.append(lam_plus - lam_minus == M_expr)

    # 对偶目标函数约束: \lambda^+ * x_ub - \lambda^- * x_lb + m_expr <= 0
    # 注意: lam_plus @ x_ub 执行的是矩阵和向量的乘法，返回维度为 (N_constr,)
    dual_obj = lam_plus @ x_ub - lam_minus @ x_lb
    constraints.append(dual_obj + m_expr <= 0)

# ===================== 定义 CVXPY 变量 =====================
# 仿射系数矩阵/向量
W_mat = cp.Variable((N_Y, N_X), name="W_IP")
w_vec = cp.Variable(N_Y, name="w_IP")
# 松弛变量
s_slack = cp.Variable(nonneg=True, name="slack_s")

# 辅助变量 (SOC 1-范数近似)
tau_PVp = cp.Variable(n_pv, name="tau_PVp")
tau_PVq = cp.Variable(n_pv, name="tau_PVq")
tau_P = cp.Variable(n_branches, name="tau_P")
tau_Q = cp.Variable(n_branches, name="tau_Q")
tau_R = cp.Variable(n_branches, name="tau_R")

# 初始化约束列表
constraints = []

# ------------------------------------------------------------------------------
# 4.1 提取不确定性 x 到各状态变量的映射矩阵 (与原代码完全一致)
# ------------------------------------------------------------------------------
M_xPL = np.hstack([np.eye(n_branches), np.zeros((n_branches, n_branches + n_pv))])
M_xQL = np.hstack([np.zeros((n_branches, n_branches)), np.eye(n_branches), np.zeros((n_branches, n_pv))])
M_xPVmax = np.hstack([np.zeros((n_pv, 2*n_branches)), np.eye(n_pv)])

# 决策变量映射
M_yPVq = W_mat[0:n_pv, :]
m_yPVq = w_vec[0:n_pv]
M_yPVp = W_mat[n_pv:2*n_pv, :]
m_yPVp = w_vec[n_pv:2*n_pv]
M_ylb = W_mat[2*n_pv:2*n_pv+n_branches, :]
m_ylb = w_vec[2*n_pv:2*n_pv+n_branches]
M_ylu = W_mat[2*n_pv+n_branches:2*n_pv+2*n_branches, :]
m_ylu = w_vec[2*n_pv+n_branches:2*n_pv+2*n_branches]

# 状态变量 P, Q
M_Pplus = -C @ M_xPL + C_MPV @ M_yPVp - D_R @ M_ylb
m_Pplus = C_MPV @ m_yPVp - D_R @ m_ylb
M_Pminus = -C @ M_xPL + C_MPV @ M_yPVp - D_R @ M_ylu
m_Pminus = C_MPV @ m_yPVp - D_R @ m_ylu

M_Qplus = -C @ M_xQL + C_MPV @ M_yPVq - D_X_plus @ M_ylb - D_X_minus @ M_ylu
m_Qplus = C_MPV @ m_yPVq - D_X_plus @ m_ylb - D_X_minus @ m_ylu
M_Qminus = -C @ M_xQL + C_MPV @ M_yPVq - D_X_plus @ M_ylu - D_X_minus @ M_ylb
m_Qminus = C_MPV @ m_yPVq - D_X_plus @ m_ylu - D_X_minus @ m_ylb

# 状态变量 V
V_base_M = -Mp @ M_xPL - Mq @ M_xQL + Mp_MPV @ M_yPVp + Mq_MPV @ M_yPVq
V_base_m = V0_vec + Mp_MPV @ m_yPVp + Mq_MPV @ m_yPVq

M_Vplus = V_base_M - H_plus @ M_ylb - H_minus @ M_ylu
m_Vplus = V_base_m - H_plus @ m_ylb - H_minus @ m_ylu
M_Vminus = V_base_M - H_plus @ M_ylu - H_minus @ M_ylb
m_Vminus = V_base_m - H_plus @ m_ylu - H_minus @ m_ylb

# ------------------------------------------------------------------------------
# 4.2 添加鲁棒线性不等式约束
# ------------------------------------------------------------------------------
# 1. 电压下限
add_robust_linear_constr_cvxpy(constraints, -M_Vminus, V_MIN**2 - m_Vminus + s_slack, x_lb, x_ub)
# 2. 电压上限
add_robust_linear_constr_cvxpy(constraints, M_Vplus, m_Vplus - V_MAX**2 + s_slack, x_lb, x_ub)
# 3. 电流上限
add_robust_linear_constr_cvxpy(constraints, M_ylu, m_ylu - branch_max_array**2 + s_slack, x_lb, x_ub)
# 4. PV有功上限
add_robust_linear_constr_cvxpy(constraints, M_yPVp - M_xPVmax, m_yPVp + s_slack, x_lb, x_ub)
# 5. PV有功下限
add_robust_linear_constr_cvxpy(constraints, -M_yPVp, -m_yPVp + s_slack, x_lb, x_ub)

# 6. 电流下限 (线性化)
M_Vi_plus = branch_from_matrix @ M_Vplus
m_Vi_plus = is_from_root * (1.0**2) + branch_from_matrix @ m_Vplus
M_Vi_minus = branch_from_matrix @ M_Vminus
m_Vi_minus = is_from_root * (1.0**2) + branch_from_matrix @ m_Vminus

M_termP = np.diag(J_P_plus) @ M_Pplus + np.diag(J_P_minus) @ M_Pminus
m_termP = np.diag(J_P_plus) @ (m_Pplus - P0_flow) + np.diag(J_P_minus) @ (m_Pminus - P0_flow)
M_termQ = np.diag(J_Q_plus) @ M_Qplus + np.diag(J_Q_minus) @ M_Qminus
m_termQ = np.diag(J_Q_plus) @ (m_Qplus - Q0_flow) + np.diag(J_Q_minus) @ (m_Qminus - Q0_flow)
M_termV = np.diag(J_V_plus) @ M_Vi_plus + np.diag(J_V_minus) @ M_Vi_minus
m_termV = np.diag(J_V_plus) @ (m_Vi_plus - 1.0**2) + np.diag(J_V_minus) @ (m_Vi_minus - 1.0**2)

M_curr_lb = M_ylb - (M_termP + M_termQ + M_termV)
m_curr_lb = m_ylb - l0 - (m_termP + m_termQ + m_termV) + s_slack
add_robust_linear_constr_cvxpy(constraints, M_curr_lb, m_curr_lb, x_lb, x_ub)

# ------------------------------------------------------------------------------
# 4.3 二阶锥(SOC) 1-范数保守近似
# ------------------------------------------------------------------------------
# (A) PV 容量约束
constraints.append(tau_PVp + tau_PVq + s_slack - PV_capacity <= 0)
add_robust_linear_constr_cvxpy(constraints, M_yPVp, m_yPVp - tau_PVp, x_lb, x_ub)
add_robust_linear_constr_cvxpy(constraints, -M_yPVp, -m_yPVp - tau_PVp, x_lb, x_ub)
add_robust_linear_constr_cvxpy(constraints, M_yPVq, m_yPVq - tau_PVq, x_lb, x_ub)
add_robust_linear_constr_cvxpy(constraints, -M_yPVq, -m_yPVq - tau_PVq, x_lb, x_ub)

# (B) 电流下包络 SOC
add_robust_linear_constr_cvxpy(constraints, 2*M_Pplus, 2*m_Pplus - tau_P, x_lb, x_ub)
add_robust_linear_constr_cvxpy(constraints, -2*M_Pplus, -2*m_Pplus - tau_P, x_lb, x_ub)
add_robust_linear_constr_cvxpy(constraints, 2*M_Pminus, 2*m_Pminus - tau_P, x_lb, x_ub)
add_robust_linear_constr_cvxpy(constraints, -2*M_Pminus, -2*m_Pminus - tau_P, x_lb, x_ub)

add_robust_linear_constr_cvxpy(constraints, 2*M_Qplus, 2*m_Qplus - tau_Q, x_lb, x_ub)
add_robust_linear_constr_cvxpy(constraints, -2*M_Qplus, -2*m_Qplus - tau_Q, x_lb, x_ub)
add_robust_linear_constr_cvxpy(constraints, 2*M_Qminus, 2*m_Qminus - tau_Q, x_lb, x_ub)
add_robust_linear_constr_cvxpy(constraints, -2*M_Qminus, -2*m_Qminus - tau_Q, x_lb, x_ub)

M_R = M_ylu * V_MIN**2
m_R = m_ylu * V_MIN**2 - s_slack - np.ones(n_branches)
add_robust_linear_constr_cvxpy(constraints, M_R, m_R - tau_R, x_lb, x_ub)
add_robust_linear_constr_cvxpy(constraints, -M_R, -m_R - tau_R, x_lb, x_ub)

M_soc = -M_ylu * V_MIN**2
m_soc = tau_P + tau_Q + tau_R - m_ylu * V_MIN**2 + s_slack - np.ones(n_branches)
add_robust_linear_constr_cvxpy(constraints, M_soc, m_soc, x_lb, x_ub)

# ==============================================================================
# 5. 构建模型 + Gurobi 求解
# ==============================================================================
# 目标函数：最大化松弛变量
objective = cp.Maximize(s_slack)
prob = cp.Problem(objective, constraints)

print("Start optimizing End-to-End Robust Master Problem...")
start_time = time.time()
# 调用 Gurobi 求解器
prob.solve(
    solver=cp.GUROBI, 
    verbose=True, 
    Method=2,           
    Crossover=0,        
    Presolve=2,         
    BarConvTol=1e-5,
    # 增加这一行，专门对付残余的数值不稳定，开启齐次算法
    BarHomogeneous=1    
)
print(f"Optimization finished in {time.time() - start_time:.4f} seconds.")

# ==============================================================================
# 6. 结果提取与保存
# ==============================================================================
if prob.status == cp.OPTIMAL:
    print(f"Optimal Slack (s): {s_slack.value:.6f}")
    # 提取仿射系数
    W_opt = W_mat.value
    w_opt = w_vec.value
    
    # 提取PV相关系数
    M_yPVq = W_opt[0:n_pv, :]
    m_yPVq = w_opt[0:n_pv]
    M_yPVp = W_opt[n_pv:2*n_pv, :]
    m_yPVp = w_opt[n_pv:2*n_pv]
    
    # 保存结果
    np.savez('System_data/robust_affine_coefficients.npz',
             M_yPVq=M_yPVq,
             m_yPVq=m_yPVq,
             M_yPVp=M_yPVp,
             m_yPVp=m_yPVp)
    print("Affine coefficients saved to System_data/robust_affine_coefficients.npz")
else:
    print(f"Optimization failed! Status: {cp.SOLUTION_STATUS[prob.status]}")