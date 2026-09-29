
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from System_data.system_config import case33,Y_bus_matrix,PV_bus_define
from gurobipy import Model,GRB,quicksum,MVar,concatenate
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.sparse as sp
np.random.seed(2)
# import psutil
# p = psutil.Process()
# p.cpu_affinity(range(16))
plt.rcParams.update({'font.size': 16}) # 改变所有字体大小，改变其他性质类似
plt.rc('font',family='Times New Roman')

# 读取测试集数据
TEST_DATA = np.load("Data_generation/dataset.npy")
N_TEST_SAMPLES = TEST_DATA.shape[0]  # 测试集样本数
print(f"已加载测试集: {N_TEST_SAMPLES} 条样本")

# 选择测试样本索引 (0 ~ N_TEST_SAMPLES-1)
TEST_SAMPLE_IDX = 10  # 可以选择不同的样本进行测试

# 从测试集提取数据
# 测试集格式: [active_load(33), reactive_load(33), pv_power(5)]
n_bus = 33
n_pv = 5
ACTIVE_LOAD_ALL = TEST_DATA[:, :n_bus].T  # 形状: (33, N_TEST_SAMPLES)
REACTIVE_LOAD_ALL = TEST_DATA[:, n_bus:2*n_bus].T  # 形状: (33, N_TEST_SAMPLES)
PV_P_POWER_ALL = TEST_DATA[:, 2*n_bus:]  # 形状: (N_TEST_SAMPLES, 5)

# 获取电网拓扑数据
SYSTEM_DATA = case33()
branch = np.array(SYSTEM_DATA['branch'])
bus = np.array(SYSTEM_DATA['bus'])
PV_bus,PV_capacity = PV_bus_define()
R_ij_matrix,X_ij_matrix,r_x_ratio,branch_max  = Y_bus_matrix()
V_MAX,V_MIN = 1.05, 0.95
V0 = 1.0 # Base voltage

# 系统基准功率（在模型和pandapower验证中都需要使用）
S_BASE_MVA = 10  # 基准功率 MVA

# 弃光选项：设置为True启用弃光，False则强制使用全部光伏功率
ENABLE_CURTAILMENT = True  # 修改此参数即可控制是否允许弃光

# Load Data
PV_p_power = PV_P_POWER_ALL[TEST_SAMPLE_IDX, :]
ACTIVE_LOAD = ACTIVE_LOAD_ALL[:, TEST_SAMPLE_IDX]
REACTIVE_LOAD = REACTIVE_LOAD_ALL[:, TEST_SAMPLE_IDX]

# ==============================================================================
# SECTION III.B: Inner Convex Approximation Implementation
# ==============================================================================

# 1. Construct Matrices based on Paper Definitions
n_nodes = bus.shape[0]       # 33 (Including Slack)
n_branches = branch.shape[0] # 32

# f_nodes[k] 表示第k条支路的起始节点 (0-based，0表示根节点)
f_nodes = branch[:, 0].astype(int) - 1  # 转换为0-based索引

# branch_from_matrix[k, j] = 1 如果支路k的起始节点是j+1 (即非根节点j)
# 维度: (n_branches, n_branches) - 每行最多一个1，表示该支路的父节点是哪个
branch_from_matrix = np.zeros((n_branches, n_branches))
for k in range(n_branches):
    f_node = f_nodes[k]
    if f_node > 0:  # 非根节点，对应 V_plus[f_node-1] 或 V_minus[f_node-1]
        branch_from_matrix[k, f_node - 1] = 1.0
    # 如果 f_node == 0，该行全为0，表示使用根节点电压 1.0**2

# 标记哪些支路起始于根节点（用于后面加常数项）
is_from_root = (f_nodes == 0).astype(float)  # shape: (n_branches,)

# --- 修正点：构建论文定义的 0/1 关联矩阵 B ---
# 定义来源于: "entry is 1 if node is connected to branch, 0 otherwise"
# B 维度: (n+1) x n_branches (33 x 32)
B = np.zeros((n_nodes, n_branches))

# 记录支路连接关系用于后续拓扑分析
# 假设 branch 数据格式: [from_bus, to_bus, r, x]
branch_map = {} 
for k in range(n_branches):
    f_node = int(branch[k, 0]) - 1 # Python 0-indexed
    t_node = int(branch[k, 1]) - 1
    
    # 根据论文定义，只要节点与支路相连，对应元素即为 1
    B[f_node, k] = 1 
    B[t_node, k] = 1
    
    branch_map[(f_node, t_node)] = k

# --- 构建支路关联矩阵 A ---
# 我们可以通过拓扑搜索直接构建 A，与 B 矩阵计算等价
A = np.zeros((n_branches, n_branches))

# 既然是辐射状网络，每一条支路 k (t_node) 的下游支路，
# 就是所有以 t_node 为起始节点的支路。
for k in range(n_branches):
    # 支路 k 的末端节点
    t_node_k = int(branch[k, 1]) - 1
    
    # 寻找所有从 t_node_k 出发的支路 m (即 k 的下游支路)
    for m in range(n_branches):
        f_node_m = int(branch[m, 0]) - 1
        if f_node_m == t_node_k:
            A[k, m] = 1 # 支路 m 是 支路 k 的直接下游

# --- 计算矩阵 C = (I - A)^-1 ---
# C 矩阵的物理含义是：C[k, m] = 1 表示支路 m 在支路 k 的下游（或者是 k 本身）。
# 用于累加下游的负荷：P_branch = C * p_injection
I_n = np.eye(n_branches)
try:
    C = np.linalg.inv(I_n - A)
except np.linalg.LinAlgError:
    # 如果不可逆（通常不会发生），使用伪逆或检查拓扑
    print("Warning: Matrix (I-A) is singular. Check topology.")
    C = np.linalg.pinv(I_n - A)

# --- 构建电阻、电抗、阻抗矩阵 ---
r_vec = branch[:, 2] * r_x_ratio 
x_vec = branch[:, 3] * r_x_ratio
z2_vec = r_vec**2 + x_vec**2

R = np.diag(r_vec)
X = np.diag(x_vec)
Z2 = np.diag(z2_vec)

# --- 计算 Mp, Mq, H ---

Mp = 2 * (C.T @ R @ C)
Mq = 2 * (C.T @ X @ C)

# H 的计算需要 D_R 和 D_X
D_R = C @ A @ R
D_X = C @ A @ X

# H = C.T * (2 * R * D_R + 2 * X * D_X + Z2)
# 注意：这里的 C 是 n x n (32x32)。得到的 Mp, Mq 也是 32x32，对应除去 Slack 节点的 32 个节点。
H_term_inner = 2 * R @ D_R + 2 * X @ D_X + Z2
H = C.T @ H_term_inner

# --- 辅助变量的正负部划分 ---
D_X_plus = np.maximum(D_X, 0)
D_X_minus = np.minimum(D_X, 0)
H_plus = np.maximum(H, 0)
H_minus = np.minimum(H, 0)

# --- 线性化点 (Nominal Operating Point)l ---
# 用于泰勒展开 (Eq 8)
baseline = np.load('System_data/baseline_operating_point.npz')
V0_mag = baseline['Bus_V'][:-1]  # 基准电压幅值 (33,)
P0_flow = baseline['P_ij']
Q0_flow = baseline['Q_ij']
l0 = baseline['branch_current'] 

# 计算雅可比矩阵 J (Eq 7)
J_P = 2 * P0_flow / (V0_mag**2)
J_Q = 2 * Q0_flow / (V0_mag**2)
J_V = -1 * (P0_flow**2 + Q0_flow**2) / (V0_mag**2)
J_P_plus, J_P_minus = np.maximum(J_P, 0), np.minimum(J_P, 0)
J_Q_plus, J_Q_minus = np.maximum(J_Q, 0), np.minimum(J_Q, 0)
J_V_plus, J_V_minus = np.maximum(J_V, 0), np.minimum(J_V, 0)

# ==============================================================================
# OPTIMIZATION MODEL
# ==============================================================================
model = Model('InnerConvexApprox')

# --- 决策变量 ---
# PV 无功出力
PV_q_power = model.addMVar(len(PV_bus), lb=-GRB.INFINITY, ub=GRB.INFINITY, name='PV_q')
# PV 实际有功出力（弃光决策变量）
PV_p_actual = model.addMVar(len(PV_bus), lb=0, vtype=GRB.CONTINUOUS, name='PV_p_actual')

# 电流平方的上下界 (l^b, l^u)
l_b = model.addMVar(n_branches, lb=-GRB.INFINITY,  name='l_b')
l_u = model.addMVar(n_branches, lb=-GRB.INFINITY, ub = branch_max**2, name='l_u')

# 状态变量辅助变量 (P+, P-, Q+, Q-, V+, V-) [cite: 145]
# 维度为 32 (对应 n_branches 或 n_nodes-1)
P_plus = model.addMVar(n_branches, lb=-GRB.INFINITY, name='P_plus')
P_minus = model.addMVar(n_branches, lb=-GRB.INFINITY, name='P_minus')
Q_plus = model.addMVar(n_branches, lb=-GRB.INFINITY, name='Q_plus')
Q_minus = model.addMVar(n_branches, lb=-GRB.INFINITY, name='Q_minus')
V_plus = model.addMVar(n_nodes-1, lb=V_MIN**2, ub=V_MAX**2, name='V_plus') 
V_minus = model.addMVar(n_nodes-1, lb=V_MIN**2, ub=V_MAX**2, name='V_minus')

# --- 构造注入向量 p, q ---
# 除去 Slack 节点 (索引 0)
active_load_vec = ACTIVE_LOAD[1:]
reactive_load_vec = REACTIVE_LOAD[1:]

# p 向量: PV 有功 - 负荷有功
# 注意：现在 PV 有功是变量 PV_p_actual，不再是常数 PV_p_power
# 创建有功注入变量向量
p_inj_expr = model.addMVar(n_nodes-1, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='p_inj')

# 设置非 PV 节点的 p_inj_expr 为常数（仅负荷）
for i in range(n_nodes-1):
    # 检查是否是 PV 节点
    is_pv_node = False
    for bus_idx in PV_bus:
        map_idx = bus_idx - 1 - 1
        if map_idx == i:
            is_pv_node = True
            break
    if not is_pv_node:
        model.addConstr(p_inj_expr[i] == -active_load_vec[i])

# 设置 PV 节点的 p_inj_expr = PV 有功变量 - 负荷
for k, bus_idx in enumerate(PV_bus):
    map_idx = bus_idx - 1 - 1
    if map_idx >= 0:
        model.addConstr(p_inj_expr[map_idx] == PV_p_actual[k] - active_load_vec[map_idx])

# q 向量构建 (包含变量)
# q = -Load_Q + PV_Q
# 构建无功注入向量 q_inj_vec（常数部分）
q_inj_vec_const = -reactive_load_vec.copy()

# 标记 PV 节点位置
pv_node_indices = set()
for bus_idx in PV_bus:
    map_idx = bus_idx - 1 - 1
    if map_idx >= 0:
        pv_node_indices.add(map_idx)

# 创建无功注入变量向量
q_inj_expr = model.addMVar(n_nodes-1, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='q_inj')

# 设置非 PV 节点的 q_inj_expr 为常数
for i in range(n_nodes-1):
    if i not in pv_node_indices:
        model.addConstr(q_inj_expr[i] == q_inj_vec_const[i])

# 设置 PV 节点的 q_inj_expr = 常数 + PV 无功变量
for k, bus_idx in enumerate(PV_bus):
    map_idx = bus_idx - 1 - 1
    if map_idx >= 0:
        model.addConstr(q_inj_expr[map_idx] == q_inj_vec_const[map_idx] + PV_q_power[k])

# --- 约束条件 ---

# 1. 状态变量定义 (Eq 6a - 6d)
# P+ = C*p - D_R*l_b
# C @ p_inj_expr 现在包含变量
Cp_expr = C @ p_inj_expr
model.addConstr(P_plus == Cp_expr - D_R @ l_b)
model.addConstr(P_minus == Cp_expr - D_R @ l_u)

# Q+ = C*q - D_X+ * l_b - D_X- * l_u
# Cq 包含变量
Cq_expr = C @ q_inj_expr
model.addConstr(Q_plus == Cq_expr - D_X_plus @ l_b - D_X_minus @ l_u)
model.addConstr(Q_minus == Cq_expr - D_X_plus @ l_u - D_X_minus @ l_b)

# V+ = V0 + Mp*p + Mq*q - H+ * l_b - H- * l_u
# 基准电压项
V0_vec = np.ones(n_nodes-1) * (1.0**2) 
Base_V_term = V0_vec + Mp @ p_inj_expr

Mq_term = Mq @ q_inj_expr

model.addConstr(V_plus == Base_V_term + Mq_term - H_plus @ l_b - H_minus @ l_u)
model.addConstr(V_minus == Base_V_term + Mq_term - H_plus @ l_u - H_minus @ l_b)

# 2. PV 容量约束 (Eq 1h) - 使用实际出力 PV_p_actual
for k in range(len(PV_bus)):
    model.addConstr(PV_q_power[k]**2 + PV_p_actual[k]**2 <= PV_capacity**2)

# 3. 弃光约束
if ENABLE_CURTAILMENT:
    # 启用弃光：实际出力不超过最大可用功率
    for k in range(len(PV_bus)):
        model.addConstr(PV_p_actual[k] <= PV_p_power[k])
else:
    # 禁用弃光：实际出力必须等于最大可用功率
    for k in range(len(PV_bus)):
        model.addConstr(PV_p_actual[k] == PV_p_power[k])

# 3. 电流下界约束
V_i_plus = is_from_root * (1.0**2) + branch_from_matrix @ V_plus
V_i_minus = is_from_root * (1.0**2) + branch_from_matrix @ V_minus
V_i_0 = 1.0**2

d_P_plus = P_plus - P0_flow
d_P_minus = P_minus - P0_flow

d_Q_plus = Q_plus - Q0_flow
d_Q_minus = Q_minus - Q0_flow

d_V_plus = V_i_plus - V_i_0
d_V_minus = V_i_minus - V_i_0

# 根据 Jacobian 正负选择对应的 bound (Eq 8)
term_P = J_P_plus * d_P_plus + J_P_minus * d_P_minus
term_Q = J_Q_plus * d_Q_plus + J_Q_minus * d_Q_minus
term_V = J_V_plus * d_V_plus + J_V_minus * d_V_minus

# 电流下界约束
model.addConstr(l_b <= l0 + term_P + term_Q + term_V)


# 4. 电流上界二阶锥约束 (Eq 9)
# l_u * V_minus >= P^2 + Q^2
v_lim_vec = is_from_root * (1.0**2) + branch_from_matrix @ V_minus
    
# 对四个角点分别约束
model.addConstr(l_u * v_lim_vec >= P_plus**2 + Q_plus**2)
model.addConstr(l_u * v_lim_vec >= P_plus**2 + Q_minus**2)
model.addConstr(l_u * v_lim_vec >= P_minus**2 + Q_plus**2)
model.addConstr(l_u * v_lim_vec >= P_minus**2 + Q_minus**2)

# 5. 目标函数

obj = 0.5 * 10**4 * p_inj_expr[0]

# 添加弃光惩罚项（仅在启用弃光时，避免不必要的弃光）
if ENABLE_CURTAILMENT:
    CURTAILMENT_PENALTY = 0.3 * 10**4  # 弃光惩罚系数，可根据需要调整
    curtailment_amount = sum(PV_p_power[i] - PV_p_actual[i] for i in range(len(PV_bus)))
    obj = obj + CURTAILMENT_PENALTY * curtailment_amount

model.setObjective(obj, GRB.MINIMIZE)
model.optimize()

# --- 输出处理 ---
if model.status == GRB.OPTIMAL:
    print(f"Inner Convex Approx Optimal: {model.ObjVal}")
elif model.status == GRB.SUBOPTIMAL:
    print(f"Warning: Model returned suboptimal solution. Objective: {model.ObjVal}")
elif model.status == GRB.NUMERIC:
    print("Warning: Numerical issues encountered. Trying to use available solution...")
else:
    print(f"Model status: {model.status}")

# 检查是否有解
if model.SolCount == 0:
    print("Error: No solution available.")
    if model.status == GRB.INFEASIBLE:
        print("\n" + "="*60)
        print("模型不可行！正在计算 IIS (最小不可行子系统)...")
        print("="*60)
        
        # 计算 IIS
        model.computeIIS()
        
        # 输出 IIS 约束
        print("\nIIS 约束 (导致不可行的最小约束集):")
        print("-"*60)
        
        constr_count = 0
        for c in model.getConstrs():
            if c.IISConstr:
                print(f"  线性约束: {c.ConstrName}")
                constr_count += 1
        
        for c in model.getQConstrs():
            if c.IISQConstr:
                print(f"  二次约束: {c.QCName}")
                constr_count += 1
        
        for v in model.getVars():
            if v.IISLB:
                print(f"  变量下界: {v.VarName} >= {v.LB}")
                constr_count += 1
            if v.IISUB:
                print(f"  变量上界: {v.VarName} <= {v.UB}")
                constr_count += 1
        
        print("-"*60)
        print(f"总共 {constr_count} 个约束/边界导致不可行")
        print("="*60)
        
        # 将 IIS 写入文件以便分析
        model.write("model_iis.ilp")
        print("\nIIS 已保存到 model_iis.ilp 文件")
    
    raise RuntimeError("模型无可行解，请检查约束条件")

# 将结果传回给后续的验证代码
# 构造 Bus_V 用于绘图 (使用 V_minus 作为保守估计)
Bus_V_vals = np.zeros(n_nodes)
Bus_V_vals[0] = 1.0
Bus_V_vals[1:] = V_minus.X

class ResultContainer:
    def __init__(self, x): self.X = x
    
Bus_V = ResultContainer(Bus_V_vals)
# PV_q_power 和 PV_p_actual 已经是 MVar，可以直接取 X

# 打印弃光信息
if ENABLE_CURTAILMENT:
    curtailment_total = sum(PV_p_power[k] - PV_p_actual.X[k] for k in range(len(PV_bus)))
    print("\n" + "="*60)
    print("PV 弃光分析")
    print("="*60)
    if curtailment_total > 1e-6:
        print(f"{'PV Node':<12}{'Max Power (p.u.)':<18}{'Actual (p.u.)':<18}{'Curtailment (p.u.)':<20}{'Ratio (%)':<12}")
        print("-"*60)
        for k in range(len(PV_bus)):
            max_p = PV_p_power[k]
            actual_p = PV_p_actual.X[k]
            curt = max_p - actual_p
            ratio = curt / max_p * 100 if max_p > 1e-6 else 0
            if curt > 1e-6:
                print(f"Bus {PV_bus[k]:<6}{max_p:<18.6f}{actual_p:<18.6f}{curt:<20.6f}{ratio:<12.2f}")
        print("-"*60)
        print(f"总弃光量: {curtailment_total:.6f} p.u. = {curtailment_total * S_BASE_MVA:.4f} MW")
        print(f"总可用光伏功率: {sum(PV_p_power):.6f} p.u.")
        print(f"总弃光比例: {curtailment_total / sum(PV_p_power) * 100:.2f}%")
    else:
        print("无弃光发生（所有光伏满发）")
    print("="*60)

# 保存结果供后续使用
pv_q_opt = PV_q_power.X
pv_p_opt = PV_p_actual.X
pv_p_max = PV_p_power
curtailment = pv_p_max - pv_p_opt

"""
使用 pandapower 计算实际调度结果对应的电压值
"""
import pandapower as pp
from System_data.create_pandapower_network import load_network
net = load_network()

# 系统基准电压
V_BASE_KV = 12.66  # kV

# 添加负荷 (所有节点) - 使用MW和Mvar单位
for i in range(bus.shape[0]):
    p_mw = ACTIVE_LOAD[i] * S_BASE_MVA  # 转换为MW
    q_mvar = REACTIVE_LOAD[i] * S_BASE_MVA  # 转换为Mvar
    if abs(p_mw) > 1e-9 or abs(q_mvar) > 1e-9:
        pp.create_load(net, bus=i, p_mw=p_mw, q_mvar=q_mvar, name=f"Load {i+1}")

# 添加 PV 发电单元 (根据优化结果设置无功出力)
count_PV = 0
for i in range(bus.shape[0]):
    if np.isin(i+1, PV_bus):  # PV 连接节点
        p_mw = PV_p_actual.X[count_PV] * S_BASE_MVA  # 使用优化后的实际有功出力(转换为MW)
        q_mvar = PV_q_power.X[count_PV] * S_BASE_MVA  # 转换为Mvar
        # 在pandapower中，sgen的p_mw正值表示向电网注入有功
        pp.create_sgen(net, bus=i, p_mw=p_mw, q_mvar=q_mvar, name=f"PV {i+1}")
        count_PV += 1

# 添加主网交换功率 (在平衡节点通过ext_grid自动平衡，不需要额外添加)
# 打印网络信息用于调试
print("\n网络信息:")
print(f"母线数量: {len(net.bus)}")
print(f"线路数量: {len(net.line)}")
print(f"负荷数量: {len(net.load)}")
print(f"分布式电源数量: {len(net.sgen)}")
print(f"总负荷有功: {net.load.p_mw.sum()*1000:.2f} kW")
print(f"总发电有功: {net.sgen.p_mw.sum()*1000:.2f} kW")

# 运行潮流计算，使用DC潮流作为初始值提高收敛性
try:
    pp.runpp(net, algorithm='nr', init='dc', max_iteration=50)
    print("潮流计算收敛成功！")
except pp.powerflow.LoadflowNotConverged:
    print("第一次尝试失败，尝试使用平坦启动...")
    try:
        pp.runpp(net, algorithm='nr', init='flat', max_iteration=100)
        print("潮流计算收敛成功！(使用平坦启动)")
    except pp.powerflow.LoadflowNotConverged:
        print("第二次尝试失败，尝试调整发电机类型...")
        # 尝试将sgen转换为gen
        for i in range(len(net.sgen)):
            bus_idx = net.sgen.bus.iloc[i]
            p_mw = net.sgen.p_mw.iloc[i]
            q_mvar = net.sgen.q_mvar.iloc[i]
            pp.create_gen(net, bus=bus_idx, p_mw=p_mw, vm_pu=1.0, name=f"Gen {i+1}")
        # 清空sgen
        net.sgen.drop(net.sgen.index, inplace=True)
        pp.runpp(net, algorithm='nr', init='dc', max_iteration=100)
        print("潮流计算收敛成功！(使用gen模型)")

# 提取电压结果
v_opt_list = []
v_pp_list = []
for i in range(bus.shape[0]):
    v_opt = np.sqrt(Bus_V.X[i])  # 优化模型中存储的是电压平方
    v_pp = net.res_bus.vm_pu.iloc[i]
    v_opt_list.append(v_opt)
    v_pp_list.append(v_pp)

# 计算电压误差统计
max_error = max(abs(v_opt_list[i] - v_pp_list[i]) for i in range(bus.shape[0]))
mean_error = np.mean([abs(v_opt_list[i] - v_pp_list[i]) for i in range(bus.shape[0])])

# ==================== 电压违约分析 ====================
print("\n" + "="*70)
print("节点电压违约分析 (基于 Pandapower 实际计算结果)")
print("="*70)
print(f"电压约束: Vmin = {V_MIN:.2f} p.u., Vmax = {V_MAX:.2f} p.u.")
print("-"*70)

voltage_violations = []
for i in range(bus.shape[0]):
    v_pp = v_pp_list[i]
    if v_pp > V_MAX  + 1e-6:
        violation = v_pp - V_MAX
        voltage_violations.append((i+1, v_pp, violation, 'OVER'))
    elif v_pp < V_MIN - 1e-6:
        violation = V_MIN - v_pp
        voltage_violations.append((i+1, v_pp, violation, 'UNDER'))

if voltage_violations:
    print(f"{'节点':<10}{'电压 (p.u.)':<18}{'违约类型':<15}{'违约量 (p.u.)':<18}{'严重程度'}")
    print("-"*70)
    for bus_id, v, viol, vtype in voltage_violations:
        severity = '严重' if viol > 0.01 else '轻微'
        vtype_str = '电压越上限' if vtype == 'OVER' else '电压越下限'
        print(f"{bus_id:<10}{v:<18.6f}{vtype_str:<15}{viol:<18.6f}{severity}")
    print("-"*70)
    print(f"总违约节点数: {len(voltage_violations)} / {bus.shape[0]}")
    print(f"违约节点比例: {len(voltage_violations)/bus.shape[0]*100:.2f}%")
else:
    print("✓ 所有节点电压均满足约束条件 (无违约)")
print("="*70)

# 可视化对比
plt.figure(figsize=(14, 5))

# 电压对比图
plt.subplot(1, 2, 1)
bus_numbers = list(range(1, bus.shape[0] + 1))
plt.plot(bus_numbers, v_opt_list, 'b-o', label='SOCR', markersize=6)
plt.plot(bus_numbers, v_pp_list, 'r-s', label='Pandapower', markersize=6)
plt.axhline(y=V_MAX, color='g', linestyle='--', label='Vmax = 1.05')
plt.axhline(y=V_MIN, color='g', linestyle='--', label='Vmin = 0.95')
plt.xlabel('Bus Number')
plt.ylabel('Voltage (p.u.)')
plt.title('Voltage Profile Comparison')
plt.legend()
plt.grid(True)

# 误差图
plt.subplot(1, 2, 2)
errors = [abs(v_opt_list[i] - v_pp_list[i]) for i in range(bus.shape[0])]
plt.bar(bus_numbers, errors, color='orange', alpha=0.7)
plt.xlabel('Bus Number')
plt.ylabel('Voltage Error (p.u.)')
plt.title('Voltage Error between Model and Pandapower')
plt.grid(True, axis='y')

plt.tight_layout()
# plt.show()

# ==================== PV 无功输出示意图 ====================
plt.figure(figsize=(12, 5))

# 获取 PV 优化结果
pv_q_opt = PV_q_power.X  # 无功输出 (p.u.)
pv_p_opt = PV_p_actual.X  # 有功输出 (p.u.) - 使用优化后的实际出力
pv_p_max = PV_p_power  # 最大可用有功功率 (p.u.)
pv_capacity = PV_capacity  # 装机容量

# 计算弃光量
curtailment = pv_p_max - pv_p_opt

# 计算无功调节范围 (受视在功率约束限制)
# Q_max = sqrt(S^2 - P^2), Q_min = -sqrt(S^2 - P^2)
pv_q_max = np.sqrt(pv_capacity**2 - pv_p_opt**2)
pv_q_min = -pv_q_max

x_pos = np.arange(len(PV_bus))
width = 0.35

# 子图1: PV 无功功率柱状图
plt.subplot(1, 2, 1)
bars1 = plt.bar(x_pos - width/2, pv_q_opt, width, label='Q_PV (Optimized)', color='steelblue', edgecolor='black')
plt.axhline(y=0, color='k', linestyle='-', linewidth=0.8)
plt.xlabel('PV Node')
plt.ylabel('Reactive Power (p.u.)')
plt.title('PV Reactive Power Output')
plt.xticks(x_pos, [f'Bus {b}' for b in PV_bus])
plt.legend()
plt.grid(True, axis='y', alpha=0.3)

# 添加数值标注
for bar, val in zip(bars1, pv_q_opt):
    height = bar.get_height()
    plt.annotate(f'{val:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3 if height >= 0 else -12),
                textcoords="offset points",
                ha='center', va='bottom' if height >= 0 else 'top',
                fontsize=9)

# 子图2: PV 有功 vs 无功 (散点图或对比图)
plt.subplot(1, 2, 2)
# 绘制无功调节范围区间
for i in range(len(PV_bus)):
    plt.plot([i, i], [pv_q_min[i], pv_q_max[i]], 'g--', alpha=0.5, linewidth=4, label='Q bounds' if i == 0 else '')

plt.scatter(x_pos, pv_p_opt, s=150, c='orange', marker='o', edgecolors='black', label='P_PV (Active)', zorder=5)
plt.scatter(x_pos, pv_q_opt, s=150, c='steelblue', marker='s', edgecolors='black', label='Q_PV (Reactive)', zorder=5)

plt.axhline(y=0, color='k', linestyle='-', linewidth=0.8)
plt.xlabel('PV Node')
plt.ylabel('Power (p.u.)')
plt.title('PV Active vs Reactive Power')
plt.xticks(x_pos, [f'Bus {b}' for b in PV_bus])
plt.legend()
plt.grid(True, alpha=0.3)

# 添加功率因数信息
pf = pv_p_opt / np.sqrt(pv_p_opt**2 + pv_q_opt**2)
for i in range(len(PV_bus)):
    plt.annotate(f'PF={pf[i]:.3f}',
                xy=(x_pos[i], min(pv_q_min[i], 0) - 0.15),
                ha='center', fontsize=9, color='purple')

plt.tight_layout()

# ==================== PV 弃光分析 ====================
if ENABLE_CURTAILMENT and np.sum(curtailment) > 1e-6:
    plt.figure(figsize=(12, 4))
    
    # 弃光量柱状图
    plt.subplot(1, 2, 1)
    bars = plt.bar(x_pos, curtailment, color='red', alpha=0.7, edgecolor='black')
    plt.xlabel('PV Node')
    plt.ylabel('Curtailment (p.u.)')
    plt.title(f'PV Curtailment (Total: {np.sum(curtailment):.4f} p.u.)')
    plt.xticks(x_pos, [f'Bus {b}' for b in PV_bus])
    plt.grid(True, axis='y', alpha=0.3)
    
    # 添加数值标注
    for bar, val in zip(bars, curtailment):
        if val > 1e-6:
            height = bar.get_height()
            plt.annotate(f'{val:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=9)
    
    # 弃光比例
    plt.subplot(1, 2, 2)
    curtailment_ratio = np.where(pv_p_max > 1e-6, curtailment / pv_p_max * 100, 0)
    bars = plt.bar(x_pos, curtailment_ratio, color='orange', alpha=0.7, edgecolor='black')
    plt.xlabel('PV Node')
    plt.ylabel('Curtailment Ratio (%)')
    plt.title(f'PV Curtailment Ratio (Avg: {np.mean(curtailment_ratio):.2f}%)')
    plt.xticks(x_pos, [f'Bus {b}' for b in PV_bus])
    plt.grid(True, axis='y', alpha=0.3)
    
    # 添加数值标注
    for bar, val in zip(bars, curtailment_ratio):
        if val > 1e-6:
            height = bar.get_height()
            plt.annotate(f'{val:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=9)
    
    plt.tight_layout()

# ==================== 调度结果：支路潮流和节点电压分布 ====================
plt.figure(figsize=(14, 10))

# --- 子图1: 节点电压分布 (以到平衡节点的电气距离为横轴) ---
plt.subplot(2, 2, 1)
# 计算每个节点到平衡节点的最短路径跳数（作为电气距离的近似）
def get_distance_from_root(branch, n_bus):
    """计算每个节点到根节点(节点1)的距离"""
    distances = [0] + [float('inf')] * (n_bus - 1)
    changed = True
    while changed:
        changed = False
        for i in range(branch.shape[0]):
            from_bus = int(branch[i, 0]) - 1
            to_bus = int(branch[i, 1]) - 1
            if distances[from_bus] + 1 < distances[to_bus]:
                distances[to_bus] = distances[from_bus] + 1
                changed = True
            if distances[to_bus] + 1 < distances[from_bus]:
                distances[from_bus] = distances[to_bus] + 1
                changed = True
    return distances

distances = get_distance_from_root(branch, bus.shape[0])
bus_numbers = list(range(1, bus.shape[0] + 1))

# 按节点编号排序的电压分布
colors = plt.cm.viridis(np.linspace(0, 1, bus.shape[0]))
scatter = plt.scatter(bus_numbers, v_pp_list, c=distances, s=100, cmap='viridis', edgecolors='black', zorder=5)
plt.axhline(y=V_MAX, color='r', linestyle='--', linewidth=1.5, label='Vmax = 1.05')
plt.axhline(y=V_MIN, color='r', linestyle='--', linewidth=1.5, label='Vmin = 0.95')
plt.axhline(y=1.0, color='gray', linestyle=':', linewidth=1, label='V = 1.0')
plt.xlabel('Bus Number')
plt.ylabel('Voltage (p.u.)')
plt.title('Node Voltage Distribution')
plt.colorbar(scatter, label='Distance from Root')
plt.legend(loc='lower left')
plt.grid(True, alpha=0.3)
plt.xlim(0, bus.shape[0] + 1)

# --- 子图2: 支路有功潮流分布 ---
plt.subplot(2, 2, 2)
# 提取支路潮流结果
if len(net.res_line) > 0:
    branch_names = [f'{int(branch[i,0])}-{int(branch[i,1])}' for i in range(branch.shape[0])]
    p_from = net.res_line['p_from_mw'].values / 10  # 转换回 p.u.
    
    colors_flow = ['green' if p > 0 else 'orange' for p in p_from]
    bars = plt.barh(range(len(p_from)), p_from, color=colors_flow, alpha=0.7, edgecolor='black')
    plt.axvline(x=0, color='k', linestyle='-', linewidth=0.8)
    plt.xlabel('Active Power Flow (p.u.)')
    plt.ylabel('Branch')
    plt.title('Branch Active Power Flow')
    plt.yticks(range(0, len(p_from), 4), [branch_names[i] for i in range(0, len(branch_names), 4)])
    plt.grid(True, axis='x', alpha=0.3)
    
    # 添加流向标注
    plt.text(0.95, 0.95, '→ Downstream\n← Upstream', transform=plt.gca().transAxes,
             fontsize=9, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# --- 子图3: 支路无功潮流分布 ---
plt.subplot(2, 2, 3)
if len(net.res_line) > 0:
    q_from = net.res_line['q_from_mvar'].values / 10  # 转换回 p.u.
    
    colors_flow = ['steelblue' if q > 0 else 'coral' for q in q_from]
    bars = plt.barh(range(len(q_from)), q_from, color=colors_flow, alpha=0.7, edgecolor='black')
    plt.axvline(x=0, color='k', linestyle='-', linewidth=0.8)
    plt.xlabel('Reactive Power Flow (p.u.)')
    plt.ylabel('Branch')
    plt.title('Branch Reactive Power Flow')
    plt.yticks(range(0, len(q_from), 4), [branch_names[i] for i in range(0, len(branch_names), 4)])
    plt.grid(True, axis='x', alpha=0.3)

# --- 子图4: 支路负载率 ---
plt.subplot(2, 2, 4)
if len(net.res_line) > 0:
    loading = net.res_line['loading_percent'].values
    
    colors_load = ['green' if l < 50 else 'orange' if l < 80 else 'red' for l in loading]
    bars = plt.bar(range(len(loading)), loading, color=colors_load, alpha=0.7, edgecolor='black')
    plt.axhline(y=100, color='red', linestyle='--', linewidth=2, label='100% (Thermal Limit)')
    plt.axhline(y=80, color='orange', linestyle='--', linewidth=1.5, label='80% (Warning)')
    plt.xlabel('Branch Index')
    plt.ylabel('Loading (%)')
    plt.title('Branch Loading Rate')
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    
    # 标注最大负载率
    max_idx = np.argmax(loading)
    plt.annotate(f'Max: {loading[max_idx]:.1f}%',
                xy=(max_idx, loading[max_idx]),
                xytext=(max_idx, loading[max_idx] + 10),
                ha='center', fontsize=9, color='red',
                arrowprops=dict(arrowstyle='->', color='red'))

plt.tight_layout()


# 计算电流基准值
I_base_ka = S_BASE_MVA / (np.sqrt(3) * V_BASE_KV)  # 电流基准值 (kA)


# 获取 pandapower 实际电流并转换为标幺值
if len(net.res_line) > 0:
    # pandapower 提供的是 i_from_ka 和 i_to_ka，取较大值
    i_from_ka = net.res_line['i_from_ka'].values
    i_to_ka = net.res_line['i_to_ka'].values
    i_actual_ka = np.maximum(i_from_ka, i_to_ka)
    i_actual_pu = i_actual_ka / I_base_ka
else:
    i_actual_pu = np.zeros(branch.shape[0])

branch_names = [f'{int(branch[i,0])}-{int(branch[i,1])}' for i in range(branch.shape[0])]
branch_indices = list(range(1, branch.shape[0] + 1))


# ==================== 支路电流违约分析 ====================
print("\n" + "="*70)
print("支路电流违约分析 (基于 Pandapower 实际计算结果)")
print("="*70)
print(f"电流约束: Imax = {branch_max:.4f} p.u.")
print("-"*70)

current_violations = []
for i in range(branch.shape[0]):
    i_actual = i_actual_pu[i]
    if i_actual > branch_max + 1e-6:  # 考虑数值误差，设置一个小的容忍度
        violation = i_actual - branch_max
        violation_ratio = violation / branch_max * 100
        current_violations.append((i+1, branch_names[i], i_actual, violation, violation_ratio))

if current_violations:
    print(f"{'支路索引':<10}{'支路名称':<12}{'电流 (p.u.)':<18}{'违约量 (p.u.)':<18}{'违约比例 (%)':<15}{'严重程度'}")
    print("-"*90)
    for idx, name, i_val, viol, ratio in current_violations:
        severity = '严重' if ratio > 5 else '轻微'
        print(f"{idx:<10}{name:<12}{i_val:<18.6f}{viol:<18.6f}{ratio:<15.2f}{severity}")
    print("-"*90)
    print(f"总违约支路数: {len(current_violations)} / {branch.shape[0]}")
    print(f"违约支路比例: {len(current_violations)/branch.shape[0]*100:.2f}%")
    print(f"最大违约比例: {max([v[4] for v in current_violations]):.2f}%")
else:
    print("✓ 所有支路电流均满足约束条件 (无违约)")
print("="*70)

plt.show()