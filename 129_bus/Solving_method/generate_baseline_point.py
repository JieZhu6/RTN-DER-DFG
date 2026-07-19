"""
生成基准运行点（基于测试集均值） 用Gurobi求全局最优值
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from System_data.system_config import case, Y_bus_matrix,PV_bus_define
from gurobipy import Model, GRB
import numpy as np
import os

# 设置随机种子
np.random.seed(2)

# 读取测试集数据
TEST_DATA = np.load("Data_generation/dataset.npy")
N_TEST_SAMPLES = TEST_DATA.shape[0]
print(f"已加载测试集: {N_TEST_SAMPLES} 条样本")



# 获取电网拓扑数据
SYSTEM_DATA = case()
branch = np.array(SYSTEM_DATA['branch'])
bus = np.array(SYSTEM_DATA['bus'])
PV_bus, PV_capacity = PV_bus_define()
R_ij_matrix, X_ij_matrix, r_x_ratio, branch_max = Y_bus_matrix()
R_vector = branch[:, 2] * r_x_ratio  # 调整后的电阻值，单位 p.u.，长度为 n_branches
# --- 自适应推导网络维度 ---
n_bus = bus.shape[0]           # 节点总数 (例如 33)
n_branches = branch.shape[0]   # 支路总数 (辐射网中通常为 n_bus - 1)
n_pv = len(PV_bus)             # PV 数量 (例如 5)
# 系统参数
V_MAX, V_MIN = 1.05, 0.95
# 测试集格式: [active_load(33), reactive_load(33), pv_power(5)]
ACTIVE_LOAD_ALL = TEST_DATA[:, :n_bus].T  # (33, N_TEST_SAMPLES)
REACTIVE_LOAD_ALL = TEST_DATA[:, n_bus:2*n_bus].T  # (33, N_TEST_SAMPLES)
PV_P_POWER_ALL = TEST_DATA[:, 2*n_bus:]  # (N_TEST_SAMPLES, 5)
print("="*60)
print("利用测试集均值生成基准运行点 (用于 Inner_convex 启动)")
print("="*60)

# 计算测试集均值
ACTIVE_LOAD = np.mean(ACTIVE_LOAD_ALL, axis=1)  # (33,)
REACTIVE_LOAD = np.mean(REACTIVE_LOAD_ALL, axis=1)  # (33,)
PV_p_power = np.mean(PV_P_POWER_ALL, axis=0)  # (5,)

print(f"\n测试集统计信息:")
print(f"  样本数: {N_TEST_SAMPLES}")
print(f"  有功负荷均值范围: [{ACTIVE_LOAD.min():.4f}, {ACTIVE_LOAD.max():.4f}] p.u.")
print(f"  无功负荷均值范围: [{REACTIVE_LOAD.min():.4f}, {REACTIVE_LOAD.max():.4f}] p.u.")
print(f"  PV有功均值: {PV_p_power} p.u.")

model = Model('')
# PV和WT的发电功率
PV_q_power = model.addMVar((len(PV_bus)),lb = -GRB.INFINITY,vtype = GRB.CONTINUOUS, name = 'PV_q_power')
# 弃光选项：设置为True启用弃光，False则强制使用全部光伏功率
ENABLE_CURTAILMENT = True  # 修改此参数即可控制是否允许弃光

# 创建PV实际有功出力决策变量
PV_p_actual = model.addMVar((len(PV_bus)), lb=0, vtype=GRB.CONTINUOUS, name='PV_p_actual')

if ENABLE_CURTAILMENT:
    # 启用弃光：实际出力不超过最大可用功率
    model.addConstrs(PV_p_actual[i] <= PV_p_power[i] for i in range(len(PV_bus)))
else:
    # 禁用弃光：实际出力必须等于最大可用功率
    model.addConstrs(PV_p_actual[i] == PV_p_power[i] for i in range(len(PV_bus)))

# 视在功率约束（使用实际出力）
model.addConstrs(PV_q_power[i]**2 + PV_p_actual[i]**2 <= PV_capacity**2 for i in range(len(PV_bus)))
# 向上级电网购买的有功和无功以及其范围约束对应的对偶变量
MG_Power = model.addMVar((2),lb = -GRB.INFINITY,ub = GRB.INFINITY,vtype = GRB.CONTINUOUS, name = 'MG_power')  
"""
定义电网约束
"""
# 支路的电流，有功和无功功率
branch_current = model.addMVar((branch.shape[0]),lb = 0,ub = branch_max**2,vtype = GRB.CONTINUOUS, name=f'branch_current')
P_ij = model.addMVar((branch.shape[0]),lb = -GRB.INFINITY,ub = GRB.INFINITY,vtype = GRB.CONTINUOUS, name=f'P_ij')
Q_ij = model.addMVar((branch.shape[0]),lb = -GRB.INFINITY,ub = GRB.INFINITY,vtype = GRB.CONTINUOUS, name=f'Q_ij')
power_loss = model.addMVar((branch.shape[0]),lb = -GRB.INFINITY,ub = GRB.INFINITY,vtype = GRB.CONTINUOUS, name=f'power_loss') # 记录下有功网损
"""节点的电压以及节点注入功率"""
Bus_V = model.addMVar((bus.shape[0]),lb = V_MIN**2,ub = V_MAX**2,vtype = GRB.CONTINUOUS, name = 'Bus_V')  # 节点电压的平方
Bus_P_inj = model.addMVar((bus.shape[0]),lb = -GRB.INFINITY,ub = GRB.INFINITY,vtype = GRB.CONTINUOUS, name = 'Bus_P_inj')
Bus_Q_inj = model.addMVar((bus.shape[0]),lb = -GRB.INFINITY,ub = GRB.INFINITY,vtype = GRB.CONTINUOUS, name = 'Bus_Q_inj')
"""平衡节点 以1节点为参考节点"""
model.addConstr(Bus_V[0] == 1)
"二阶锥松弛的distflow潮流传输约束"
"""支路功率和节点注入电压的约束"""
for i in range(bus.shape[0]):
    if i == 65:
        x = 0
    # 父节点和子节点的索引
    upper_index,lower_index = np.int16(branch[np.where(branch[:, 1] == i+1)[0],0]-1),np.int16(branch[np.where(branch[:, 0] == i+1)[0],1]-1)
    # 流入支路和流出支路的索引
    upper_branch_indices,lower_branch_indices = [],[]
    for j in upper_index: # 节点i作为支路末端节点
        branch_indices = np.where((branch[:,0].astype(int) == j+1) & (branch[:,1].astype(int) == i+1))[0] # 检索支路索引
        if len(branch_indices) != 0:
            upper_branch_indices.append(branch_indices[0])  # 记录上游支路
    for j in lower_index: # 节点i作为支路末端节点
        branch_indices = np.where((branch[:,0].astype(int) == i+1) & (branch[:,1].astype(int) == j+1))[0] # 检索支路索引
        if len(branch_indices) != 0:
            lower_branch_indices.append(branch_indices[0])  # 记录下游支路            
    "计算有功功率"
    upper_flow_power,lower_flow_power = 0,0
    if len(upper_index) != 0: # 有上游节点的情况，主要是排除首节点        
        upper_flow_power += P_ij[upper_branch_indices].sum(axis = 0)
        for j in range(len(upper_index)):
            upper_flow_power -= R_ij_matrix[i,upper_index[j]] * branch_current[upper_branch_indices[j]] # 减去线路损耗
            model.addConstr(power_loss[upper_branch_indices[j]] == R_ij_matrix[i,upper_index[j]] * branch_current[upper_branch_indices[j]]) # 记录有功网损
    if len(lower_index) != 0: # 有下游节点的情况，主要是排除末端节点
        lower_flow_power += P_ij[lower_branch_indices].sum(axis = 0)
    model.addConstr(Bus_P_inj[i] == lower_flow_power - upper_flow_power) # 节点注入功率等于流出的 - 上一节点流出的 - 线路的损耗
    "计算无功功率"
    upper_flow_power,lower_flow_power = 0,0
    if len(upper_index) != 0: # 有上游节点的情况，主要是排除首节点        
        upper_flow_power += Q_ij[upper_branch_indices].sum(axis = 0)
        for j in range(len(upper_index)):
            upper_flow_power -= X_ij_matrix[i,upper_index[j]] * branch_current[upper_branch_indices[j]] # 减去线路损耗
    if len(lower_index) != 0: # 有下游节点的情况，主要是排除末端节点
        lower_flow_power += Q_ij[lower_branch_indices].sum(axis = 0)
    model.addConstr(Bus_Q_inj[i] == lower_flow_power - upper_flow_power) # 节点注入功率等于流出的 - 上一节点流出的 - 线路的损耗
"""电压降落约束，以及二阶锥松弛约束"""
for i in range(bus.shape[0]):
    # 这里电压降落只用考虑子节点就好
    lower_index = np.int16(branch[np.where(branch[:, 0] == i+1)[0],1]-1)
    # 流出支路的索引
    lower_branch_indices = []
    for j in lower_index: # 节点i作为支路末端节点
        branch_indices = np.where((branch[:,0].astype(int) == i+1) & (branch[:,1].astype(int) == j+1))[0] # 检索支路索引
        if len(branch_indices) != 0:
            lower_branch_indices.append(branch_indices[0])  # 记录下游支路   
    for j in range(len(lower_index)):
        # print(R_ij_matrix[i,lower_index[j]])
        drop_part1 = 2*(R_ij_matrix[i,lower_index[j]]*P_ij[lower_branch_indices[j]] + X_ij_matrix[i,lower_index[j]]*Q_ij[lower_branch_indices[j]])
        drop_part2 = (R_ij_matrix[i,lower_index[j]]**2 + X_ij_matrix[i,lower_index[j]]**2)*branch_current[lower_branch_indices[j]] # 网损项
        model.addConstr(Bus_V[lower_index[j]] == Bus_V[i] - drop_part1 + drop_part2)
        """支路电流、电压和有功潮流、无功潮流的平方的二阶锥松弛"""
        model.addConstr(branch_current[lower_branch_indices[j]] * Bus_V[i] == (P_ij[lower_branch_indices[j]]**2 + Q_ij[lower_branch_indices[j]]**2))
"节点注入功率平衡约束"
count_PV =  0
for i in range(bus.shape[0]):
    if i == 0:
        model.addConstr(Bus_P_inj[i] == MG_Power[0] - ACTIVE_LOAD[i])
        model.addConstr(Bus_Q_inj[i] == MG_Power[1] - REACTIVE_LOAD[i])
    elif np.isin(i+1,PV_bus): # PV连接节点
        model.addConstr(Bus_P_inj[i] == PV_p_actual[count_PV] - ACTIVE_LOAD[i])
        model.addConstr(Bus_Q_inj[i] == PV_q_power[count_PV] - REACTIVE_LOAD[i])
        count_PV += 1
    else:
        model.addConstr(Bus_P_inj[i] == - ACTIVE_LOAD[i])
        model.addConstr(Bus_Q_inj[i] == - REACTIVE_LOAD[i])

# 目标函数：最小化从主网购电功率
obj = power_loss.sum()

# 添加弃光惩罚项（仅在启用弃光时，避免不必要的弃光）
if ENABLE_CURTAILMENT:
    CURTAILMENT_PENALTY = 1  # 弃光惩罚系数，可根据需要调整
    curtailment_amount = sum(PV_p_power[i] - PV_p_actual[i] for i in range(len(PV_bus)))
    obj = obj + CURTAILMENT_PENALTY * curtailment_amount

model.setObjective(obj, GRB.MINIMIZE)

# 求解模型
print("\n正在求解均值场景模型...")
model.optimize()

if model.status == GRB.OPTIMAL:
    print(f"✓ 模型求解成功！目标值 (网损): {model.ObjVal:.6f} p.u.")
    
    # 提取基准运行点（仅保存关键变量用于启动）
    baseline_point = {
        'Bus_V': Bus_V.X.copy(),           # 节点电压平方 (33,)
        'P_ij': P_ij.X.copy(),             # 支路有功 (n_branch,)
        'Q_ij': Q_ij.X.copy(),             # 支路无功 (n_branch,)
        'branch_current': branch_current.X.copy(),  # 支路电流 (n_branch,)
    }
    
    # 保存到 System_data 文件夹
    save_dir = Path(__file__).parent.parent / "System_data"
    save_path = save_dir / "baseline_operating_point.npz"
    np.savez(save_path, **baseline_point)
    print(f"\n✓ 基准运行点已保存至: {save_path}")
    
    # 打印关键信息
    print("\n基准运行点详细信息:")
    print("-" * 50)
    print(f"节点电压:")
    print(f"  范围: [{np.sqrt(Bus_V.X.min()):.4f}, {np.sqrt(Bus_V.X.max()):.4f}] p.u.")
    print(f"  均值: {np.sqrt(Bus_V.X.mean()):.4f} p.u.")
    
    print(f"\n支路有功潮流:")
    print(f"  范围: [{P_ij.X.min():.4f}, {P_ij.X.max():.4f}] p.u.")
    print(f"  均值: {P_ij.X.mean():.4f} p.u.")
    
    print(f"\n支路无功潮流:")
    print(f"  范围: [{Q_ij.X.min():.4f}, {Q_ij.X.max():.4f}] p.u.")
    print(f"  均值: {Q_ij.X.mean():.4f} p.u.")
    
    print(f"\nPV无功出力:")
    for i, (bus_idx, q_val) in enumerate(zip(PV_bus, PV_q_power.X)):
        print(f"  Bus {bus_idx}: Q_PV = {q_val:.4f} p.u.")
    
    print(f"\n主网交换功率:")
    print(f"  P_MG = {MG_Power.X[0]:.4f} p.u.")
    print(f"  Q_MG = {MG_Power.X[1]:.4f} p.u.")
    
    # 验证松弛间隙
    slack_gap = np.zeros((branch.shape[0]))
    for i in range(bus.shape[0]):
        lower_index = np.int16(branch[np.where(branch[:, 0] == i+1)[0], 1]-1)
        lower_branch_indices = []
        for j in lower_index:
            branch_indices = np.where((branch[:, 0].astype(int) == i+1) & (branch[:, 1].astype(int) == j+1))[0]
            if len(branch_indices) != 0:
                lower_branch_indices.append(branch_indices[0])
        for j in range(len(lower_index)):
            slack_gap[lower_branch_indices[j]] = (branch_current.X[lower_branch_indices[j]] * Bus_V.X[i] - 
                                                 (P_ij.X[lower_branch_indices[j]]**2 + Q_ij.X[lower_branch_indices[j]]**2))
    print(f"\n二阶锥松弛间隙总和: {np.sum(slack_gap):.2e}")
    print("=" * 60)

    # 提供加载示例
    print("\n使用说明:")
    print("在 Inner_convex 模型中加载基准运行点:")
    print("-" * 50)
    print("""
    # 加载基准运行点
    baseline = np.load('System_data/baseline_operating_point.npz')

    # 获取启动点
    Bus_V_start = baseline['Bus_V']           # 节点电压平方 (33,)
    P_ij_start = baseline['P_ij']             # 支路有功 (n_branch,)
    Q_ij_start = baseline['Q_ij']             # 支路无功 (n_branch,)
    branch_current_start = baseline['branch_current']  # 支路电流 (n_branch,)

    # 设置变量初始值（示例）
    # for i in range(n_bus):
    #     Bus_V[i].Start = Bus_V_start[i]
    # for i in range(n_branch):
    #     P_ij[i].Start = P_ij_start[i]
    #     Q_ij[i].Start = Q_ij_start[i]
    #     branch_current[i].Start = branch_current_start[i]
    """)
    print("-" * 50)

    
else:
    print(f"✗ 模型求解失败，状态码: {model.status}")

