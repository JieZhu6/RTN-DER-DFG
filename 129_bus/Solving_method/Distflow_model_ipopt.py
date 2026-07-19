# 使用 IPOPT 求解的标准 Distflow 模型（非凸精确模型）
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from System_data.system_config import case, Y_bus_matrix, PV_bus_define
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import warnings
from NN_Model.powerflow_env import check_feasibility, run_powerflow_numpy_single
# Pyomo 相关导入
from pyomo.environ import *
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
ENABLE_CURTAILMENT = True  # 是否启用弃光选项
np.random.seed(2)
plt.rcParams.update({'font.size': 16})
plt.rc('font', family='Times New Roman')

# 首先获取系统基础拓扑和配置
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
print(f"System loaded adaptively: {n_bus} Buses, {n_branches} Branches, {n_pv} PVs")
V_MAX,V_MIN= 1.05,0.95
# 系统基准功率（在模型和pandapower验证中都需要使用）
S_BASE_MVA = 10  # 基准功率 MVA

# 配置参数
N_VALIDATE_SAMPLES = None  # 设为 None 求解全部，或设为整数如 20 求解前20个
# N_VALIDATE_SAMPLES = 300 
# 读取测试集数据
data = np.load("Data_generation/dataset_split.npz")
TEST_DATA = data['X_val_raw']
N_TEST_TOTAL = TEST_DATA.shape[0]

if N_VALIDATE_SAMPLES is not None and N_VALIDATE_SAMPLES > 0:
    N_TEST_SAMPLES = min(N_VALIDATE_SAMPLES, N_TEST_TOTAL)
    TEST_DATA = TEST_DATA[:N_TEST_SAMPLES]
    print(f"已加载测试集: {N_TEST_SAMPLES}/{N_TEST_TOTAL} 条样本")
else:
    N_TEST_SAMPLES = N_TEST_TOTAL
    print(f"已加载测试集: {N_TEST_SAMPLES} 条样本")


print("="*60)
print("开始使用 IPOPT 求解标准 Distflow 模型...")
print("="*60)

def optimize(pv_p_power, active_load, reactive_load):
    """
    对单个样本求解最优调度问题
    
    参数:
        active_load: 有功负荷 (n_bus,)
        reactive_load: 无功负荷 (n_bus,)
        pv_p_power: PV 最大有功出力 (n_pv,)
    
    返回:
        success: 是否求解成功
        pv_p_opt: 最优 PV 有功功率 (n_pv,)
        pv_q_opt: 最优 PV 无功功率 (n_pv,)
        obj_value: 目标函数值
        solve_time: 求解时间
    """
    
    # 创建 Pyomo 模型
    model = ConcreteModel()
    
    # 集合定义
    model.BUSES = RangeSet(0, n_bus - 1)
    model.BRANCHES = RangeSet(0, n_branches - 1)
    model.PV_UNITS = RangeSet(0, n_pv - 1)
    
    # PV 节点映射
    pv_bus_indices = {i: PV_bus[i] - 1 for i in range(n_pv)}
    bus_to_pv = {PV_bus[i] - 1: i for i in range(n_pv)}
    
    # 支路映射
    branch_from = {i: int(branch[i, 0]) - 1 for i in range(n_branches)}
    branch_to = {i: int(branch[i, 1]) - 1 for i in range(n_branches)}
    
    # 父子节点关系
    def get_parent_child_relations():
        parent_of = {i: [] for i in range(n_bus)}
        children_of = {i: [] for i in range(n_bus)}
        branch_to_parent = {}
        branch_from_parent = {}
        
        for b in range(n_branches):
            f, t = branch_from[b], branch_to[b]
            parent_of[t].append(f)
            children_of[f].append(t)
            branch_to_parent[t] = b
            branch_from_parent[f] = branch_from_parent.get(f, []) + [b]
        
        return parent_of, children_of, branch_to_parent, branch_from_parent
    
    parent_of, children_of, branch_to_parent, branch_from_parent = get_parent_child_relations()
    
    # ==================== 决策变量 ====================
    model.PV_q_power = Var(model.PV_UNITS, initialize=0.0)
    
    if ENABLE_CURTAILMENT:
        model.PV_p_actual = Var(model.PV_UNITS, bounds=(0, None), 
                                initialize=lambda m, i: pv_p_power[i])
    else:
        model.PV_p_actual = Param(model.PV_UNITS, initialize=lambda m, i: pv_p_power[i])
    
    model.MG_Power = Var(RangeSet(0, 1), initialize=0.0)
    model.branch_current = Var(model.BRANCHES, bounds=(0, branch_max**2), initialize=0.01)
    model.P_ij = Var(model.BRANCHES, initialize=0.0)
    model.Q_ij = Var(model.BRANCHES, initialize=0.0)
    model.power_loss = Var(model.BRANCHES, initialize=0.0)
    model.Bus_V = Var(model.BUSES, bounds=(V_MIN**2, V_MAX**2), initialize=1.0)
    model.Bus_P_inj = Var(model.BUSES, initialize=0.0)
    model.Bus_Q_inj = Var(model.BUSES, initialize=0.0)
    
    # ==================== 约束定义 ====================
    
    # 1. 平衡节点电压固定
    def ref_voltage_rule(m):
        return m.Bus_V[0] == 1.0
    model.ref_voltage = Constraint(rule=ref_voltage_rule)
    
    # 2. PV 视在功率约束
    def pv_apparent_power_rule(m, i):
        if ENABLE_CURTAILMENT:
            return m.PV_q_power[i]**2 + m.PV_p_actual[i]**2 <= PV_capacity**2
        else:
            return m.PV_q_power[i]**2 + pv_p_power[i]**2 <= PV_capacity**2
    model.pv_apparent_power = Constraint(model.PV_UNITS, rule=pv_apparent_power_rule)
    
    # 3. 弃光上限约束
    if ENABLE_CURTAILMENT:
        def pv_curtailment_upper_rule(m, i):
            return m.PV_p_actual[i] <= pv_p_power[i]
        model.pv_curtailment_upper = Constraint(model.PV_UNITS, rule=pv_curtailment_upper_rule)
    
    # 4. 节点有功功率平衡
    def bus_p_balance_rule(m, i):
        inflow = 0
        if i in branch_to_parent:
            b_in = branch_to_parent[i]
            inflow += m.P_ij[b_in]
            parent = parent_of[i][0] if parent_of[i] else 0
            r_ij = R_ij_matrix[i, parent] if parent < n_bus else 0
            inflow -= r_ij * m.branch_current[b_in]
        
        outflow = 0
        if i in branch_from_parent:
            for b_out in branch_from_parent[i]:
                outflow += m.P_ij[b_out]
        
        return m.Bus_P_inj[i] == outflow - inflow
    model.bus_p_balance = Constraint(model.BUSES, rule=bus_p_balance_rule)
    
    # 5. 节点无功功率平衡
    def bus_q_balance_rule(m, i):
        inflow = 0
        if i in branch_to_parent:
            b_in = branch_to_parent[i]
            inflow += m.Q_ij[b_in]
            parent = parent_of[i][0] if parent_of[i] else 0
            x_ij = X_ij_matrix[i, parent] if parent < n_bus else 0
            inflow -= x_ij * m.branch_current[b_in]
        
        outflow = 0
        if i in branch_from_parent:
            for b_out in branch_from_parent[i]:
                outflow += m.Q_ij[b_out]
        
        return m.Bus_Q_inj[i] == outflow - inflow
    model.bus_q_balance = Constraint(model.BUSES, rule=bus_q_balance_rule)
    
    # 6. 网损记录
    def power_loss_rule(m, i):
        if i in branch_to_parent:
            b_in = branch_to_parent[i]
            parent = parent_of[i][0] if parent_of[i] else 0
            r_ij = R_ij_matrix[i, parent] if parent < n_bus else 0
            return m.power_loss[b_in] == r_ij * m.branch_current[b_in]
        return Constraint.Skip
    model.power_loss_constr = Constraint(model.BUSES, rule=power_loss_rule)
    
    # 7. 电压降落约束
    model.voltage_drop = ConstraintList()
    for i in range(n_bus):
        if i in branch_from_parent:
            for b_out in branch_from_parent[i]:
                j = branch_to[b_out]
                r_ij = R_ij_matrix[i, j]
                x_ij = X_ij_matrix[i, j]
                drop_part1 = 2 * (r_ij * model.P_ij[b_out] + x_ij * model.Q_ij[b_out])
                drop_part2 = (r_ij**2 + x_ij**2) * model.branch_current[b_out]
                model.voltage_drop.add(model.Bus_V[j] == model.Bus_V[i] - drop_part1 + drop_part2)
    
    # 8. 标准 DistFlow 约束（非凸等式）
    def distflow_exact_rule(m, b):
        i = branch_from[b]
        return m.branch_current[b] * m.Bus_V[i] == m.P_ij[b]**2 + m.Q_ij[b]**2
    model.distflow_exact = Constraint(model.BRANCHES, rule=distflow_exact_rule)
    
    # 9. 节点注入功率平衡
    def bus_p_injection_rule(m, i):
        if i == 0:
            return m.Bus_P_inj[i] == m.MG_Power[0] - active_load[i]
        elif i in bus_to_pv:
            pv_idx = bus_to_pv[i]
            if ENABLE_CURTAILMENT:
                return m.Bus_P_inj[i] == m.PV_p_actual[pv_idx] - active_load[i]
            else:
                return m.Bus_P_inj[i] == pv_p_power[pv_idx] - active_load[i]
        else:
            return m.Bus_P_inj[i] == -active_load[i]
    model.bus_p_injection = Constraint(model.BUSES, rule=bus_p_injection_rule)
    
    def bus_q_injection_rule(m, i):
        if i == 0:
            return m.Bus_Q_inj[i] == m.MG_Power[1] - reactive_load[i]
        elif i in bus_to_pv:
            pv_idx = bus_to_pv[i]
            return m.Bus_Q_inj[i] == m.PV_q_power[pv_idx] - reactive_load[i]
        else:
            return m.Bus_Q_inj[i] == -reactive_load[i]
    model.bus_q_injection = Constraint(model.BUSES, rule=bus_q_injection_rule)
    
    # ==================== 目标函数 ====================
    def objective_rule(m):
        obj = summation(model.power_loss)  # 加个小二次方惩罚项，确定无功解的最优性，防止同样的x对于多个q解都最优的情况出现，导致无法拟合。
        if ENABLE_CURTAILMENT:
            curtailment = sum(pv_p_power[i] - m.PV_p_actual[i] for i in m.PV_UNITS)
            CURTAILMENT_PENALTY = 1
            obj += CURTAILMENT_PENALTY * curtailment
        return obj
    model.objective = Objective(rule=objective_rule, sense=minimize)


    # 配置 IPOPT 求解器
    solver = SolverFactory('ipopt', executable='D:/anaconda/envs/py3.10/Library/bin/ipopt.exe')

    # IPOPT 求解选项
    solver.options['print_level'] = 0         # 0: 无输出, 5: 详细输出
    # solver.options['tol'] = 1e-6              # 收敛容差
    # solver.options['max_iter'] = 300          # 最大迭代次数
    # solver.options['mu_strategy'] = 'adaptive'  # 障碍参数策略
    # solver.options['linear_solver'] = 'mumps'   # 线性求解器
    
    # 求解 (tee=False 屏蔽控制台输出)
    results = solver.solve(model, tee=False)
    # 检查求解状态：只信任 solver 返回的 termination condition
    solution_available = (
        results.solver.status == SolverStatus.ok and
        results.solver.termination_condition in [
            TerminationCondition.optimal,
            TerminationCondition.locallyOptimal,
            TerminationCondition.feasible
        ]
    )
    
    if solution_available:
        return model, results

# 从测试集提取数据
ACTIVE_LOAD_ALL = TEST_DATA[:, :n_bus].T  # 形状: (33, N_TEST_SAMPLES)
REACTIVE_LOAD_ALL = TEST_DATA[:, n_bus:2*n_bus].T  # 形状: (33, N_TEST_SAMPLES)
PV_P_POWER_ALL = TEST_DATA[:, 2*n_bus:]  # 形状: (N_TEST_SAMPLES)

decision_times = []  # NN 决策时间
feasible = []
objective_values = []    # 目标函数值

# 确保 PV_capacity 是数组
if np.isscalar(PV_capacity):
    PV_capacity_array = np.full(n_pv, PV_capacity, dtype=np.float32)
else:
    PV_capacity_array = np.array(PV_capacity, dtype=np.float32)

for i in range(N_TEST_SAMPLES):
    print(f"正在求解测试样本 {i+1}/{N_TEST_SAMPLES}...")
    time_start = time.time()
    model, results = optimize(PV_P_POWER_ALL[i, :], ACTIVE_LOAD_ALL[:, i], REACTIVE_LOAD_ALL[:, i])
    time_end = time.time()
    solve_time = time_end - time_start

    PV_q_values = np.array([value(model.PV_q_power[i]) for i in model.PV_UNITS])
    PV_p_actual_values = np.array([value(model.PV_p_actual[i]) for i in model.PV_UNITS])
    
    # 提取 IPOPT 模型内部求解得到的电压和电流，与 numpy 求解器结果比对
    V_model = np.array([value(model.Bus_V[j]) for j in model.BUSES])
    l_model = np.array([value(model.branch_current[b]) for b in model.BRANCHES])
    
    feasible.append(check_feasibility(
        V_model, l_model, PV_p_actual_values, PV_q_values, PV_P_POWER_ALL[i, :], PV_capacity
    ))
    decision_times.append(solve_time)
    # 目标函数
    objective_values.append(value(model.objective))

# 结果保存
results = {
    'method': 'IPOPT',
    'decision_times': decision_times,
    'objective_values': objective_values,
    'feasible': feasible,
    'avg_decision_time': np.mean(decision_times),
    'feasibility_rate': np.mean(feasible),
    'avg_objective_value': np.mean(objective_values)
}

np.savez("Test_result/results_distflow_ipopt.npz", **results)
print(f"\n结果已保存到: Test_result/results_distflow_ipopt.npz")

# 打印统计信息
print("\n" + "="*60)
print("IPOPT 求解统计")
print("="*60)
print(f"测试样本数: {N_TEST_SAMPLES}")
print(f"可行决策: {np.sum(feasible)}/{N_TEST_SAMPLES} ({100*np.mean(feasible):.2f}%)")
print(f"平均求解时间: {np.mean(decision_times):.4f} s")
print(f"总求解时间: {np.sum(decision_times):.4f} s")
print(f"平均目标函数值: {np.mean(objective_values):.4f}")






