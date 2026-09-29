"""
生成监督学习数据集
X: 负荷和光伏数据 (来自 dataset.npy)
Y: 最优 PV 有功和无功功率 (通过 IPOPT 求解 Distflow 模型获得)

使用方法:
    python generate_optimal_dataset.py
    
输出:
    - dataset_supervised.npz: 包含 X (输入特征) 和 Y (最优标签)
    - 可选: dataset_supervised.csv 用于查看
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from System_data.system_config import case, Y_bus_matrix, PV_bus_define
import numpy as np
import pandas as pd
import time
import os
import warnings
warnings.filterwarnings('ignore')

# Pyomo 相关导入
from pyomo.environ import *
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition

# ==============================================================================
# 0. 配置参数
# ==============================================================================
# IPOPT 求解器路径（请根据实际情况修改）
IPOPT_PATH = 'D:/anaconda/envs/py3.10/Library/bin/ipopt.exe'

# 是否启用弃光
ENABLE_CURTAILMENT = True

# 求解时间限制（秒）
TIME_LIMIT = 60

# 是否保存求解失败的样本
SAVE_FAILED_SAMPLES = True

# ==============================================================================
# 1. 加载系统和测试数据
# ==============================================================================
print("="*70)
print("加载系统和测试数据")
print("="*70)

# 获取路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SYSTEM_DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "System_data")

# 加载测试数据
TEST_DATA = np.load(os.path.join(SCRIPT_DIR, "dataset.npy"))
N_SAMPLES = TEST_DATA.shape[0]
print(f"已加载测试集: {N_SAMPLES} 条样本")

# 获取系统基础拓扑和配置
SYSTEM_DATA = case()
branch = np.array(SYSTEM_DATA['branch'])
bus = np.array(SYSTEM_DATA['bus'])
PV_bus, PV_capacity = PV_bus_define()
R_ij_matrix, X_ij_matrix, r_x_ratio, branch_max = Y_bus_matrix()

n_bus = bus.shape[0]
n_branches = branch.shape[0]
n_pv = len(PV_bus)

print(f"系统配置: {n_bus} 节点, {n_branches} 支路, {n_pv} 个 PV 单元")
print(f"PV 节点: {PV_bus}")
print(f"PV 装机容量: {PV_capacity} p.u.")

V_MAX, V_MIN = 1.05, 0.95
S_BASE_MVA = 10  # 基准功率 MVA

# 提取所有样本数据
ACTIVE_LOAD_ALL = TEST_DATA[:, :n_bus].T  # 形状: (n_bus, N_SAMPLES)
REACTIVE_LOAD_ALL = TEST_DATA[:, n_bus:2*n_bus].T  # 形状: (n_bus, N_SAMPLES)
PV_P_POWER_ALL = TEST_DATA[:, 2*n_bus:]  # 形状: (N_SAMPLES, n_pv)

# ==============================================================================
# 2. 定义求解函数
# ==============================================================================
def solve_single_sample(sample_idx, active_load, reactive_load, pv_p_power):
    """
    对单个样本求解最优调度问题
    
    参数:
        sample_idx: 样本索引
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
        obj = summation(model.power_loss)
        if ENABLE_CURTAILMENT:
            curtailment = sum(pv_p_power[i] - m.PV_p_actual[i] for i in m.PV_UNITS)
            CURTAILMENT_PENALTY = 1
            obj += CURTAILMENT_PENALTY * curtailment
        return obj
    model.objective = Objective(rule=objective_rule, sense=minimize)
    
    # ==================== 求解 ====================
    try:
        solver = SolverFactory('ipopt', executable=IPOPT_PATH)
    except Exception as e:
        print(f"  错误: 无法加载 IPOPT 求解器: {e}")
        return False, None, None, None, 0
    
    # IPOPT 选项
    # solver.options['tol'] = 1e-4
    # solver.options['max_iter'] = 200
    solver.options['print_level'] = 0  # 关闭详细输出
    # solver.options['mu_strategy'] = 'adaptive'
    # solver.options['linear_solver'] = 'mumps'
    # solver.options['accept_every_trial_step'] = 'no'
    
    start_time = time.time()
    try:
        results = solver.solve(model, tee=False)
        solve_time = time.time() - start_time
    except Exception as e:
        print(f"  求解异常: {e}")
        return False, None, None, None, time.time() - start_time
    
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
        pv_q_opt = np.array([value(model.PV_q_power[i]) for i in model.PV_UNITS])
        if ENABLE_CURTAILMENT:
            pv_p_opt = np.array([value(model.PV_p_actual[i]) for i in model.PV_UNITS])
        else:
            pv_p_opt = pv_p_power.copy()
        obj_value = value(model.objective)
        branch_current = np.array([value(model.branch_current[i]) for i in model.BRANCHES])
        bus_voltage = np.array([value(model.Bus_V[i]) for i in model.BUSES])
        curtailment = pv_p_power - pv_p_opt if ENABLE_CURTAILMENT else np.zeros(n_pv)
        return True, pv_p_opt, pv_q_opt, obj_value, solve_time, curtailment, branch_current, bus_voltage
    else:
        return False, None, None, None, solve_time, None, None, None


# ==============================================================================
# 3. 主循环：遍历所有样本并求解
# ==============================================================================
print("\n" + "="*70)
print("开始求解所有样本")
print("="*70)

# 存储结果
X_list = []  # 输入特征
Y_list = []  # 最优标签 (PV_P, PV_Q)
success_count = 0
failed_indices = []
solve_times = []
obj_values = []
curtailment_list = []      # 弃风弃光
branch_current_list = []   # 支路电流
bus_voltage_list = []      # 节点电压

# 设置求解样本数量（可用于测试）
# N_SAMPLES_TO_SOLVE = 100  # 测试时只解前10个样本
N_SAMPLES_TO_SOLVE = N_SAMPLES  # 生产环境解全部样本

start_time_total = time.time()

for idx in range(N_SAMPLES_TO_SOLVE):
    print(f"\n[{idx+1}/{N_SAMPLES_TO_SOLVE}] 求解样本 {idx}...", end=" ")
    
    # 获取当前样本数据
    active_load = ACTIVE_LOAD_ALL[:, idx]
    reactive_load = REACTIVE_LOAD_ALL[:, idx]
    pv_p_power = PV_P_POWER_ALL[idx, :]
    
    # 求解
    success, pv_p_opt, pv_q_opt, obj_value, solve_time, curtailment, branch_current, bus_voltage = solve_single_sample(
        idx, active_load, reactive_load, pv_p_power
    )
    
    solve_times.append(solve_time)
    
    if success:
        print(f"✓ 成功 (时间: {solve_time:.2f}s, 目标: {obj_value:.4f})")
        success_count += 1
        
        # 构建输入特征 X
        x = np.concatenate([active_load, reactive_load, pv_p_power])
        X_list.append(x)
        
        # 构建标签 Y (PV 有功 + 无功)
        y = np.concatenate([pv_p_opt/pv_p_power, pv_q_opt])

        Y_list.append(y)
        
        obj_values.append(obj_value)
        curtailment_list.append(curtailment)
        branch_current_list.append(branch_current)
        bus_voltage_list.append(bus_voltage)
    else:
        print(f"✗ 失败 (时间: {solve_time:.2f}s)")
        failed_indices.append(idx)

end_time_total = time.time()
total_time = end_time_total - start_time_total

# ==============================================================================
# 4. 保存结果
# ==============================================================================
print("\n" + "="*70)
print("求解统计")
print("="*70)
print(f"总样本数: {N_SAMPLES_TO_SOLVE}")
print(f"成功求解: {success_count}")
print(f"求解失败: {len(failed_indices)}")
print(f"成功率: {success_count/N_SAMPLES_TO_SOLVE*100:.2f}%")
print(f"总耗时: {total_time:.2f} 秒")
print(f"平均求解时间: {np.mean(solve_times):.3f} 秒")
if obj_values:
    print(f"平均目标函数值: {np.mean(obj_values):.6f}")

if success_count > 0:
    # 转换为 numpy 数组
    X_array = np.array(X_list)
    Y_array = np.array(Y_list)
    
    print(f"\n数据集维度:")
    print(f"  X (输入特征): {X_array.shape}")
    print(f"  Y (最优标签): {Y_array.shape}")
    
    # 保存原始数据集
    output_path = os.path.join(SCRIPT_DIR, "dataset_supervised.npz")
    np.savez(output_path, X=X_array, Y=Y_array)
    print(f"\n原始数据集已保存到: {output_path}")
    
    # 打印数据统计信息
    print("\n" + "="*70)
    print("标签 Y 统计信息")
    print("="*70)
    print(f"PV 有功功率范围: [{np.min(Y_array[:, :n_pv]):.4f}, {np.max(Y_array[:, :n_pv]):.4f}]")
    print(f"PV 无功功率范围: [{np.min(Y_array[:, n_pv:]):.4f}, {np.max(Y_array[:, n_pv:]):.4f}]")


print("\n" + "="*70)
print("数据生成完成!")
print("="*70)
print("\n使用示例:")
print("  # 加载原始数据")
print("  data = np.load('Data_generation/dataset_supervised.npz')")
print("  X = data['X']  # 输入特征")
print("  Y = data['Y']  # 最优标签")
print("\n  # 下一步: 运行 normalize_dataset.py 进行标准化")
print("="*70)

# ==============================================================================
# 5. 可视化结果
# ==============================================================================
if success_count > 0 and curtailment_list:
    import matplotlib.pyplot as plt
    
    C = np.array(curtailment_list)          # (N, n_pv)
    I = np.sqrt(np.array(branch_current_list))  # (N, n_branches)
    V = np.sqrt(np.array(bus_voltage_list))     # (N, n_bus)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 弃风弃光
    axes[0].boxplot([C[:, i] for i in range(n_pv)], labels=[f'PV{i+1}' for i in range(n_pv)])
    axes[0].set_title('Curtailment (p.u.)')
    axes[0].set_ylabel('Curtailment')
    
    # 最大支路电流
    max_I = I.max(axis=1)
    axes[1].hist(max_I, bins=min(20, success_count), edgecolor='k')
    axes[1].axvline(branch_max, color='r', linestyle='--', label=f'Limit={branch_max:.3f}')
    axes[1].set_title(f'Max Branch Current (p.u.)\nmean={max_I.mean():.4f}')
    axes[1].set_xlabel('Max Current')
    axes[1].legend()
    
    # 节点电压
    axes[2].boxplot([V[:, i] for i in range(n_bus)], labels=[f'{i+1}' for i in range(n_bus)])
    axes[2].axhline(V_MAX, color='r', linestyle='--', linewidth=0.8)
    axes[2].axhline(V_MIN, color='r', linestyle='--', linewidth=0.8)
    axes[2].set_title('Bus Voltage (p.u.)')
    axes[2].set_ylabel('Voltage')
    axes[2].tick_params(axis='x', rotation=90, labelsize=6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, 'dataset_visualization.png'), dpi=150)
    plt.show()
    print(f"\n可视化结果已保存到: {os.path.join(SCRIPT_DIR, 'dataset_visualization.png')}")

