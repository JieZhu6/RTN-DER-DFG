"""
生成 12 时段联合优化的监督学习数据集。

dataset.npy 中连续 12 行被视为一个多时段样本。每组数据在同一个
Pyomo/DistFlow 模型中由 IPOPT 一次求解，目标函数为 12 个时段的
总网损与总弃光惩罚之和。

输出 dataset_supervised.npz：
    X: (样本数, 12 * (2 * n_bus + n_pv))，按时段优先展平
    Y: (样本数, 12 * (2 * n_pv))，每时段依次为 PV 有功比例和无功
"""

import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from pyomo.environ import (
    ConcreteModel,
    Constraint,
    Objective,
    Param,
    RangeSet,
    Var,
    minimize,
    value,
)
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition

sys.path.insert(0, str(Path(__file__).parent.parent))
from System_data.system_config import PV_bus_define, Y_bus_matrix, case

warnings.filterwarnings("ignore")


# ==============================================================================
# 0. 配置参数
# ==============================================================================
IPOPT_PATH = "D:/anaconda/envs/py3.10/Library/bin/ipopt.exe"
N_PERIODS = 12
ENABLE_CURTAILMENT = True
CURTAILMENT_PENALTY = 1.0
TIME_LIMIT = 600
SAVE_FAILED_SAMPLES = True


# ==============================================================================
# 1. 加载系统与数据
# ==============================================================================
print("=" * 70)
print(f"加载系统和数据）")
print("=" * 70)

SCRIPT_DIR = Path(__file__).resolve().parent
TEST_DATA = np.load(SCRIPT_DIR / "dataset.npy")

SYSTEM_DATA = case()
branch = np.asarray(SYSTEM_DATA["branch"])
bus = np.asarray(SYSTEM_DATA["bus"])
PV_bus, PV_capacity = PV_bus_define()
R_ij_matrix, X_ij_matrix, _, branch_max = Y_bus_matrix()

n_bus = bus.shape[0]
n_branches = branch.shape[0]
n_pv = len(PV_bus)
n_features_per_period = 2 * n_bus + n_pv

if TEST_DATA.ndim != 2 or TEST_DATA.shape[1] != n_features_per_period:
    raise ValueError(
        "dataset.npy 维度与当前系统不一致："
        f"期望 (*, {n_features_per_period})，实际 {TEST_DATA.shape}"
    )

n_raw_samples = TEST_DATA.shape[0]
n_multi_period_samples = n_raw_samples // N_PERIODS
n_dropped_samples = n_raw_samples % N_PERIODS
if n_multi_period_samples == 0:
    raise ValueError(f"dataset.npy 至少需要 {N_PERIODS} 行数据")

# 只使用完整的 12 时段窗口；每个窗口中的数据保持原始行顺序。
grouped_data = TEST_DATA[: n_multi_period_samples * N_PERIODS].reshape(
    n_multi_period_samples, N_PERIODS, n_features_per_period
)
ACTIVE_LOAD_ALL = grouped_data[:, :, :n_bus]
REACTIVE_LOAD_ALL = grouped_data[:, :, n_bus : 2 * n_bus]
PV_P_POWER_ALL = grouped_data[:, :, 2 * n_bus :]

print(f"原始数据: {n_raw_samples} 个单时段样本")
print(f"联合数据: {n_multi_period_samples} 个 {N_PERIODS} 时段样本")
if n_dropped_samples:
    print(f"末尾 {n_dropped_samples} 个不足 {N_PERIODS} 时段的样本不参与求解")
print(f"系统配置: {n_bus} 节点, {n_branches} 支路, {n_pv} 个 PV 单元")

V_MAX, V_MIN = 1.05, 0.95


# ==============================================================================
# 2. 预计算不随样本变化的网络拓扑
# ==============================================================================
branch_from = np.asarray(branch[:, 0], dtype=int) - 1
branch_to = np.asarray(branch[:, 1], dtype=int) - 1
bus_to_pv = {int(pv_bus) - 1: i for i, pv_bus in enumerate(PV_bus)}

incoming_branch = {}
outgoing_branches = {i: [] for i in range(n_bus)}
for b, (from_bus, to_bus) in enumerate(zip(branch_from, branch_to)):
    incoming_branch[int(to_bus)] = b
    outgoing_branches[int(from_bus)].append(b)


# ==============================================================================
# 3. 12 时段联合优化模型
# ==============================================================================
def solve_multi_period_sample(sample_idx, active_load, reactive_load, pv_p_power):
    """一次求解一个 12 时段样本。

    输入形状分别为 (12, n_bus)、(12, n_bus) 和 (12, n_pv)；
    返回的最优调度和网络状态均保留时段维度。
    """
    expected_shapes = (
        (N_PERIODS, n_bus),
        (N_PERIODS, n_bus),
        (N_PERIODS, n_pv),
    )
    actual_shapes = (active_load.shape, reactive_load.shape, pv_p_power.shape)
    if actual_shapes != expected_shapes:
        raise ValueError(
            f"样本 {sample_idx} 的输入形状错误：期望 {expected_shapes}，实际 {actual_shapes}"
        )

    model = ConcreteModel()
    model.TIMES = RangeSet(0, N_PERIODS - 1)
    model.BUSES = RangeSet(0, n_bus - 1)
    model.BRANCHES = RangeSet(0, n_branches - 1)
    model.PV_UNITS = RangeSet(0, n_pv - 1)
    model.MG_COMPONENTS = RangeSet(0, 1)

    # 决策变量的第一个索引均为时段。
    model.PV_q_power = Var(model.TIMES, model.PV_UNITS, initialize=0.0)
    if ENABLE_CURTAILMENT:
        model.PV_p_actual = Var(
            model.TIMES,
            model.PV_UNITS,
            bounds=(0, None),
            initialize=lambda m, t, i: pv_p_power[t, i],
        )
    else:
        model.PV_p_actual = Param(
            model.TIMES,
            model.PV_UNITS,
            initialize=lambda m, t, i: pv_p_power[t, i],
        )

    model.MG_Power = Var(model.TIMES, model.MG_COMPONENTS, initialize=0.0)
    model.branch_current = Var(
        model.TIMES,
        model.BRANCHES,
        bounds=(0, branch_max**2),
        initialize=0.01,
    )
    model.P_ij = Var(model.TIMES, model.BRANCHES, initialize=0.0)
    model.Q_ij = Var(model.TIMES, model.BRANCHES, initialize=0.0)
    model.power_loss = Var(model.TIMES, model.BRANCHES, initialize=0.0)
    model.Bus_V = Var(
        model.TIMES,
        model.BUSES,
        bounds=(V_MIN**2, V_MAX**2),
        initialize=1.0,
    )
    model.Bus_P_inj = Var(model.TIMES, model.BUSES, initialize=0.0)
    model.Bus_Q_inj = Var(model.TIMES, model.BUSES, initialize=0.0)

    def ref_voltage_rule(m, t):
        return m.Bus_V[t, 0] == 1.0

    model.ref_voltage = Constraint(model.TIMES, rule=ref_voltage_rule)

    def pv_apparent_power_rule(m, t, i):
        return m.PV_q_power[t, i] ** 2 + m.PV_p_actual[t, i] ** 2 <= PV_capacity**2

    model.pv_apparent_power = Constraint(
        model.TIMES, model.PV_UNITS, rule=pv_apparent_power_rule
    )

    if ENABLE_CURTAILMENT:
        def pv_curtailment_upper_rule(m, t, i):
            return m.PV_p_actual[t, i] <= pv_p_power[t, i]

        model.pv_curtailment_upper = Constraint(
            model.TIMES, model.PV_UNITS, rule=pv_curtailment_upper_rule
        )

    def bus_p_balance_rule(m, t, i):
        inflow = 0.0
        if i in incoming_branch:
            b_in = incoming_branch[i]
            parent = int(branch_from[b_in])
            inflow = (
                m.P_ij[t, b_in]
                - R_ij_matrix[i, parent] * m.branch_current[t, b_in]
            )
        outflow = sum(m.P_ij[t, b] for b in outgoing_branches[i])
        return m.Bus_P_inj[t, i] == outflow - inflow

    model.bus_p_balance = Constraint(
        model.TIMES, model.BUSES, rule=bus_p_balance_rule
    )

    def bus_q_balance_rule(m, t, i):
        inflow = 0.0
        if i in incoming_branch:
            b_in = incoming_branch[i]
            parent = int(branch_from[b_in])
            inflow = (
                m.Q_ij[t, b_in]
                - X_ij_matrix[i, parent] * m.branch_current[t, b_in]
            )
        outflow = sum(m.Q_ij[t, b] for b in outgoing_branches[i])
        return m.Bus_Q_inj[t, i] == outflow - inflow

    model.bus_q_balance = Constraint(
        model.TIMES, model.BUSES, rule=bus_q_balance_rule
    )

    def power_loss_rule(m, t, b):
        i, j = int(branch_from[b]), int(branch_to[b])
        return m.power_loss[t, b] == R_ij_matrix[i, j] * m.branch_current[t, b]

    model.power_loss_constr = Constraint(
        model.TIMES, model.BRANCHES, rule=power_loss_rule
    )

    def voltage_drop_rule(m, t, b):
        i, j = int(branch_from[b]), int(branch_to[b])
        r_ij = R_ij_matrix[i, j]
        x_ij = X_ij_matrix[i, j]
        return m.Bus_V[t, j] == (
            m.Bus_V[t, i]
            - 2 * (r_ij * m.P_ij[t, b] + x_ij * m.Q_ij[t, b])
            + (r_ij**2 + x_ij**2) * m.branch_current[t, b]
        )

    model.voltage_drop = Constraint(
        model.TIMES, model.BRANCHES, rule=voltage_drop_rule
    )

    def distflow_exact_rule(m, t, b):
        i = int(branch_from[b])
        return (
            m.branch_current[t, b] * m.Bus_V[t, i]
            == m.P_ij[t, b] ** 2 + m.Q_ij[t, b] ** 2
        )

    model.distflow_exact = Constraint(
        model.TIMES, model.BRANCHES, rule=distflow_exact_rule
    )

    def bus_p_injection_rule(m, t, i):
        if i == 0:
            return m.Bus_P_inj[t, i] == m.MG_Power[t, 0] - active_load[t, i]
        if i in bus_to_pv:
            pv_idx = bus_to_pv[i]
            return (
                m.Bus_P_inj[t, i]
                == m.PV_p_actual[t, pv_idx] - active_load[t, i]
            )
        return m.Bus_P_inj[t, i] == -active_load[t, i]

    model.bus_p_injection = Constraint(
        model.TIMES, model.BUSES, rule=bus_p_injection_rule
    )

    def bus_q_injection_rule(m, t, i):
        if i == 0:
            return m.Bus_Q_inj[t, i] == m.MG_Power[t, 1] - reactive_load[t, i]
        if i in bus_to_pv:
            pv_idx = bus_to_pv[i]
            return (
                m.Bus_Q_inj[t, i]
                == m.PV_q_power[t, pv_idx] - reactive_load[t, i]
            )
        return m.Bus_Q_inj[t, i] == -reactive_load[t, i]

    model.bus_q_injection = Constraint(
        model.TIMES, model.BUSES, rule=bus_q_injection_rule
    )

    def objective_rule(m):
        total_loss = sum(
            m.power_loss[t, b] for t in m.TIMES for b in m.BRANCHES
        )
        if not ENABLE_CURTAILMENT:
            return total_loss
        total_curtailment = sum(
            pv_p_power[t, i] - m.PV_p_actual[t, i]
            for t in m.TIMES
            for i in m.PV_UNITS
        )
        return total_loss + CURTAILMENT_PENALTY * total_curtailment

    model.objective = Objective(rule=objective_rule, sense=minimize)

    try:
        solver = SolverFactory("ipopt", executable=IPOPT_PATH)
        solver.options["print_level"] = 0
        solver.options["max_cpu_time"] = TIME_LIMIT
    except Exception as exc:
        print(f"  错误: 无法加载 IPOPT 求解器: {exc}")
        return False, None, None, None, 0.0, None, None, None

    start_time = time.time()
    try:
        results = solver.solve(model, tee=False)
        solve_time = time.time() - start_time
    except Exception as exc:
        print(f"  求解异常: {exc}")
        return False, None, None, None, time.time() - start_time, None, None, None

    solution_available = (
        results.solver.status == SolverStatus.ok
        and results.solver.termination_condition
        in {
            TerminationCondition.optimal,
            TerminationCondition.locallyOptimal,
            TerminationCondition.feasible,
        }
    )
    if not solution_available:
        return False, None, None, None, solve_time, None, None, None

    pv_p_opt = np.array(
        [[value(model.PV_p_actual[t, i]) for i in model.PV_UNITS] for t in model.TIMES]
    )
    pv_q_opt = np.array(
        [[value(model.PV_q_power[t, i]) for i in model.PV_UNITS] for t in model.TIMES]
    )
    branch_current_opt = np.array(
        [[value(model.branch_current[t, b]) for b in model.BRANCHES] for t in model.TIMES]
    )
    bus_voltage_opt = np.array(
        [[value(model.Bus_V[t, i]) for i in model.BUSES] for t in model.TIMES]
    )
    curtailment = pv_p_power - pv_p_opt

    return (
        True,
        pv_p_opt,
        pv_q_opt,
        value(model.objective),
        solve_time,
        curtailment,
        branch_current_opt,
        bus_voltage_opt,
    )


# ==============================================================================
# 4. 遍历所有样本
# ==============================================================================
print("\n" + "=" * 70)
print(f"开始求解")
print("=" * 70)

X_list = []
Y_list = []
successful_sample_indices = []
failed_indices = []
solve_times = []
obj_values = []
curtailment_list = []
branch_current_list = []
bus_voltage_list = []

N_SAMPLES_TO_SOLVE = n_multi_period_samples
start_time_total = time.time()

for idx in range(N_SAMPLES_TO_SOLVE):
    print(f"\n[{idx + 1}/{N_SAMPLES_TO_SOLVE}] 求解时段组 {idx}...", end=" ")

    active_load = ACTIVE_LOAD_ALL[idx]
    reactive_load = REACTIVE_LOAD_ALL[idx]
    pv_p_power = PV_P_POWER_ALL[idx]

    result = solve_multi_period_sample(
        idx, active_load, reactive_load, pv_p_power
    )
    (
        success,
        pv_p_opt,
        pv_q_opt,
        obj_value,
        solve_time,
        curtailment,
        branch_current_opt,
        bus_voltage_opt,
    ) = result
    solve_times.append(solve_time)

    if not success:
        print(f"失败 (时间: {solve_time:.2f}s)")
        failed_indices.append(idx)
        continue

    print(f"成功 (时间: {solve_time:.2f}s, 12 时段总目标: {obj_value:.4f})")
    successful_sample_indices.append(idx)

    # 按 [t0全部特征, t1全部特征, ...] 展平，兼容现有二维学习流程。
    x_temporal = np.concatenate(
        [active_load, reactive_load, pv_p_power], axis=1
    )
    pv_p_ratio = np.divide(
        pv_p_opt,
        pv_p_power,
        out=np.ones_like(pv_p_opt),
        where=pv_p_power > 1e-10,
    )
    y_temporal = np.concatenate([pv_p_ratio, pv_q_opt], axis=1)

    X_list.append(x_temporal.reshape(-1))
    Y_list.append(y_temporal.reshape(-1))
    obj_values.append(obj_value)
    curtailment_list.append(curtailment)
    branch_current_list.append(branch_current_opt)
    bus_voltage_list.append(bus_voltage_opt)

total_time = time.time() - start_time_total
success_count = len(successful_sample_indices)


# ==============================================================================
# 5. 保存与统计
# ==============================================================================
print("\n" + "=" * 70)
print("求解统计")
print("=" * 70)
print(f"总时段组数: {N_SAMPLES_TO_SOLVE}")
print(f"成功求解: {success_count}")
print(f"求解失败: {len(failed_indices)}")
print(f"成功率: {success_count / N_SAMPLES_TO_SOLVE * 100:.2f}%")
print(f"总耗时: {total_time:.2f} 秒")
if solve_times:
    print(f"平均每组求解时间: {np.mean(solve_times):.3f} 秒")
if obj_values:
    print(f"目标值: {np.mean(obj_values):.6f}")

if SAVE_FAILED_SAMPLES:
    np.save(SCRIPT_DIR / "failed_samples.npy", np.asarray(failed_indices, dtype=int))

if success_count > 0:
    X_array = np.asarray(X_list)
    Y_array = np.asarray(Y_list)

    print("\n数据集维度:")
    print(f"  X: {X_array.shape}")
    print(f"  Y: {Y_array.shape}")
    print(
        f"  恢复时段维度: X.reshape(-1, {N_PERIODS}, {n_features_per_period}), "
        f"Y.reshape(-1, {N_PERIODS}, {2 * n_pv})"
    )

    output_path = SCRIPT_DIR / "dataset_supervised.npz"
    np.savez(
        output_path,
        X=X_array,
        Y=Y_array,
        n_periods=N_PERIODS,
        n_bus=n_bus,
        n_pv=n_pv,
        successful_sample_indices=np.asarray(successful_sample_indices, dtype=int),
    )
    print(f"\n数据集已保存到: {output_path}")

    Y_temporal = Y_array.reshape(success_count, N_PERIODS, 2 * n_pv)
    print("\n" + "=" * 70)
    print("标签 Y 统计信息")
    print("=" * 70)
    print(
        "PV 有功比例范围: "
        f"[{Y_temporal[:, :, :n_pv].min():.4f}, "
        f"{Y_temporal[:, :, :n_pv].max():.4f}]"
    )
    print(
        "PV 无功功率范围: "
        f"[{Y_temporal[:, :, n_pv:].min():.4f}, "
        f"{Y_temporal[:, :, n_pv:].max():.4f}]"
    )

print("\n" + "=" * 70)
print("数据生成已经完成!")
print("=" * 70)


# ==============================================================================
# 6. 可视化所有成功样本中的逐时段结果
# ==============================================================================
if success_count > 0 and curtailment_list:
    import matplotlib.pyplot as plt

    C = np.asarray(curtailment_list).reshape(-1, n_pv)
    I = np.sqrt(np.asarray(branch_current_list)).reshape(-1, n_branches)
    V = np.sqrt(np.asarray(bus_voltage_list)).reshape(-1, n_bus)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].boxplot(
        [C[:, i] for i in range(n_pv)],
        tick_labels=[f"PV{i + 1}" for i in range(n_pv)],
    )
    axes[0].set_title("Curtailment (p.u.)")
    axes[0].set_ylabel("Curtailment")

    max_I = I.max(axis=1)
    axes[1].hist(max_I, bins=min(20, len(max_I)), edgecolor="k")
    axes[1].axvline(
        branch_max, color="r", linestyle="--", label=f"Limit={branch_max:.3f}"
    )
    axes[1].set_title(f"Max Branch Current (p.u.)\nmean={max_I.mean():.4f}")
    axes[1].set_xlabel("Max Current")
    axes[1].legend()

    axes[2].boxplot(
        [V[:, i] for i in range(n_bus)],
        tick_labels=[str(i + 1) for i in range(n_bus)],
    )
    axes[2].axhline(V_MAX, color="r", linestyle="--", linewidth=0.8)
    axes[2].axhline(V_MIN, color="r", linestyle="--", linewidth=0.8)
    axes[2].set_title("Bus Voltage (p.u.)")
    axes[2].set_ylabel("Voltage")
    axes[2].tick_params(axis="x", rotation=90, labelsize=4)

    plt.tight_layout()
    figure_path = SCRIPT_DIR / "dataset_visualization.png"
    plt.savefig(figure_path, dpi=150)
    plt.close(fig)
    print(f"可视化结果已保存到: {figure_path}")
