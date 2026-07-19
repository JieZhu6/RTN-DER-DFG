"""12 时段统一建模的精确 DistFlow/IPOPT 基线。

本文件同时提供 SOCP 基线复用的模型构建与结果提取函数。两个模型只有
支路功率关系不同：精确模型使用等式，SOCP 使用旋转锥松弛。
"""

import sys
import time
from pathlib import Path

import numpy as np
from pyomo.environ import ConcreteModel, Constraint, Objective, Param, RangeSet, Var, minimize, value
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from System_data.system_config import PV_bus_define, Y_bus_matrix, case


N_PERIODS = 12
N_VALIDATE_SAMPLES = None
ENABLE_CURTAILMENT = True
CURTAILMENT_PENALTY = 1.0
IPOPT_PATH = "D:/anaconda/envs/py3.10/Library/bin/ipopt.exe"
TIME_LIMIT = 600
V_MAX, V_MIN = 1.05, 0.95
FEASIBILITY_TOL = 1e-4
EXACTNESS_TOL = 1e-4


def load_system():
    system_data = case()
    branch = np.asarray(system_data["branch"])
    bus = np.asarray(system_data["bus"])
    pv_bus, pv_capacity = PV_bus_define()
    R_matrix, X_matrix, _, branch_max = Y_bus_matrix()

    n_bus = bus.shape[0]
    n_branches = branch.shape[0]
    n_pv = len(pv_bus)
    branch_from = np.asarray(branch[:, 0], dtype=int) - 1
    branch_to = np.asarray(branch[:, 1], dtype=int) - 1
    incoming_branch = {}
    outgoing_branches = {i: [] for i in range(n_bus)}
    for b, (from_bus, to_bus) in enumerate(zip(branch_from, branch_to)):
        incoming_branch[int(to_bus)] = b
        outgoing_branches[int(from_bus)].append(b)

    return {
        "branch": branch,
        "bus": bus,
        "pv_bus": pv_bus,
        "pv_capacity": pv_capacity,
        "R_matrix": R_matrix,
        "X_matrix": X_matrix,
        "branch_max": branch_max,
        "n_bus": n_bus,
        "n_branches": n_branches,
        "n_pv": n_pv,
        "branch_from": branch_from,
        "branch_to": branch_to,
        "incoming_branch": incoming_branch,
        "outgoing_branches": outgoing_branches,
        "bus_to_pv": {int(bus_idx) - 1: i for i, bus_idx in enumerate(pv_bus)},
    }


def load_multi_period_validation_data(system):
    data_path = ROOT_DIR / "Data_generation" / "dataset_split.npz"
    if not data_path.exists():
        raise FileNotFoundError(
            f"缺少 {data_path}，请先生成 12 时段最优数据并运行 normalize_dataset.py"
        )
    data = np.load(data_path)
    X_raw = data["X_val_raw"]
    features_per_period = 2 * system["n_bus"] + system["n_pv"]
    expected_dim = N_PERIODS * features_per_period
    if X_raw.ndim != 2 or X_raw.shape[1] != expected_dim:
        raise ValueError(
            f"12 时段验证输入期望形状 (*, {expected_dim})，实际 {X_raw.shape}"
        )

    n_samples = len(X_raw)
    if N_VALIDATE_SAMPLES is not None:
        n_samples = min(n_samples, N_VALIDATE_SAMPLES)
    return X_raw[:n_samples].reshape(n_samples, N_PERIODS, features_per_period)


def build_multi_period_model(active_load, reactive_load, pv_available, system, relaxation):
    """构建一个包含全部 12 时段的 DistFlow 模型。

    relaxation 为 ``exact`` 或 ``socp``。各时段没有耦合约束，但共享一个模型
    和一个总目标，由求解器一次统一求解。
    """
    n_bus = system["n_bus"]
    n_branches = system["n_branches"]
    n_pv = system["n_pv"]
    expected = ((N_PERIODS, n_bus), (N_PERIODS, n_bus), (N_PERIODS, n_pv))
    actual = (active_load.shape, reactive_load.shape, pv_available.shape)
    if actual != expected:
        raise ValueError(f"12 时段输入形状期望 {expected}，实际 {actual}")
    if relaxation not in {"exact", "socp"}:
        raise ValueError(f"未知模型类型: {relaxation}")

    branch_from = system["branch_from"]
    branch_to = system["branch_to"]
    incoming_branch = system["incoming_branch"]
    outgoing_branches = system["outgoing_branches"]
    bus_to_pv = system["bus_to_pv"]
    R_matrix = system["R_matrix"]
    X_matrix = system["X_matrix"]
    pv_capacity = system["pv_capacity"]
    branch_max = system["branch_max"]

    model = ConcreteModel()
    model.TIMES = RangeSet(0, N_PERIODS - 1)
    model.BUSES = RangeSet(0, n_bus - 1)
    model.BRANCHES = RangeSet(0, n_branches - 1)
    model.PV_UNITS = RangeSet(0, n_pv - 1)
    model.MG_COMPONENTS = RangeSet(0, 1)

    model.PV_q_power = Var(model.TIMES, model.PV_UNITS, initialize=0.0)
    if ENABLE_CURTAILMENT:
        model.PV_p_actual = Var(
            model.TIMES,
            model.PV_UNITS,
            bounds=(0, None),
            initialize=lambda m, t, i: pv_available[t, i],
        )
    else:
        model.PV_p_actual = Param(
            model.TIMES,
            model.PV_UNITS,
            initialize=lambda m, t, i: pv_available[t, i],
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

    model.ref_voltage = Constraint(
        model.TIMES, rule=lambda m, t: m.Bus_V[t, 0] == 1.0
    )

    def pv_apparent_power_rule(m, t, i):
        return m.PV_p_actual[t, i] ** 2 + m.PV_q_power[t, i] ** 2 <= pv_capacity**2

    model.pv_apparent_power = Constraint(
        model.TIMES, model.PV_UNITS, rule=pv_apparent_power_rule
    )
    if ENABLE_CURTAILMENT:
        model.pv_upper = Constraint(
            model.TIMES,
            model.PV_UNITS,
            rule=lambda m, t, i: m.PV_p_actual[t, i] <= pv_available[t, i],
        )

    def bus_p_balance_rule(m, t, i):
        inflow = 0.0
        if i in incoming_branch:
            b_in = incoming_branch[i]
            parent = int(branch_from[b_in])
            inflow = (
                m.P_ij[t, b_in]
                - R_matrix[i, parent] * m.branch_current[t, b_in]
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
                - X_matrix[i, parent] * m.branch_current[t, b_in]
            )
        outflow = sum(m.Q_ij[t, b] for b in outgoing_branches[i])
        return m.Bus_Q_inj[t, i] == outflow - inflow

    model.bus_q_balance = Constraint(
        model.TIMES, model.BUSES, rule=bus_q_balance_rule
    )

    def power_loss_rule(m, t, b):
        i, j = int(branch_from[b]), int(branch_to[b])
        return m.power_loss[t, b] == R_matrix[i, j] * m.branch_current[t, b]

    model.power_loss_constr = Constraint(
        model.TIMES, model.BRANCHES, rule=power_loss_rule
    )

    def voltage_drop_rule(m, t, b):
        i, j = int(branch_from[b]), int(branch_to[b])
        r_ij, x_ij = R_matrix[i, j], X_matrix[i, j]
        return m.Bus_V[t, j] == (
            m.Bus_V[t, i]
            - 2 * (r_ij * m.P_ij[t, b] + x_ij * m.Q_ij[t, b])
            + (r_ij**2 + x_ij**2) * m.branch_current[t, b]
        )

    model.voltage_drop = Constraint(
        model.TIMES, model.BRANCHES, rule=voltage_drop_rule
    )

    if relaxation == "exact":
        def branch_power_rule(m, t, b):
            i = int(branch_from[b])
            return (
                m.branch_current[t, b] * m.Bus_V[t, i]
                == m.P_ij[t, b] ** 2 + m.Q_ij[t, b] ** 2
            )
    else:
        def branch_power_rule(m, t, b):
            # ||(2P, 2Q, l-v)||_2 <= l+v 等价于 P²+Q² <= l*v。
            i = int(branch_from[b])
            return (
                (2 * m.P_ij[t, b]) ** 2
                + (2 * m.Q_ij[t, b]) ** 2
                + (m.branch_current[t, b] - m.Bus_V[t, i]) ** 2
                <= (m.branch_current[t, b] + m.Bus_V[t, i]) ** 2
            )

    model.branch_power_relation = Constraint(
        model.TIMES, model.BRANCHES, rule=branch_power_rule
    )

    def bus_p_injection_rule(m, t, i):
        if i == 0:
            return m.Bus_P_inj[t, i] == m.MG_Power[t, 0] - active_load[t, i]
        if i in bus_to_pv:
            return (
                m.Bus_P_inj[t, i]
                == m.PV_p_actual[t, bus_to_pv[i]] - active_load[t, i]
            )
        return m.Bus_P_inj[t, i] == -active_load[t, i]

    model.bus_p_injection = Constraint(
        model.TIMES, model.BUSES, rule=bus_p_injection_rule
    )

    def bus_q_injection_rule(m, t, i):
        if i == 0:
            return m.Bus_Q_inj[t, i] == m.MG_Power[t, 1] - reactive_load[t, i]
        if i in bus_to_pv:
            return (
                m.Bus_Q_inj[t, i]
                == m.PV_q_power[t, bus_to_pv[i]] - reactive_load[t, i]
            )
        return m.Bus_Q_inj[t, i] == -reactive_load[t, i]

    model.bus_q_injection = Constraint(
        model.TIMES, model.BUSES, rule=bus_q_injection_rule
    )

    total_loss = sum(
        model.power_loss[t, b] for t in model.TIMES for b in model.BRANCHES
    )
    total_curtailment = sum(
        pv_available[t, i] - model.PV_p_actual[t, i]
        for t in model.TIMES
        for i in model.PV_UNITS
    )
    objective_expr = total_loss
    if ENABLE_CURTAILMENT:
        objective_expr += CURTAILMENT_PENALTY * total_curtailment
    model.objective = Objective(expr=objective_expr, sense=minimize)
    model._relaxation_type = relaxation
    return model


def solve_model(model, relaxation, solver_factory=None):
    if solver_factory is None:
        if relaxation == "exact":
            solver = SolverFactory("ipopt", executable=IPOPT_PATH)
            solver.options["print_level"] = 0
            solver.options["max_cpu_time"] = TIME_LIMIT
        else:
            solver = SolverFactory("gurobi")
            solver.options["OutputFlag"] = 0
            solver.options["TimeLimit"] = TIME_LIMIT
    else:
        solver = solver_factory(relaxation)

    start = time.perf_counter()
    try:
        results = solver.solve(model, tee=False)
    except Exception as exc:
        return False, time.perf_counter() - start, str(exc)
    solve_time = time.perf_counter() - start
    success = (
        results.solver.status == SolverStatus.ok
        and results.solver.termination_condition
        in {
            TerminationCondition.optimal,
            TerminationCondition.locallyOptimal,
            TerminationCondition.feasible,
        }
    )
    return success, solve_time, str(results.solver.termination_condition)


def extract_solution(model, pv_available, system):
    n_bus = system["n_bus"]
    n_branches = system["n_branches"]
    n_pv = system["n_pv"]
    branch_from = system["branch_from"]
    branch_to = system["branch_to"]
    R_matrix = system["R_matrix"]
    pv_capacity = system["pv_capacity"]
    branch_max = system["branch_max"]

    pv_p = np.array(
        [[value(model.PV_p_actual[t, i]) for i in range(n_pv)] for t in range(N_PERIODS)]
    )
    pv_q = np.array(
        [[value(model.PV_q_power[t, i]) for i in range(n_pv)] for t in range(N_PERIODS)]
    )
    V_sq = np.array(
        [[value(model.Bus_V[t, i]) for i in range(n_bus)] for t in range(N_PERIODS)]
    )
    l_sq = np.array(
        [
            [value(model.branch_current[t, b]) for b in range(n_branches)]
            for t in range(N_PERIODS)
        ]
    )
    P_ij = np.array(
        [[value(model.P_ij[t, b]) for b in range(n_branches)] for t in range(N_PERIODS)]
    )
    Q_ij = np.array(
        [[value(model.Q_ij[t, b]) for b in range(n_branches)] for t in range(N_PERIODS)]
    )

    resistance = np.array(
        [R_matrix[int(branch_from[b]), int(branch_to[b])] for b in range(n_branches)]
    )
    period_objective = (l_sq * resistance).sum(axis=1)
    if ENABLE_CURTAILMENT:
        period_objective += CURTAILMENT_PENALTY * (pv_available - pv_p).sum(axis=1)
    from_voltage = V_sq[:, branch_from]
    cone_slack = l_sq * from_voltage - P_ij**2 - Q_ij**2

    pv_capacity_ok = np.all(pv_p**2 + pv_q**2 <= pv_capacity**2 + FEASIBILITY_TOL, axis=1)
    pv_upper_ok = np.all(pv_p <= pv_available + FEASIBILITY_TOL, axis=1)
    voltage_ok = np.all(
        (V_sq >= V_MIN**2 - FEASIBILITY_TOL)
        & (V_sq <= V_MAX**2 + FEASIBILITY_TOL),
        axis=1,
    )
    current_ok = np.all(
        (l_sq >= -FEASIBILITY_TOL) & (l_sq <= branch_max**2 + FEASIBILITY_TOL),
        axis=1,
    )
    branch_relation_ok = np.min(cone_slack, axis=1) >= -FEASIBILITY_TOL
    period_feasible = (
        pv_capacity_ok & pv_upper_ok & voltage_ok & current_ok & branch_relation_ok
    )
    period_exact_feasible = np.max(np.abs(cone_slack), axis=1) <= EXACTNESS_TOL

    return {
        "pv_p": pv_p,
        "pv_q": pv_q,
        "bus_voltage_sq": V_sq,
        "branch_current_sq": l_sq,
        "branch_p": P_ij,
        "branch_q": Q_ij,
        "period_objective_values": period_objective,
        "objective_value": period_objective.sum(),
        "period_feasible": period_feasible,
        "feasible": period_feasible.all(),
        "cone_slack": cone_slack,
        "period_exact_feasible": period_exact_feasible,
        "exact_feasible": period_exact_feasible.all(),
    }


def run_validation(relaxation="exact"):
    system = load_system()
    X_temporal = load_multi_period_validation_data(system)
    n_samples = len(X_temporal)
    n_bus, n_pv = system["n_bus"], system["n_pv"]
    method_name = "IPOPT_Exact_T12" if relaxation == "exact" else "Gurobi_SOCP_T12"
    output_name = (
        "results_distflow_ipopt_T12.npz"
        if relaxation == "exact"
        else "results_distflow_socp_T12.npz"
    )

    print("=" * 70)
    print(f"{method_name}: {n_samples} 个 12 时段验证样本统一建模求解")
    print(f"系统: {n_bus} 节点, {system['n_branches']} 支路, {n_pv} 个 PV")
    print("=" * 70)

    solve_times = np.zeros(n_samples)
    solver_success = np.zeros(n_samples, dtype=bool)
    termination = np.empty(n_samples, dtype=object)
    period_feasible = np.zeros((n_samples, N_PERIODS), dtype=bool)
    period_exact_feasible = np.zeros((n_samples, N_PERIODS), dtype=bool)
    period_objectives = np.full((n_samples, N_PERIODS), np.nan)
    objective_values = np.full(n_samples, np.nan)
    pv_p = np.full((n_samples, N_PERIODS, n_pv), np.nan)
    pv_q = np.full_like(pv_p, np.nan)
    V_sq = np.full((n_samples, N_PERIODS, n_bus), np.nan)
    l_sq = np.full((n_samples, N_PERIODS, system["n_branches"]), np.nan)
    cone_slack_max = np.full((n_samples, N_PERIODS), np.nan)

    for sample_idx in range(n_samples):
        x = X_temporal[sample_idx]
        active_load = x[:, :n_bus]
        reactive_load = x[:, n_bus : 2 * n_bus]
        pv_available = x[:, 2 * n_bus :]
        model = build_multi_period_model(
            active_load, reactive_load, pv_available, system, relaxation
        )
        success, solve_time, condition = solve_model(model, relaxation)
        solve_times[sample_idx] = solve_time
        solver_success[sample_idx] = success
        termination[sample_idx] = condition

        if success:
            solution = extract_solution(model, pv_available, system)
            pv_p[sample_idx] = solution["pv_p"]
            pv_q[sample_idx] = solution["pv_q"]
            V_sq[sample_idx] = solution["bus_voltage_sq"]
            l_sq[sample_idx] = solution["branch_current_sq"]
            period_feasible[sample_idx] = solution["period_feasible"]
            period_exact_feasible[sample_idx] = solution["period_exact_feasible"]
            period_objectives[sample_idx] = solution["period_objective_values"]
            objective_values[sample_idx] = solution["objective_value"]
            cone_slack_max[sample_idx] = np.max(
                np.abs(solution["cone_slack"]), axis=1
            )

        print(
            f"[{sample_idx + 1}/{n_samples}] {condition}, "
            f"time={solve_time:.2f}s"
        )

    sample_feasible = solver_success & period_feasible.all(axis=1)
    sample_exact_feasible = solver_success & period_exact_feasible.all(axis=1)
    result_dir = ROOT_DIR / "Test_result"
    result_dir.mkdir(exist_ok=True)
    output_path = result_dir / output_name
    np.savez(
        output_path,
        method=method_name,
        n_periods=N_PERIODS,
        solver_success=solver_success,
        termination=termination.astype(str),
        decision_times=solve_times,
        pv_p=pv_p,
        pv_q=pv_q,
        bus_voltage_sq=V_sq,
        branch_current_sq=l_sq,
        period_feasible=period_feasible,
        feasible=sample_feasible,
        period_exact_feasible=period_exact_feasible,
        exact_feasible=sample_exact_feasible,
        period_objective_values=period_objectives,
        objective_values=objective_values,
        cone_slack_max=cone_slack_max,
    )

    solved_objectives = objective_values[solver_success]
    print("\n" + "=" * 70)
    print(f"求解成功率: {solver_success.mean() * 100:.2f}%")
    print(f"12 时段约束可行率: {sample_feasible.mean() * 100:.2f}%")
    print(f"12 时段精确等式满足率: {sample_exact_feasible.mean() * 100:.2f}%")
    print(f"平均求解时间: {solve_times.mean():.4f} s")
    if len(solved_objectives):
        print(f"平均 12 时段总目标: {np.nanmean(solved_objectives):.6f}")
    print(f"结果已保存到: {output_path}")
    return output_path


def main():
    run_validation(relaxation="exact")


if __name__ == "__main__":
    main()
