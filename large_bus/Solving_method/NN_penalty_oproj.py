"""12 时段 penalty NN 调度的逐时段 IPOPT 可行域投影。"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
from pyomo.environ import ConcreteModel, Constraint, Objective, RangeSet, Var, minimize, value
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from NN_Model.model import DERDispatchNet
from NN_Model.powerflow_env import check_feasibility, run_powerflow_numpy_single
from Solving_method.NN_direct import compute_objective_single, evaluate_periods
from System_data.system_config import PV_bus_define, Y_bus_matrix, case


N_PERIODS = 12
N_VALIDATE_SAMPLES = None
HIDDEN_DIMS = [256, 256, 256]
LEARNING_RATE = 1e-3
TRAIN_BATCH_SIZE = 1
IPOPT_PATH = "D:/anaconda/envs/py3.10/Library/bin/ipopt.exe"
IPOPT_TIME_LIMIT = 600
V_MAX, V_MIN = 1.05, 0.95


def build_topology(branch, n_bus):
    branch_from = np.asarray(branch[:, 0], dtype=int) - 1
    branch_to = np.asarray(branch[:, 1], dtype=int) - 1
    incoming_branch = {}
    outgoing_branches = {i: [] for i in range(n_bus)}
    for b, (from_bus, to_bus) in enumerate(zip(branch_from, branch_to)):
        incoming_branch[int(to_bus)] = b
        outgoing_branches[int(from_bus)].append(b)
    return branch_from, branch_to, incoming_branch, outgoing_branches


def ipopt_projection_single_period(
    pv_p_raw,
    pv_q_raw,
    p_load,
    q_load,
    p_available,
    pv_bus,
    pv_capacity,
    branch_from,
    branch_to,
    incoming_branch,
    outgoing_branches,
    R_matrix,
    X_matrix,
    branch_max,
):
    """仅对一个时段建立 DistFlow 投影模型。"""
    n_bus = len(p_load)
    n_branches = len(branch_from)
    n_pv = len(pv_bus)
    bus_to_pv = {int(bus_idx) - 1: i for i, bus_idx in enumerate(pv_bus)}

    model = ConcreteModel()
    model.BUSES = RangeSet(0, n_bus - 1)
    model.BRANCHES = RangeSet(0, n_branches - 1)
    model.PV_UNITS = RangeSet(0, n_pv - 1)
    model.MG_COMPONENTS = RangeSet(0, 1)

    model.PV_p_actual = Var(
        model.PV_UNITS,
        bounds=(0, None),
        initialize=lambda m, i: pv_p_raw[i],
    )
    model.PV_q_power = Var(
        model.PV_UNITS, initialize=lambda m, i: pv_q_raw[i]
    )
    model.MG_Power = Var(model.MG_COMPONENTS, initialize=0.0)
    model.branch_current = Var(
        model.BRANCHES, bounds=(0, branch_max**2), initialize=0.01
    )
    model.P_ij = Var(model.BRANCHES, initialize=0.0)
    model.Q_ij = Var(model.BRANCHES, initialize=0.0)
    model.Bus_V = Var(
        model.BUSES, bounds=(V_MIN**2, V_MAX**2), initialize=1.0
    )
    model.Bus_P_inj = Var(model.BUSES, initialize=0.0)
    model.Bus_Q_inj = Var(model.BUSES, initialize=0.0)

    model.ref_voltage = Constraint(expr=model.Bus_V[0] == 1.0)

    def pv_capacity_rule(m, i):
        return m.PV_p_actual[i] ** 2 + m.PV_q_power[i] ** 2 <= pv_capacity**2

    model.pv_capacity = Constraint(model.PV_UNITS, rule=pv_capacity_rule)
    model.pv_upper = Constraint(
        model.PV_UNITS,
        rule=lambda m, i: m.PV_p_actual[i] <= p_available[i],
    )

    def bus_p_balance_rule(m, i):
        inflow = 0.0
        if i in incoming_branch:
            b_in = incoming_branch[i]
            parent = int(branch_from[b_in])
            inflow = m.P_ij[b_in] - R_matrix[i, parent] * m.branch_current[b_in]
        outflow = sum(m.P_ij[b] for b in outgoing_branches[i])
        return m.Bus_P_inj[i] == outflow - inflow

    model.bus_p_balance = Constraint(model.BUSES, rule=bus_p_balance_rule)

    def bus_q_balance_rule(m, i):
        inflow = 0.0
        if i in incoming_branch:
            b_in = incoming_branch[i]
            parent = int(branch_from[b_in])
            inflow = m.Q_ij[b_in] - X_matrix[i, parent] * m.branch_current[b_in]
        outflow = sum(m.Q_ij[b] for b in outgoing_branches[i])
        return m.Bus_Q_inj[i] == outflow - inflow

    model.bus_q_balance = Constraint(model.BUSES, rule=bus_q_balance_rule)

    def voltage_drop_rule(m, b):
        i, j = int(branch_from[b]), int(branch_to[b])
        r_ij, x_ij = R_matrix[i, j], X_matrix[i, j]
        return m.Bus_V[j] == (
            m.Bus_V[i]
            - 2 * (r_ij * m.P_ij[b] + x_ij * m.Q_ij[b])
            + (r_ij**2 + x_ij**2) * m.branch_current[b]
        )

    model.voltage_drop = Constraint(model.BRANCHES, rule=voltage_drop_rule)

    def distflow_rule(m, b):
        i = int(branch_from[b])
        return (
            m.branch_current[b] * m.Bus_V[i]
            == m.P_ij[b] ** 2 + m.Q_ij[b] ** 2
        )

    model.distflow = Constraint(model.BRANCHES, rule=distflow_rule)

    def p_injection_rule(m, i):
        if i == 0:
            return m.Bus_P_inj[i] == m.MG_Power[0] - p_load[i]
        if i in bus_to_pv:
            return m.Bus_P_inj[i] == m.PV_p_actual[bus_to_pv[i]] - p_load[i]
        return m.Bus_P_inj[i] == -p_load[i]

    model.p_injection = Constraint(model.BUSES, rule=p_injection_rule)

    def q_injection_rule(m, i):
        if i == 0:
            return m.Bus_Q_inj[i] == m.MG_Power[1] - q_load[i]
        if i in bus_to_pv:
            return m.Bus_Q_inj[i] == m.PV_q_power[bus_to_pv[i]] - q_load[i]
        return m.Bus_Q_inj[i] == -q_load[i]

    model.q_injection = Constraint(model.BUSES, rule=q_injection_rule)
    model.objective = Objective(
        expr=sum(
            (model.PV_p_actual[i] - pv_p_raw[i]) ** 2
            + (model.PV_q_power[i] - pv_q_raw[i]) ** 2
            for i in model.PV_UNITS
        ),
        sense=minimize,
    )

    solver = SolverFactory("ipopt", executable=IPOPT_PATH)
    solver.options["print_level"] = 0
    solver.options["max_cpu_time"] = IPOPT_TIME_LIMIT
    start = time.perf_counter()
    try:
        results = solver.solve(model, tee=False)
    except Exception as exc:
        solve_time = time.perf_counter() - start
        print(f"单时段 IPOPT 投影异常: {exc}")
        return pv_p_raw.copy(), pv_q_raw.copy(), solve_time, False
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
    if not success:
        return pv_p_raw.copy(), pv_q_raw.copy(), solve_time, False

    pv_p_proj = np.array([value(model.PV_p_actual[i]) for i in model.PV_UNITS])
    pv_q_proj = np.array([value(model.PV_q_power[i]) for i in model.PV_UNITS])
    return pv_p_proj, pv_q_proj, solve_time, True


def main():
    system_data = case()
    branch = np.asarray(system_data["branch"])
    bus = np.asarray(system_data["bus"])
    pv_bus, pv_capacity = PV_bus_define()
    R_matrix, X_matrix, r_x_ratio, branch_max = Y_bus_matrix()
    resistance = branch[:, 2] * r_x_ratio
    n_bus = bus.shape[0]
    n_pv = len(pv_bus)
    features_per_period = 2 * n_bus + n_pv
    outputs_per_period = 2 * n_pv
    topology = build_topology(branch, n_bus)

    data = np.load(ROOT_DIR / "Data_generation" / "dataset_split.npz")
    X_norm = data["X_val_norm"]
    X_raw = data["X_val_raw"]
    Y_raw = data["Y_val_raw"]
    expected_x = N_PERIODS * features_per_period
    expected_y = N_PERIODS * outputs_per_period
    if X_norm.shape[1] != expected_x or X_raw.shape[1] != expected_x:
        raise ValueError(f"12 时段 X 维度应为 {expected_x}，实际为 {X_raw.shape[1]}")
    if Y_raw.shape[1] != expected_y:
        raise ValueError(f"12 时段 Y 维度应为 {expected_y}，实际为 {Y_raw.shape[1]}")

    n_samples = len(X_norm)
    if N_VALIDATE_SAMPLES is not None:
        n_samples = min(n_samples, N_VALIDATE_SAMPLES)
    X_norm = X_norm[:n_samples]
    X_temporal = X_raw[:n_samples].reshape(n_samples, N_PERIODS, features_per_period)
    Y_temporal = Y_raw[:n_samples].reshape(n_samples, N_PERIODS, outputs_per_period)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DERDispatchNet(
        n_bus, n_pv, hidden_dims=HIDDEN_DIMS, n_periods=N_PERIODS
    ).to(device)
    model_path = ROOT_DIR / "NN_parameter" / (
        f"penalty_model_T{N_PERIODS}_{HIDDEN_DIMS}_"
        f"{TRAIN_BATCH_SIZE}_{LEARNING_RATE}.pth"
    )
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    if checkpoint.get("n_periods", 1) != N_PERIODS:
        raise ValueError(f"checkpoint 不是 {N_PERIODS} 时段模型: {model_path}")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("=" * 70)
    print(f"12 时段 penalty NN + 逐时段 IPOPT 投影: {n_samples} 个验证样本")
    print("=" * 70)

    projected_dispatch = []
    period_feasible_before_list = []
    period_feasible_after_list = []
    period_projection_success_list = []
    period_objective_list = []
    period_projection_time_list = []
    nn_times = []

    for i in range(n_samples):
        x_tensor = torch.tensor(X_norm[i : i + 1], dtype=torch.float32, device=device)
        start_nn = time.perf_counter()
        with torch.no_grad():
            y_pred = model(x_tensor).cpu().numpy().reshape(
                N_PERIODS, outputs_per_period
            )
        nn_times.append(time.perf_counter() - start_nn)

        sample_dispatch = np.empty((N_PERIODS, outputs_per_period))
        sample_feasible_before = np.empty(N_PERIODS, dtype=bool)
        sample_feasible_after = np.empty(N_PERIODS, dtype=bool)
        sample_projection_success = np.ones(N_PERIODS, dtype=bool)
        sample_objectives = np.empty(N_PERIODS)
        sample_projection_times = np.zeros(N_PERIODS)

        for t in range(N_PERIODS):
            x_period = X_temporal[i, t]
            p_load = x_period[:n_bus]
            q_load = x_period[n_bus : 2 * n_bus]
            p_available = x_period[2 * n_bus :]
            pv_p_nn = y_pred[t, :n_pv] * p_available
            pv_q_nn = y_pred[t, n_pv:]

            V_nn, l_nn, _, _ = run_powerflow_numpy_single(
                p_load, q_load, pv_p_nn, pv_q_nn, pv_bus
            )
            sample_feasible_before[t] = check_feasibility(
                V_nn, l_nn, pv_p_nn, pv_q_nn, p_available, pv_capacity
            )

            if sample_feasible_before[t]:
                pv_p_proj, pv_q_proj = pv_p_nn, pv_q_nn
            else:
                (
                    pv_p_proj,
                    pv_q_proj,
                    sample_projection_times[t],
                    sample_projection_success[t],
                ) = ipopt_projection_single_period(
                    pv_p_nn,
                    pv_q_nn,
                    p_load,
                    q_load,
                    p_available,
                    pv_bus,
                    pv_capacity,
                    *topology,
                    R_matrix,
                    X_matrix,
                    branch_max,
                )

            V_sq, l_sq, _, _ = run_powerflow_numpy_single(
                p_load, q_load, pv_p_proj, pv_q_proj, pv_bus
            )
            sample_feasible_after[t] = check_feasibility(
                V_sq, l_sq, pv_p_proj, pv_q_proj, p_available, pv_capacity
            )
            sample_objectives[t] = compute_objective_single(
                l_sq, pv_p_proj, p_available, resistance
            )
            sample_dispatch[t] = np.concatenate([pv_p_proj, pv_q_proj])

        projected_dispatch.append(sample_dispatch)
        period_feasible_before_list.append(sample_feasible_before)
        period_feasible_after_list.append(sample_feasible_after)
        period_projection_success_list.append(sample_projection_success)
        period_objective_list.append(sample_objectives)
        period_projection_time_list.append(sample_projection_times)

        if (i + 1) % 10 == 0 or i + 1 == n_samples:
            print(f"已处理 {i + 1}/{n_samples} 个 12 时段样本")

    projected_dispatch = np.asarray(projected_dispatch)
    period_feasible_before = np.asarray(period_feasible_before_list)
    period_feasible_after = np.asarray(period_feasible_after_list)
    period_projection_success = np.asarray(period_projection_success_list)
    period_objectives = np.asarray(period_objective_list)
    period_projection_times = np.asarray(period_projection_time_list)
    sample_feasible_before = period_feasible_before.all(axis=1)
    sample_feasible_after = period_feasible_after.all(axis=1)
    sample_objectives = period_objectives.sum(axis=1)
    projection_times = period_projection_times.sum(axis=1)
    decision_times = np.asarray(nn_times) + projection_times

    label_period_objectives = []
    for i in range(n_samples):
        _, _, objectives = evaluate_periods(
            X_temporal[i],
            Y_temporal[i],
            pv_bus,
            pv_capacity,
            branch_max,
            resistance,
        )
        label_period_objectives.append(objectives)
    label_period_objectives = np.asarray(label_period_objectives)
    label_objectives = label_period_objectives.sum(axis=1)

    result_dir = ROOT_DIR / "Test_result"
    result_dir.mkdir(exist_ok=True)
    result_path = result_dir / "results_O_NN_T12.npz"
    np.savez(
        result_path,
        method="NN_IPOPT_Projection_T12",
        dispatch=projected_dispatch,
        nn_times=np.asarray(nn_times),
        period_projection_times=period_projection_times,
        projection_times=projection_times,
        decision_times=decision_times,
        period_projection_success=period_projection_success,
        period_feasible_before=period_feasible_before,
        period_feasible=period_feasible_after,
        feasible_before=sample_feasible_before,
        feasible=sample_feasible_after,
        period_objective_values=period_objectives,
        objective_values=sample_objectives,
        label_period_objective_values=label_period_objectives,
        label_objective_values=label_objectives,
        n_periods=N_PERIODS,
    )

    projected_mask = ~period_feasible_before
    projected_time_mean = (
        period_projection_times[projected_mask].mean() if projected_mask.any() else 0.0
    )
    print(f"投影前样本可行率: {sample_feasible_before.mean() * 100:.2f}%")
    print(f"投影后样本可行率: {sample_feasible_after.mean() * 100:.2f}%")
    print(f"投影后逐时段可行率: {period_feasible_after.mean() * 100:.2f}%")
    print(f"需投影时段平均 IPOPT 时间: {projected_time_mean:.4f} s")
    print(f"12 时段总目标均值: {sample_objectives.mean():.6f}")
    print(f"Label 总目标均值: {label_objectives.mean():.6f}")
    print(f"结果已保存到: {result_path}")


if __name__ == "__main__":
    main()
