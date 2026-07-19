"""12 时段 NN 调度的逐时段内点计算与二分投影。"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

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
BISECTION_TOL = 1e-3
MAX_BISECTION_ITER = 50


def compute_inner_point(x_period, W_ip, b_ip, n_bus, n_pv):
    """用单时段仿射规则计算该时段的实际 PV 有功、无功内点。"""
    x_ip = np.concatenate(
        [
            x_period[1:n_bus],
            x_period[n_bus + 1 : 2 * n_bus],
            x_period[2 * n_bus : 2 * n_bus + n_pv],
        ]
    )
    return W_ip @ x_ip + b_ip


def bisection_projection(
    f_nn,
    f_ip,
    p_load,
    q_load,
    p_available,
    pv_bus,
    pv_capacity,
    feasible_before=None,
    bisection_tol=BISECTION_TOL,
    max_iter=MAX_BISECTION_ITER,
):
    """沿单时段内点到 NN 调度的线段搜索最靠近 NN 的可行点。"""
    n_pv = len(pv_bus)

    def is_feasible(f):
        pv_p, pv_q = f[:n_pv], f[n_pv:]
        V_sq, l_sq, _, _ = run_powerflow_numpy_single(
            p_load, q_load, pv_p, pv_q, pv_bus
        )
        return check_feasibility(
            V_sq, l_sq, pv_p, pv_q, p_available, pv_capacity
        )

    if feasible_before is None:
        feasible_before = is_feasible(f_nn)
    if feasible_before:
        return f_nn, 1.0, 0
    if not is_feasible(f_ip):
        raise RuntimeError("单时段仿射内点不可行，无法保证二分投影结果可行")

    k_lower, k_upper = 0.0, 1.0
    n_iter = 0
    while k_upper - k_lower >= bisection_tol and n_iter < max_iter:
        n_iter += 1
        k_mid = 0.5 * (k_lower + k_upper)
        f_test = f_ip + k_mid * (f_nn - f_ip)
        if is_feasible(f_test):
            k_lower = k_mid
        else:
            k_upper = k_mid

    return f_ip + k_lower * (f_nn - f_ip), k_lower, n_iter


def main():
    system_data = case()
    branch = np.asarray(system_data["branch"])
    bus = np.asarray(system_data["bus"])
    pv_bus, pv_capacity = PV_bus_define()
    _, _, r_x_ratio, branch_max = Y_bus_matrix()
    resistance = branch[:, 2] * r_x_ratio
    n_bus = bus.shape[0]
    n_pv = len(pv_bus)
    features_per_period = 2 * n_bus + n_pv
    outputs_per_period = 2 * n_pv

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

    affine = np.load(ROOT_DIR / "System_data" / "robust_affine_coefficients.npz")
    W_ip = np.vstack([affine["M_yPVp"], affine["M_yPVq"]])
    b_ip = np.concatenate([affine["m_yPVp"], affine["m_yPVq"]])
    expected_ip_input = 2 * (n_bus - 1) + n_pv
    if W_ip.shape != (outputs_per_period, expected_ip_input):
        raise ValueError(
            f"单时段内点系数 W_IP 期望 {(outputs_per_period, expected_ip_input)}，"
            f"实际 {W_ip.shape}。请重新运行 Robust_Affine_IP_Inner_convex_Distflow.py。"
        )

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
    print(f"12 时段 NN + 单时段内点二分投影: {n_samples} 个验证样本")
    print("=" * 70)

    projected_dispatch = []
    period_feasible_before_list = []
    period_feasible_after_list = []
    period_objective_list = []
    period_projection_time_list = []
    period_kappa_list = []
    period_iteration_list = []
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
        sample_objectives = np.empty(N_PERIODS)
        sample_projection_times = np.zeros(N_PERIODS)
        sample_kappa = np.ones(N_PERIODS)
        sample_iterations = np.zeros(N_PERIODS, dtype=int)

        for t in range(N_PERIODS):
            x_period = X_temporal[i, t]
            p_load = x_period[:n_bus]
            q_load = x_period[n_bus : 2 * n_bus]
            p_available = x_period[2 * n_bus :]

            f_nn = y_pred[t].copy()
            f_nn[:n_pv] *= p_available
            f_ip = compute_inner_point(x_period, W_ip, b_ip, n_bus, n_pv)

            V_nn, l_nn, _, _ = run_powerflow_numpy_single(
                p_load, q_load, f_nn[:n_pv], f_nn[n_pv:], pv_bus
            )
            sample_feasible_before[t] = check_feasibility(
                V_nn,
                l_nn,
                f_nn[:n_pv],
                f_nn[n_pv:],
                p_available,
                pv_capacity,
            )

            if sample_feasible_before[t]:
                f_proj, kappa, n_iter = f_nn, 1.0, 0
            else:
                start_projection = time.perf_counter()
                f_proj, kappa, n_iter = bisection_projection(
                    f_nn,
                    f_ip,
                    p_load,
                    q_load,
                    p_available,
                    pv_bus,
                    pv_capacity,
                    feasible_before=False,
                )
                sample_projection_times[t] = time.perf_counter() - start_projection
            sample_kappa[t] = kappa
            sample_iterations[t] = n_iter

            V_sq, l_sq, _, _ = run_powerflow_numpy_single(
                p_load, q_load, f_proj[:n_pv], f_proj[n_pv:], pv_bus
            )
            sample_feasible_after[t] = check_feasibility(
                V_sq,
                l_sq,
                f_proj[:n_pv],
                f_proj[n_pv:],
                p_available,
                pv_capacity,
            )
            sample_objectives[t] = compute_objective_single(
                l_sq, f_proj[:n_pv], p_available, resistance
            )
            sample_dispatch[t] = f_proj

        projected_dispatch.append(sample_dispatch)
        period_feasible_before_list.append(sample_feasible_before)
        period_feasible_after_list.append(sample_feasible_after)
        period_objective_list.append(sample_objectives)
        period_projection_time_list.append(sample_projection_times)
        period_kappa_list.append(sample_kappa)
        period_iteration_list.append(sample_iterations)

        if (i + 1) % 10 == 0 or i + 1 == n_samples:
            print(f"已处理 {i + 1}/{n_samples} 个 12 时段样本")

    projected_dispatch = np.asarray(projected_dispatch)
    period_feasible_before = np.asarray(period_feasible_before_list)
    period_feasible_after = np.asarray(period_feasible_after_list)
    period_objectives = np.asarray(period_objective_list)
    period_projection_times = np.asarray(period_projection_time_list)
    period_kappa = np.asarray(period_kappa_list)
    period_iterations = np.asarray(period_iteration_list)
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
    result_path = result_dir / "results_B_NN_T12.npz"
    np.savez(
        result_path,
        method="NN_Bisection_T12",
        dispatch=projected_dispatch,
        nn_times=np.asarray(nn_times),
        period_projection_times=period_projection_times,
        projection_times=projection_times,
        decision_times=decision_times,
        period_kappa=period_kappa,
        period_projection_iterations=period_iterations,
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
    print(f"需投影时段平均投影时间: {projected_time_mean:.4f} s")
    print(f"12 时段总目标均值: {sample_objectives.mean():.6f}")
    print(f"Label 总目标均值: {label_objectives.mean():.6f}")
    print(f"结果已保存到: {result_path}")


if __name__ == "__main__":
    main()
