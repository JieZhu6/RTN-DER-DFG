"""验证 12 时段 penalty NN 的直接调度结果。"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from NN_Model.model import DERDispatchNet
from Solving_method.NN_direct import evaluate_periods
from System_data.system_config import PV_bus_define, Y_bus_matrix, case


N_PERIODS = 12
N_VALIDATE_SAMPLES = None
HIDDEN_DIMS = [256, 256, 256]
LEARNING_RATE = 1e-3
TRAIN_BATCH_SIZE = 1


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
    print(f"12 时段 penalty NN 直接调度: {n_samples} 个验证样本")
    print(f"系统: {n_bus} 节点, {len(branch)} 支路, {n_pv} 个 PV")
    print("=" * 70)

    dispatch_list = []
    period_feasible_list = []
    period_objective_list = []
    nn_times = []

    for i in range(n_samples):
        x_tensor = torch.tensor(X_norm[i : i + 1], dtype=torch.float32, device=device)
        start = time.perf_counter()
        with torch.no_grad():
            y_pred = model(x_tensor).cpu().numpy().reshape(
                N_PERIODS, outputs_per_period
            )
        nn_times.append(time.perf_counter() - start)

        dispatch, feasible, objectives = evaluate_periods(
            X_temporal[i],
            y_pred,
            pv_bus,
            pv_capacity,
            branch_max,
            resistance,
        )
        dispatch_list.append(dispatch)
        period_feasible_list.append(feasible)
        period_objective_list.append(objectives)

        if (i + 1) % 10 == 0 or i + 1 == n_samples:
            print(f"已处理 {i + 1}/{n_samples} 个 12 时段样本")

    dispatch_array = np.asarray(dispatch_list)
    period_feasible = np.asarray(period_feasible_list)
    period_objectives = np.asarray(period_objective_list)
    sample_feasible = period_feasible.all(axis=1)
    sample_objectives = period_objectives.sum(axis=1)

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
    result_path = result_dir / "results_P_NN_T12.npz"
    np.savez(
        result_path,
        method="NN_Penalty_T12",
        dispatch=dispatch_array,
        nn_times=np.asarray(nn_times),
        decision_times=np.asarray(nn_times),
        period_feasible=period_feasible,
        feasible=sample_feasible,
        period_objective_values=period_objectives,
        objective_values=sample_objectives,
        label_period_objective_values=label_period_objectives,
        label_objective_values=label_objectives,
        n_periods=N_PERIODS,
    )

    gap = (sample_objectives.mean() - label_objectives.mean()) / (
        abs(label_objectives.mean()) + 1e-8
    )
    print(f"样本可行率（12 时段全部可行）: {sample_feasible.mean() * 100:.2f}%")
    print(f"逐时段可行率: {period_feasible.mean() * 100:.2f}%")
    print(f"NN 12 时段总目标均值: {sample_objectives.mean():.6f}")
    print(f"Label 12 时段总目标均值: {label_objectives.mean():.6f}")
    print(f"平均目标差距: {gap * 100:.3f}%")
    print(f"结果已保存到: {result_path}")


if __name__ == "__main__":
    main()
