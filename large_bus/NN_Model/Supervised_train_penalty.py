"""使用逐时段可微潮流约束训练 12 时段联合 PV 调度网络。"""

import copy
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from NN_Model.model import DERDispatchNet
from NN_Model.powerflow_env import run_differentiable_powerflow
from System_data.system_config import PV_bus_define, Y_bus_matrix, case


N_PERIODS = 12
LEARNING_RATE = 1e-3
# powerflow_env 的反向传播会构造稠密雅可比；大规模系统必须使用小 batch。
BATCH_SIZE = 1
EPOCHS = 3000
WEIGHT_DECAY = 1e-3
HIDDEN_DIMS = [256, 256, 256]
EARLY_STOP_PATIENCE = 15
USE_PRETRAINED = True

K_OBJ = 1.0
K_V = 1.0
K_L = 1.0
K_PV_P = 1.0
K_PV_S = 1.0


def validate_dataset_shapes(X, Y, n_bus, n_pv, name):
    expected_x = N_PERIODS * (2 * n_bus + n_pv)
    expected_y = N_PERIODS * 2 * n_pv
    if X.ndim != 2 or X.shape[1] != expected_x:
        raise ValueError(
            f"{name} X 期望形状 (*, {expected_x})，实际 {X.shape}。"
            "请先重新运行 generate_optimal_dataset.py 和 normalize_dataset.py。"
        )
    if Y.ndim != 2 or Y.shape[1] != expected_y:
        raise ValueError(
            f"{name} Y 期望形状 (*, {expected_y})，实际 {Y.shape}。"
            "请先重新运行 generate_optimal_dataset.py 和 normalize_dataset.py。"
        )


def compute_multi_period_loss_terms(
    x_batch_norm,
    y_pred_flat,
    y_label_flat,
    x_mean,
    x_scale,
    n_bus,
    n_pv,
    pv_bus,
    pv_capacity,
    branch_max,
    resistance,
):
    """逐时段调用潮流，并将 12 时段目标和约束违反量聚合。"""
    batch_size = x_batch_norm.shape[0]
    features_per_period = 2 * n_bus + n_pv
    outputs_per_period = 2 * n_pv

    x_raw = (x_batch_norm * x_scale + x_mean).reshape(
        batch_size, N_PERIODS, features_per_period
    )
    y_pred = y_pred_flat.reshape(batch_size, N_PERIODS, outputs_per_period)
    y_label = y_label_flat.reshape(batch_size, N_PERIODS, outputs_per_period)

    total_objective_pred = torch.zeros(
        batch_size, dtype=x_batch_norm.dtype, device=x_batch_norm.device
    )
    total_objective_label = torch.zeros_like(total_objective_pred)
    voltage_penalty = torch.zeros((), dtype=x_batch_norm.dtype, device=x_batch_norm.device)
    line_penalty = torch.zeros_like(voltage_penalty)
    pv_p_penalty = torch.zeros_like(voltage_penalty)
    pv_s_penalty = torch.zeros_like(voltage_penalty)

    # powerflow_env 接收单时段二维 batch，因此明确逐时段调用。
    for t in range(N_PERIODS):
        p_load = x_raw[:, t, :n_bus]
        q_load = x_raw[:, t, n_bus : 2 * n_bus]
        p_available = x_raw[:, t, 2 * n_bus :]

        p_pv = y_pred[:, t, :n_pv] * p_available
        q_pv = y_pred[:, t, n_pv:]
        p_pv_label = y_label[:, t, :n_pv] * p_available
        q_pv_label = y_label[:, t, n_pv:]

        V_sq, l_sq = run_differentiable_powerflow(
            p_load, q_load, p_pv, q_pv, pv_bus
        )
        # 标签潮流只用于构造目标值，不需要建立反向传播图。
        with torch.no_grad():
            _, l_sq_label = run_differentiable_powerflow(
                p_load, q_load, p_pv_label, q_pv_label, pv_bus
            )

        total_objective_pred = total_objective_pred + (
            (resistance * l_sq).sum(dim=1)
            + (p_available - p_pv).sum(dim=1)
        )
        total_objective_label = total_objective_label + (
            (resistance * l_sq_label).sum(dim=1)
            + (p_available - p_pv_label).sum(dim=1)
        )

        # 先对每个样本的节点/支路/PV 求和，再对 batch 取平均；时段间累加。
        voltage_penalty = voltage_penalty + (
            torch.relu(0.95**2 - V_sq).sum(dim=1)
            + torch.relu(V_sq - 1.05**2).sum(dim=1)
        ).mean()
        line_penalty = line_penalty + torch.relu(
            l_sq - branch_max**2
        ).sum(dim=1).mean()
        pv_p_penalty = pv_p_penalty + torch.relu(
            p_pv - p_available
        ).sum(dim=1).mean()
        pv_s_penalty = pv_s_penalty + torch.relu(
            p_pv**2 + q_pv**2 - pv_capacity**2
        ).sum(dim=1).mean()

    objective_loss = nn.functional.mse_loss(
        total_objective_pred, total_objective_label
    )
    return {
        "objective": objective_loss,
        "voltage": voltage_penalty,
        "line": line_penalty,
        "pv_p": pv_p_penalty,
        "pv_s": pv_s_penalty,
    }


def weighted_total_loss(terms):
    return (
        K_OBJ * terms["objective"]
        + K_V * terms["voltage"]
        + K_L * terms["line"]
        + K_PV_P * terms["pv_p"]
        + K_PV_S * terms["pv_s"]
    )


def main():
    np.random.seed(2)
    torch.manual_seed(2)
    plt.rcParams.update({"font.size": 16})
    plt.rc("font", family="Times New Roman")
    device = torch.device("cpu")

    system_data = case()
    branch = np.asarray(system_data["branch"])
    bus = np.asarray(system_data["bus"])
    pv_bus, pv_capacity = PV_bus_define()
    _, _, r_x_ratio, branch_max = Y_bus_matrix()
    n_bus = bus.shape[0]
    n_pv = len(pv_bus)
    resistance = torch.tensor(
        branch[:, 2] * r_x_ratio, dtype=torch.float32, device=device
    )

    data = np.load(ROOT_DIR / "Data_generation" / "dataset_split.npz")
    scaler = np.load(ROOT_DIR / "Data_generation" / "scaler_params.npz")
    X_train_np = data["X_train_norm"]
    Y_train_np = data["Y_train_raw"]
    X_test_np = data["X_test_norm"]
    Y_test_np = data["Y_test_raw"]

    validate_dataset_shapes(X_train_np, Y_train_np, n_bus, n_pv, "训练集")
    validate_dataset_shapes(X_test_np, Y_test_np, n_bus, n_pv, "测试集")
    expected_x_dim = N_PERIODS * (2 * n_bus + n_pv)
    if scaler["X_mean"].shape != (expected_x_dim,):
        raise ValueError(
            f"scaler_params.npz 的 X_mean 维度为 {scaler['X_mean'].shape}，"
            f"期望 ({expected_x_dim},)。请重新运行 normalize_dataset.py。"
        )

    x_mean = torch.tensor(scaler["X_mean"], dtype=torch.float32, device=device)
    x_scale = torch.tensor(scaler["X_scale"], dtype=torch.float32, device=device)

    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    Y_train = torch.tensor(Y_train_np, dtype=torch.float32)
    X_test = torch.tensor(X_test_np, dtype=torch.float32)
    Y_test = torch.tensor(Y_test_np, dtype=torch.float32)
    train_loader = DataLoader(
        TensorDataset(X_train, Y_train), batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(X_test, Y_test), batch_size=BATCH_SIZE, shuffle=False
    )

    print("=" * 70)
    print(f"加载 {N_PERIODS} 时段 penalty 训练数据")
    print("=" * 70)
    print(f"训练集: {len(X_train)}，测试集: {len(X_test)}")
    print(f"输入维度: {X_train.shape[1]}，输出维度: {Y_train.shape[1]}")
    print(f"Penalty batch size: {BATCH_SIZE}")

    model = DERDispatchNet(
        n_bus,
        n_pv,
        hidden_dims=HIDDEN_DIMS,
        n_periods=N_PERIODS,
    ).to(device)

    parameter_dir = ROOT_DIR / "NN_parameter"
    parameter_dir.mkdir(exist_ok=True)
    pretrained_path = (
        parameter_dir / f"supervised_trained_model_T{N_PERIODS}_{HIDDEN_DIMS}.pth"
    )
    if USE_PRETRAINED and pretrained_path.exists():
        checkpoint = torch.load(pretrained_path, map_location=device, weights_only=True)
        checkpoint_shape = (
            checkpoint.get("n_bus"),
            checkpoint.get("n_pv"),
            checkpoint.get("n_periods", 1),
        )
        expected_shape = (n_bus, n_pv, N_PERIODS)
        if checkpoint_shape == expected_shape:
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"已加载 12 时段监督预训练模型: {pretrained_path}")
        else:
            print(
                f"预训练模型维度 {checkpoint_shape} 与当前 {expected_shape} 不一致，"
                "将从头训练。"
            )
    elif USE_PRETRAINED:
        print(f"未找到预训练模型 {pretrained_path}，将从头训练。")

    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    model_path = parameter_dir / (
        f"penalty_model_T{N_PERIODS}_{HIDDEN_DIMS}_{BATCH_SIZE}_{LEARNING_RATE}.pth"
    )

    best_test_loss = float("inf")
    best_model_state = None
    no_improve_count = 0
    train_losses = []
    test_losses = []
    penalty_losses = []
    start_time = time.time()

    term_names = ("objective", "voltage", "line", "pv_p", "pv_s")
    for epoch in range(EPOCHS):
        model.train()
        train_sums = {name: 0.0 for name in term_names}
        train_total = 0.0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            batch_size = x_batch.shape[0]

            optimizer.zero_grad()
            y_pred = model(x_batch)
            terms = compute_multi_period_loss_terms(
                x_batch,
                y_pred,
                y_batch,
                x_mean,
                x_scale,
                n_bus,
                n_pv,
                pv_bus,
                pv_capacity,
                branch_max,
                resistance,
            )
            loss = weighted_total_loss(terms)
            loss.backward()
            optimizer.step()

            train_total += loss.item() * batch_size
            for name in term_names:
                train_sums[name] += terms[name].item() * batch_size

        train_total /= len(X_train)
        train_means = {name: value / len(X_train) for name, value in train_sums.items()}
        train_penalty = sum(train_means[name] for name in term_names[1:])
        train_losses.append(train_total)
        penalty_losses.append(train_penalty)

        model.eval()
        test_sums = {name: 0.0 for name in term_names}
        test_total = 0.0
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                batch_size = x_batch.shape[0]
                terms = compute_multi_period_loss_terms(
                    x_batch,
                    model(x_batch),
                    y_batch,
                    x_mean,
                    x_scale,
                    n_bus,
                    n_pv,
                    pv_bus,
                    pv_capacity,
                    branch_max,
                    resistance,
                )
                test_total += weighted_total_loss(terms).item() * batch_size
                for name in term_names:
                    test_sums[name] += terms[name].item() * batch_size

        test_total /= len(X_test)
        test_means = {name: value / len(X_test) for name, value in test_sums.items()}
        test_penalty = sum(test_means[name] for name in term_names[1:])
        test_losses.append(test_total)

        selection_loss = test_means["objective"] + test_penalty
        if selection_loss < best_test_loss:
            best_test_loss = selection_loss
            best_model_state = copy.deepcopy(model.state_dict())
            no_improve_count = 0
            torch.save(
                {
                    "model_state_dict": best_model_state,
                    "train_loss": train_losses,
                    "test_loss": test_losses,
                    "best_test_loss": best_test_loss,
                    "n_bus": n_bus,
                    "n_pv": n_pv,
                    "n_periods": N_PERIODS,
                    "hidden_dims": HIDDEN_DIMS,
                },
                model_path,
            )
        else:
            no_improve_count += 1

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] | "
            f"Train Total={train_total:.4f} Obj={train_means['objective']:.4f} "
            f"Penalty={train_penalty:.4f} | "
            f"Test Total={test_total:.4f} Obj={test_means['objective']:.4f} "
            f"Penalty={test_penalty:.4f} | "
            f"Time={time.time() - start_time:.2f}s"
        )

        if no_improve_count >= EARLY_STOP_PATIENCE:
            print(f"连续 {EARLY_STOP_PATIENCE} 轮未改善，提前停止训练。")
            break

    print(f"\n训练完成，耗时: {time.time() - start_time:.2f} 秒")
    print(f"最佳测试 Loss: {best_test_loss:.6f}")
    print(f"模型已保存到: {model_path}")

    result_dir = ROOT_DIR / "Test_result"
    result_dir.mkdir(exist_ok=True)
    figure_path = result_dir / "USL_penalty_curve_T12.png"
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(train_losses, label="Train Loss")
    ax.semilogy(test_losses, label="Test Loss")
    ax.semilogy(penalty_losses, label="Train Penalty")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title("12-period Penalty Training")
    ax.legend()
    ax.grid(True)
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"训练曲线已保存到: {figure_path}")


if __name__ == "__main__":
    main()
