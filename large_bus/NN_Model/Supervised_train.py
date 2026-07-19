"""训练 12 时段联合 PV 调度网络。"""

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
from System_data.system_config import PV_bus_define, case


N_PERIODS = 12
LEARNING_RATE = 1e-3
BATCH_SIZE = 256
EPOCHS = 3000
WEIGHT_DECAY = 1e-3
HIDDEN_DIMS = [256, 256, 256]
EARLY_STOP_PATIENCE = 300


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


def main():
    np.random.seed(2)
    torch.manual_seed(2)
    plt.rcParams.update({"font.size": 16})
    plt.rc("font", family="Times New Roman")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    system_data = case()
    n_bus = np.asarray(system_data["bus"]).shape[0]
    n_pv = len(PV_bus_define()[0])

    data_path = ROOT_DIR / "Data_generation" / "dataset_split.npz"
    data = np.load(data_path)
    X_train_np = data["X_train_norm"]
    Y_train_np = data["Y_train_raw"]
    X_test_np = data["X_test_norm"]
    Y_test_np = data["Y_test_raw"]

    validate_dataset_shapes(X_train_np, Y_train_np, n_bus, n_pv, "训练集")
    validate_dataset_shapes(X_test_np, Y_test_np, n_bus, n_pv, "测试集")

    print("=" * 70)
    print(f"加载 {N_PERIODS} 时段监督学习数据集")
    print("=" * 70)
    print(f"训练集: {len(X_train_np)} 样本")
    print(f"测试集: {len(X_test_np)} 样本")
    print(f"输入维度: {X_train_np.shape[1]}")
    print(f"输出维度: {Y_train_np.shape[1]}")

    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    Y_train = torch.tensor(Y_train_np, dtype=torch.float32)
    X_test = torch.tensor(X_test_np, dtype=torch.float32)
    Y_test = torch.tensor(Y_test_np, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_train, Y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(X_test, Y_test),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = DERDispatchNet(
        n_bus,
        n_pv,
        hidden_dims=HIDDEN_DIMS,
        n_periods=N_PERIODS,
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    best_test_loss = float("inf")
    best_model_state = None
    no_improve_count = 0
    train_losses = []
    test_losses = []

    parameter_dir = ROOT_DIR / "NN_parameter"
    parameter_dir.mkdir(exist_ok=True)
    model_path = parameter_dir / f"supervised_trained_model_T{N_PERIODS}_{HIDDEN_DIMS}.pth"

    print("\n" + "=" * 70)
    print("开始训练")
    print("=" * 70)
    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        epoch_train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * x_batch.shape[0]

        epoch_train_loss /= len(X_train)
        train_losses.append(epoch_train_loss)

        model.eval()
        epoch_test_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                loss = criterion(model(x_batch), y_batch)
                epoch_test_loss += loss.item() * x_batch.shape[0]

        epoch_test_loss /= len(X_test)
        test_losses.append(epoch_test_loss)

        if epoch_test_loss < best_test_loss:
            best_test_loss = epoch_test_loss
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

        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch [{epoch + 1}/{EPOCHS}], "
                f"Train Loss: {epoch_train_loss:.6f}, "
                f"Test Loss: {epoch_test_loss:.6f}, "
                f"LR: {current_lr:.2e}"
            )

        if no_improve_count >= EARLY_STOP_PATIENCE:
            print(f"\n连续 {EARLY_STOP_PATIENCE} 轮未改善，提前停止训练。")
            break

    model.load_state_dict(best_model_state)
    training_time = time.time() - start_time
    print(f"\n训练完成，耗时: {training_time:.2f} 秒")
    print(f"最佳测试 Loss: {best_test_loss:.6f}")
    print(f"模型已保存到: {model_path}")

    result_dir = ROOT_DIR / "Test_result"
    result_dir.mkdir(exist_ok=True)
    figure_path = result_dir / "Supervised_training_curve_T12.png"
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(train_losses, label="Train Loss")
    ax.semilogy(test_losses, label="Test Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (log scale)")
    ax.set_title("12-period Supervised Training")
    ax.legend()
    ax.grid(True)
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"训练曲线已保存到: {figure_path}")


if __name__ == "__main__":
    main()
