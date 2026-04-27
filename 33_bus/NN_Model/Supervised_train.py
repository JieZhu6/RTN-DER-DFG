"""
使用生成的监督学习数据集进行训练
X: 负荷和光伏数据 (已标准化)
Y: 最优 PV 有功和无功功率 (已标准化)

训练集和测试集划分已在 Data_generation/normalize_dataset.py 中完成
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from System_data.system_config import case, Y_bus_matrix, PV_bus_define
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# 导入神经网络模型
from NN_Model.model import DERDispatchNet

# 随机种子和设备
np.random.seed(2)
torch.manual_seed(2)
plt.rcParams.update({'font.size': 16})
plt.rc('font', family='Times New Roman')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")  # 强制使用 CPU 进行训练
print(f"使用设备: {device}")

# ==============================================================================
# 1. 加载已划分的数据集
# ==============================================================================
print("="*70)
print("加载训练集和测试集")
print("="*70)

# 加载划分好的数据集
data = np.load("Data_generation/dataset_split.npz")

# 训练集（标准化空间）
X_train_np = data['X_train_norm']
Y_train_np = data['Y_train_raw']

# 测试集（标准化空间）
X_test_np = data['X_test_norm']
Y_test_np = data['Y_test_raw']
tets = data['Y_test_raw']  # 原始空间的测试输入
# 获取维度信息
n_train = X_train_np.shape[0]
n_test = X_test_np.shape[0]

print(f"训练集: {n_train} 样本")
print(f"测试集: {n_test} 样本")
print(f"输入特征维度: {X_train_np.shape[1]}")
print(f"输出标签维度: {Y_train_np.shape[1]}")

# 转换为 PyTorch 张量
X_train = torch.tensor(X_train_np, dtype=torch.float32)
Y_train = torch.tensor(Y_train_np, dtype=torch.float32)
X_test = torch.tensor(X_test_np, dtype=torch.float32)
Y_test = torch.tensor(Y_test_np, dtype=torch.float32)

# ==============================================================================
# 训练超参数设置 (在此调整)
# ==============================================================================

LEARNING_RATE = 1e-4          # 学习率
BATCH_SIZE = 128             # 批大小
EPOCHS = 3000                 # 训练轮数
weight_decay = 1e-4                # 权重衰减（L2正则化）
HIDDEN_DIMS = [32,32]      # 隐藏层维度

# 创建 DataLoader
train_dataset = TensorDataset(X_train, Y_train)
test_dataset = TensorDataset(X_test, Y_test)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ==============================================================================
# 2. 获取系统参数
# ==============================================================================
SYSTEM_DATA = case()
branch = np.array(SYSTEM_DATA['branch'])
bus = np.array(SYSTEM_DATA['bus'])
PV_bus, PV_capacity = PV_bus_define()
R_ij_matrix, X_ij_matrix, r_x_ratio, branch_max = Y_bus_matrix()

n_bus = bus.shape[0]
n_branches = branch.shape[0]
n_pv = len(PV_bus)

print(f"\n系统配置:")
print(f"  节点数: {n_bus}")
print(f"  支路数: {n_branches}")
print(f"  PV数: {n_pv}")

# ==============================================================================
# 3. 定义神经网络模型
# ==============================================================================
print("\n" + "="*70)
print("初始化神经网络模型")
print("="*70)

model = DERDispatchNet(n_bus, n_pv, hidden_dims=HIDDEN_DIMS).to(device)

# ==============================================================================
# 4. 训练配置
# ==============================================================================
criterion = nn.MSELoss()
# optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=weight_decay)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
epochs = EPOCHS
best_test_loss = float('inf')
best_model_state = None
no_improve_count = 0   # 可用于 early stopping（可选）
early_stop_patience = 300  # 如果启用 early stopping

train_losses = []
test_losses = []

# ==============================================================================
# 5. 训练循环（含调度器）
# ==============================================================================
print("\n" + "="*70)
print("开始训练")
print("="*70)

start_time = time.time()

for epoch in range(epochs):
    # 训练阶段
    model.train()
    epoch_train_loss = 0.0
    
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item() * x_batch.size(0)
    
    epoch_train_loss /= n_train
    train_losses.append(epoch_train_loss)
    # 测试阶段
    model.eval()
    epoch_test_loss = 0.0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            epoch_test_loss += loss.item() * x_batch.size(0)
    
    epoch_test_loss /= n_test
    test_losses.append(epoch_test_loss)
    
    # 保存最佳模型
    if epoch_test_loss < best_test_loss:
        best_test_loss = epoch_test_loss
        best_model_state = model.state_dict().copy()
        no_improve_count = 0  # 重置计数
        os.makedirs("NN_parameter", exist_ok=True)

        model_save_path = f"NN_parameter/supervised_trained_model_{HIDDEN_DIMS}.pth"
        torch.save({
            'model_state_dict': best_model_state,
            'train_loss': train_losses,
            'test_loss': test_losses,
            'best_test_loss': best_test_loss,
            'n_bus': n_bus,
            'n_pv': n_pv,
        }, model_save_path)
    else:
        no_improve_count += 1

    # （可选）Early Stopping：如果学习率已很低且长时间不改善，提前停止
    current_lr = optimizer.param_groups[0]['lr']
    if no_improve_count >= early_stop_patience:
        print(f"\n[Early Stop] 学习率已达最小值 {current_lr:.2e} 且 {early_stop_patience} 轮无改善，提前终止训练。")
        break

    # 每10轮打印一次
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], "
              f"Train Loss: {epoch_train_loss:.6f}, "
              f"Test Loss: {epoch_test_loss:.6f}, "
              f"LR: {current_lr:.2e}")

training_time = time.time() - start_time
print(f"\n训练完成! 总耗时: {training_time:.2f} 秒")
print(f"最佳测试 Loss: {best_test_loss:.6f}")

# 加载最佳模型
model.load_state_dict(best_model_state)

# ==============================================================================
# 6. 保存训练好的模型
# ==============================================================================


# 绘制训练曲线
fig, ax = plt.subplots(figsize=(10, 6))
ax.semilogy(train_losses, label='Train Loss')
ax.semilogy(test_losses, label='Test Loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE Loss (log scale)')
ax.set_title('Training Curve')
ax.legend()
ax.grid(True)
plt.savefig('Test_result/Supervised_training_curve.png', dpi=300, bbox_inches='tight')
print(f"训练曲线已保存到: Test_result/Supervised_training_curve.png")

print("\n" + "="*70)
print("训练完成!")
