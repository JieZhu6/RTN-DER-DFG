import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from System_data.system_config import case, Y_bus_matrix, PV_bus_define
from NN_Model.powerflow_env import run_differentiable_powerflow
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import gradcheck
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
import os
# 导入自定义模块
from NN_Model.model import DERDispatchNet

# ==============================================================================
# 0. 全局设置
# ==============================================================================
np.random.seed(2)
torch.manual_seed(2)
plt.rcParams.update({'font.size': 16})
plt.rc('font', family='Times New Roman')
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
# ==============================================================================
# 1. 获取系统拓扑与数据加载
# ==============================================================================
SYSTEM_DATA = case()
branch = np.array(SYSTEM_DATA['branch'])
bus = np.array(SYSTEM_DATA['bus'])
PV_bus, PV_capacity = PV_bus_define()
R_ij_matrix, X_ij_matrix, r_x_ratio, branch_max = Y_bus_matrix()

R_vector = branch[:, 2] * r_x_ratio  # 调整后的电阻值，单位 p.u.，长度为 n_branches
# 网损项
R_vector_tensor = torch.tensor(R_vector, dtype=torch.float32, device=device)
n_bus = bus.shape[0]
n_branches = branch.shape[0]
n_pv = len(PV_bus)
# 加载划分好的数据集
data = np.load("Data_generation/dataset_split.npz")
scaler = np.load("Data_generation/scaler_params.npz")

# 训练集（标准化空间）
X_train_np = data['X_train_norm']
Y_train_np = data['Y_train_raw']

# 测试集（标准化空间）
X_test_np = data['X_test_norm']
Y_test_np = data['Y_test_raw']

# 加载标准化参数（用于反标准化）
X_mean = scaler['X_mean']
X_scale = scaler['X_scale']
Y_mean = scaler['Y_mean']
Y_scale = scaler['Y_scale']

# 转换为 torch tensor
X_mean_tensor = torch.tensor(X_mean, dtype=torch.float32, device=device)
X_scale_tensor = torch.tensor(X_scale, dtype=torch.float32, device=device)
Y_mean_tensor = torch.tensor(Y_mean, dtype=torch.float32, device=device)
Y_scale_tensor = torch.tensor(Y_scale, dtype=torch.float32, device=device)

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

LEARNING_RATE = 1e-3          # 学习率
BATCH_SIZE = 64           # 批大小
EPOCHS = 3000                 # 训练轮数
weight_decay = 1e-3                # 权重衰减（L2正则化）
# HIDDEN_DIMS = [64, 64, 64]      # 隐藏层维度
# HIDDEN_DIMS = [64, 64] 
HIDDEN_DIMS = [32,32]
# HIDDEN_DIMS = [128,128] 
# 创建 DataLoader
train_dataset = TensorDataset(X_train, Y_train)
test_dataset = TensorDataset(X_test, Y_test)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ==============================================================================
# 3. 初始化模型与训练环境
# ==============================================================================
model = DERDispatchNet(n_bus, n_pv, HIDDEN_DIMS).to(device)

# 加载监督学习预训练模型参数（如果存在）
PRETRAINED_MODEL_PATH = f"NN_parameter/supervised_trained_model_{HIDDEN_DIMS}.pth"
# PRETRAINED_MODEL_PATH = f"NN_parameter/penalty_model_{HIDDEN_DIMS}.pth"
USE_PRETRAINED = True  # 设置为 False 则从头训练

if USE_PRETRAINED and os.path.exists(PRETRAINED_MODEL_PATH):
    print(f"\n加载监督学习预训练模型: {PRETRAINED_MODEL_PATH}")
    checkpoint = torch.load(PRETRAINED_MODEL_PATH, map_location=device, weights_only=True)
    
    # 检查模型结构是否匹配
    pretrained_n_bus = checkpoint.get('n_bus', n_bus)
    pretrained_n_pv = checkpoint.get('n_pv', n_pv)
    
    if pretrained_n_bus == n_bus and pretrained_n_pv == n_pv:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("预训练参数加载成功！")
        print(f"  - 预训练模型最优Loss: {checkpoint.get('best_test_loss', 'N/A')}")
    else:
        print(f"警告: 模型结构不匹配！预训练模型 (n_bus={pretrained_n_bus}, n_pv={pretrained_n_pv}) "
              f"vs 当前模型 (n_bus={n_bus}, n_pv={n_pv})")
        print("将从头开始训练...")
else:
    if USE_PRETRAINED:
        print(f"\n未找到预训练模型: {PRETRAINED_MODEL_PATH}")
        print("将从头开始训练...")
    else:
        print("\nUSE_PRETRAINED=False，从头开始训练...")

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=weight_decay)
# optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
epochs = EPOCHS
best_test_loss = float('inf')
best_model_state = None
no_improve_count = 0   # 可用于 early stopping（可选）
early_stop_patience = 15  # 如果启用 early stopping
# ==============================================================================
# 4. 训练主循环（等式约束通过神经网络直接完成）
# ==============================================================================
start_time = time.time()
train_losses = []
test_losses = []
penalty_losses = []
# 自动调节罚函数系数
# penalty 系数初始化
k_obj = 1
k_initial = 1
# 罚函数
k_bar = 1 # 上限值
k_min = 0.1
k_v,k_l,k_pv_P,k_pv_S = k_initial,k_initial,k_initial,k_initial 
time_start_epoch = time.time()
for epoch in range(epochs):
    # 训练阶段
    model.train()
    epoch_train_loss = 0.0
    epoch_train_penalty_loss = 0.0
    # 统计目标函数和罚函数的值
    sum_obj = 0.0
    sum_voltage = 0.0
    sum_line = 0.0
    sum_pv_P = 0.0
    sum_pv_S = 0.0
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        y_pred = model(x_batch)

        # 反标准化：将标准化的 P 和 Q 转换为原始空间
        # x_batch 反标准化
        x_batch_raw = x_batch * X_scale_tensor + X_mean_tensor
        P_load_batch = x_batch_raw[:, :n_bus]
        Q_load_batch = x_batch_raw[:, n_bus:2*n_bus]
        p_available = x_batch_raw[:, 2*n_bus:2*n_bus+n_pv]
        P_pv_batch = y_pred[:, :n_pv] * p_available  # 得到实际出力
        Q_pv_batch = y_pred[:, n_pv:]
        P_pv_label = y_batch[:, :n_pv] * p_available  # 标签中的实际出力
        Q_pv_label = y_batch[:, n_pv:]

        # 运行潮流方程得到状态变量（使用原始空间数值）
        V_sq, l_sq = run_differentiable_powerflow(P_load_batch, Q_load_batch, P_pv_batch, Q_pv_batch, PV_bus)
        V_sq_label, l_sq_label = run_differentiable_powerflow(P_load_batch, Q_load_batch, P_pv_label, Q_pv_label, PV_bus)
        # 正确的目标函数：网损总和 + 弃光总和
        obj_pre = (R_vector_tensor * l_sq).sum(dim=1) + (p_available - P_pv_batch).sum(dim=1)
        obj_label = (R_vector_tensor * l_sq_label).sum(dim=1) + (p_available - P_pv_label).sum(dim=1)

        # 确保是标量，用于MSE计算
        obj_pre = obj_pre.view(-1, 1)  # 或保持 (batch,) 均可，只要与 obj_label 一致
        obj_label = obj_label.view(-1, 1)

        # 4. 计算损失函数
        # obj_loss = obj_pre.mean() 
        obj_loss = nn.MSELoss()(y_pred, y_batch) + nn.MSELoss()(obj_pre, obj_label)
        # obj_loss = nn.MSELoss()(obj_pre, obj_label)
        # obj_loss = nn.MSELoss()(y_pred, y_batch)
        # 约束违反：每个样本的惩罚取平均
        voltage_penalty = torch.relu(-V_sq + 0.95**2).sum(dim=1).mean() \
                        + torch.relu(V_sq - 1.05**2).sum(dim=1).mean()

        line_penalty = torch.relu(l_sq - branch_max**2).sum(dim=1).mean()

        pv_penalty_P = torch.relu(P_pv_batch - p_available).sum(dim=1).mean()
        pv_penalty_S = torch.relu(P_pv_batch**2 + Q_pv_batch**2 - PV_capacity**2).sum(dim=1).mean()

        # 动态罚函数
        loss = k_obj * obj_loss \
            + k_v * voltage_penalty \
            + k_l * line_penalty \
            + k_pv_P * pv_penalty_P\
            + k_pv_S * pv_penalty_S
        # 5. 梯度回传
        loss.backward()
        optimizer.step()

        # 统计损失函数值 (累加每个 batch 的平均值，最后求 epoch 平均)
        sum_obj += obj_loss.item()
        sum_voltage += voltage_penalty.item()
        sum_line += line_penalty.item()
        sum_pv_P += pv_penalty_P.item()
        sum_pv_S += pv_penalty_S.item()

        epoch_train_loss += loss.item()
        epoch_train_penalty_loss += voltage_penalty.item() + line_penalty.item() + pv_penalty_P.item() + pv_penalty_S.item()

    num_batches = len(train_loader)
    epoch_train_loss /= num_batches
    epoch_train_penalty_loss /= num_batches
    epoch_obj_loss = sum_obj / num_batches
    epoch_voltage_penalty = sum_voltage / num_batches
    epoch_line_penalty = sum_line / num_batches
    epoch_pv_penalty_P = sum_pv_P / num_batches
    epoch_pv_penalty_S = sum_pv_S / num_batches
    train_losses.append(epoch_train_loss)    
    penalty_losses.append(epoch_train_penalty_loss)
    
    # 测试阶段
    model.eval()
    epoch_test_loss = 0.0
    epoch_test_obj = 0.0
    epoch_test_penalty = 0.0
    # 分开统计各惩罚项
    sum_test_voltage = 0.0
    sum_test_line = 0.0
    sum_test_pv_P = 0.0
    sum_test_pv_S = 0.0
    
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred = model(x_batch)
            x_batch_raw = x_batch * X_scale_tensor + X_mean_tensor
            P_load_batch = x_batch_raw[:, :n_bus]
            Q_load_batch = x_batch_raw[:, n_bus:2*n_bus]
            p_available = x_batch_raw[:, 2*n_bus:2*n_bus+n_pv]
            P_pv_batch = y_pred[:, :n_pv] * p_available  # 得到实际出力
            Q_pv_batch = y_pred[:, n_pv:]
            P_pv_label = y_batch[:, :n_pv] * p_available  # 标签中的实际出力
            Q_pv_label = y_batch[:, n_pv:]

            # 运行潮流方程得到状态变量（使用原始空间数值）
            V_sq, l_sq = run_differentiable_powerflow(P_load_batch, Q_load_batch, P_pv_batch, Q_pv_batch, PV_bus)
            V_sq_label, l_sq_label = run_differentiable_powerflow(P_load_batch, Q_load_batch, P_pv_label, Q_pv_label, PV_bus)
            # 正确的目标函数：网损总和 + 弃光总和
            obj_pre = (R_vector_tensor * l_sq).sum(dim=1) + (p_available - P_pv_batch).sum(dim=1)
            obj_label = (R_vector_tensor * l_sq_label).sum(dim=1) + (p_available - P_pv_label).sum(dim=1)

            # 确保是标量，用于MSE计算
            obj_pre = obj_pre.view(-1, 1)  # 或保持 (batch,) 均可，只要与 obj_label 一致
            obj_label = obj_label.view(-1, 1)

            # 4. 计算损失函数
            # obj_loss = obj_pre.mean() 
            obj_loss = nn.MSELoss()(y_pred, y_batch) + nn.MSELoss()(obj_pre, obj_label) 
            # obj_loss = nn.MSELoss()(obj_pre, obj_label)
            # obj_loss = nn.MSELoss()(y_pred, y_batch)

            # 约束违反：每个样本的惩罚取平均
            voltage_penalty = torch.relu(-V_sq + 0.95**2).sum(dim=1).mean() \
                            + torch.relu(V_sq - 1.05**2).sum(dim=1).mean()
            line_penalty = torch.relu(l_sq - branch_max**2).sum(dim=1).mean()
            pv_penalty_P = torch.relu(P_pv_batch - p_available).sum(dim=1).mean()
            pv_penalty_S = torch.relu(P_pv_batch**2 + Q_pv_batch**2 - PV_capacity**2).sum(dim=1).mean()
            
            # 测试时使用当前的罚函数系数计算总损失
            test_loss = k_obj * obj_loss \
                + k_v * voltage_penalty \
                + k_l * line_penalty \
                + k_pv_P * pv_penalty_P \
                + k_pv_S * pv_penalty_S
            
            epoch_test_loss += test_loss.item()
            epoch_test_obj += obj_loss.item()
            epoch_test_penalty += voltage_penalty.item() + line_penalty.item() + pv_penalty_P.item() + pv_penalty_S.item()
            
            # 分开统计各惩罚项
            sum_test_voltage += voltage_penalty.item()
            sum_test_line += line_penalty.item()
            sum_test_pv_P += pv_penalty_P.item()
            sum_test_pv_S += pv_penalty_S.item()
    
    num_test_batches = len(test_loader)
    epoch_test_loss /= num_test_batches
    epoch_test_obj /= num_test_batches
    epoch_test_penalty /= num_test_batches
    # 各惩罚项平均值
    epoch_test_voltage = sum_test_voltage / num_test_batches
    epoch_test_line = sum_test_line / num_test_batches
    epoch_test_pv_P = sum_test_pv_P / num_test_batches
    epoch_test_pv_S = sum_test_pv_S / num_test_batches
    test_losses.append(epoch_test_loss)
    # 调节罚函数系数
    eps = 1e-8

    # 保存最佳模型（根据测试集损失） 这里记录罚函数系数在变 只记录罚函数没有的
    if epoch_test_obj + epoch_test_penalty < best_test_loss:
        best_test_loss = epoch_test_obj + epoch_test_penalty
        best_model_state = model.state_dict().copy()
        no_improve_count = 0  # 重置计数
        
        os.makedirs("NN_parameter", exist_ok=True)
        model_save_path = f"NN_parameter/penalty_model_{HIDDEN_DIMS}_{BATCH_SIZE}_{LEARNING_RATE}.pth"
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
        
    # if (epoch + 1) % 5 == 0:
    #     # k_v = max(min(k_obj * sum_obj / (sum_voltage + eps), k_bar) ,k_min)
    #     # k_l = max(min(k_obj * sum_obj / (sum_line + eps), k_bar) ,k_min)
    #     # k_pv_P = max(min(k_obj * sum_obj / (sum_pv_P + eps), k_bar) ,k_min)
    #     # k_pv_S = max(min(k_obj * sum_obj / (sum_pv_S + eps), k_bar) ,k_min)
    #     k_v = min(k_obj * sum_obj / (sum_voltage + eps), k_bar)
    #     k_l = min(k_obj * sum_obj / (sum_line + eps), k_bar) 
    #     k_pv_P = min(k_obj * sum_obj / (sum_pv_P + eps), k_bar) 
    #     k_pv_S = min(k_obj * sum_obj / (sum_pv_S + eps), k_bar) 
    if (epoch + 1) % 1 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] | "
              f"Train: Total={epoch_train_loss:.4f} Obj={epoch_obj_loss:.4f} Penalty={epoch_train_penalty_loss:.4f} "
              f"(V={epoch_voltage_penalty:.4f} L={epoch_line_penalty:.4f} P={epoch_pv_penalty_P:.4f} S={epoch_pv_penalty_S:.4f}) | "
              f"Test: Total={epoch_test_loss:.4f} Obj={epoch_test_obj:.4f} Penalty={epoch_test_penalty:.4f} "
              f"(V={epoch_test_voltage:.4f} L={epoch_test_line:.4f} P={epoch_test_pv_P:.4f} S={epoch_test_pv_S:.4f}) | "
              f"K=[{k_v:.3f}, {k_l:.3f}, {k_pv_P:.3f}, {k_pv_S:.3f}]"
              f' | Time: {time.time() - time_start_epoch:.2f}s')

training_time = time.time() - start_time
print(f"\n训练完成! 总耗时: {training_time:.2f} 秒")
print(f"最佳测试 Loss: {best_test_loss:.6f}")


# 绘制训练曲线
fig, ax = plt.subplots(figsize=(10, 6))
ax.semilogy(train_losses, label='Train Loss')
ax.semilogy(penalty_losses, label='Penalty Loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE Loss (log scale)')
ax.set_title('Training Curve')
ax.legend()
ax.grid(True)
plt.savefig('Test_result/USL_penalty_curve.png', dpi=300, bbox_inches='tight')
print(f"训练曲线已保存到: Test_result/USL_train_penalty_curve.png")

print("\n" + "="*70)
print("训练完成!")