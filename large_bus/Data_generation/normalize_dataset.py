"""
数据标准化脚本
使用 sklearn StandardScaler 对 dataset_supervised.npz 进行标准化
并划分训练集、验证集和测试集（8:1:1）

输入:
    dataset_supervised.npz (X, Y)
    
输出:
    dataset_split.npz (包含划分好的训练集、验证集和测试集，原始空间+标准化空间)
    scaler_params.npz (X_mean, X_scale, Y_mean, Y_scale)

使用方法:
    python normalize_dataset.py
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
import os

# 获取脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

print("="*70)
print("数据标准化与数据集划分")
print("="*70)

# 加载原始数据集
data_path = os.path.join(SCRIPT_DIR, "dataset_supervised.npz")
print(f"加载数据集: {data_path}")

data = np.load(data_path)
X_raw = data['X']
Y_raw = data['Y']

n_samples = X_raw.shape[0]
print(f"样本数: {n_samples}")
print(f"X 维度: {X_raw.shape}")
print(f"Y 维度: {Y_raw.shape}")

# ==============================================================================
# 1. 划分训练集、验证集和测试集（在标准化前划分，确保测试集和验证集不泄露）
# ==============================================================================
print("\n" + "="*70)
print("划分训练集、验证集和测试集 (5:1:1)")
print("="*70)

train_ratio = 0.5
val_ratio = 0.1
test_ratio = 0.1

n_train = int(train_ratio/(train_ratio + val_ratio + test_ratio) * n_samples)
n_val = int(val_ratio/(train_ratio + val_ratio + test_ratio) * n_samples)
# 测试集取剩余样本（避免舍入误差）
n_test = n_samples - n_train - n_val

# 固定随机种子，确保可重复
np.random.seed(2)
indices = np.random.permutation(n_samples)
train_indices = indices[:n_train]
val_indices = indices[n_train:n_train+n_val]
test_indices = indices[n_train+n_val:]

# 划分原始数据
X_train_raw = X_raw[train_indices]
Y_train_raw = Y_raw[train_indices]
X_val_raw = X_raw[val_indices]
Y_val_raw = Y_raw[val_indices]
X_test_raw = X_raw[test_indices]
Y_test_raw = Y_raw[test_indices]

print(f"训练集: {n_train} 样本 (80%)")
print(f"验证集: {n_val} 样本 (10%)")
print(f"测试集: {n_test} 样本 (10%)")

# 保存划分索引
indices_save_path = os.path.join(SCRIPT_DIR, "split_indices.npz")
np.savez(indices_save_path, 
         train_indices=train_indices, 
         val_indices=val_indices,
         test_indices=test_indices,
         n_train=n_train,
         n_val=n_val,
         n_test=n_test)
print(f"划分索引已保存到: {indices_save_path}")

# ==============================================================================
# 2. 标准化（在训练集上 fit，然后 transform 训练集、验证集和测试集）
# ==============================================================================
print("\n" + "="*70)
print("数据标准化")
print("="*70)

# 创建标准化器
X_scaler = StandardScaler()
Y_scaler = StandardScaler()

# 在训练集上 fit
print("在训练集上拟合标准化器...")
X_scaler.fit(X_train_raw)
Y_scaler.fit(Y_train_raw)

# transform 训练集、验证集和测试集
X_train_norm = X_scaler.transform(X_train_raw)
Y_train_norm = Y_scaler.transform(Y_train_raw)
X_val_norm = X_scaler.transform(X_val_raw)
Y_val_norm = Y_scaler.transform(Y_val_raw)
X_test_norm = X_scaler.transform(X_test_raw)
Y_test_norm = Y_scaler.transform(Y_test_raw)

print(f"\n标准化后统计:")
print(f"  X_train: 均值={np.mean(X_train_norm):.6f}, 标准差={np.std(X_train_norm):.6f}")
print(f"  Y_train: 均值={np.mean(Y_train_norm):.6f}, 标准差={np.std(Y_train_norm):.6f}")
print(f"  X_val:   均值={np.mean(X_val_norm):.6f}, 标准差={np.std(X_val_norm):.6f}")
print(f"  Y_val:   均值={np.mean(Y_val_norm):.6f}, 标准差={np.std(Y_val_norm):.6f}")
print(f"  X_test:  均值={np.mean(X_test_norm):.6f}, 标准差={np.std(X_test_norm):.6f}")
print(f"  Y_test:  均值={np.mean(Y_test_norm):.6f}, 标准差={np.std(Y_test_norm):.6f}")

# 保存标准化参数
scaler_params = {
    'X_mean': X_scaler.mean_,
    'X_scale': X_scaler.scale_,
    'Y_mean': Y_scaler.mean_,
    'Y_scale': Y_scaler.scale_,
}
scaler_path = os.path.join(SCRIPT_DIR, "scaler_params.npz")
np.savez(scaler_path, **scaler_params)
print(f"\n标准化参数已保存到: {scaler_path}")

# 保存划分后的数据集（包含原始空间和标准化空间）
output_path = os.path.join(SCRIPT_DIR, "dataset_split.npz")
np.savez(output_path,
         # 训练集（原始空间）
         X_train_raw=X_train_raw, Y_train_raw=Y_train_raw,
         # 训练集（标准化空间）
         X_train_norm=X_train_norm, Y_train_norm=Y_train_norm,
         # 验证集（原始空间）
         X_val_raw=X_val_raw, Y_val_raw=Y_val_raw,
         # 验证集（标准化空间）
         X_val_norm=X_val_norm, Y_val_norm=Y_val_norm,
         # 测试集（原始空间）
         X_test_raw=X_test_raw, Y_test_raw=Y_test_raw,
         # 测试集（标准化空间）
         X_test_norm=X_test_norm, Y_test_norm=Y_test_norm)
print(f"划分后的数据集已保存到: {output_path}")

print("\n" + "="*70)
print("标准化与数据集划分完成!")
print("="*70)
print("\n输出文件:")
print(f"  1. dataset_split.npz - 包含训练集、验证集和测试集（原始+标准化）")
print(f"  2. scaler_params.npz - 标准化参数")
print(f"  3. split_indices.npz - 划分索引")
print("\n使用示例:")
print("  # 加载训练数据")
print("  data = np.load('Data_generation/dataset_split.npz')")
print("  X_train = data['X_train_norm']  # 或 X_train_raw")
print("  Y_train = data['Y_train_norm']  # 或 Y_train_raw")
print("  X_val   = data['X_val_norm']    # 或 X_val_raw")
print("  Y_val   = data['Y_val_norm']    # 或 Y_val_raw")
print("  X_test  = data['X_test_norm']   # 或 X_test_raw")
print("  Y_test  = data['Y_test_norm']   # 或 Y_test_raw")
print("="*70)
