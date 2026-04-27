
# 验证仿射内点的可行性

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from System_data.system_config import case, Y_bus_matrix, PV_bus_define
import numpy as np
import matplotlib.pyplot as plt
from NN_Model.powerflow_env import run_powerflow_pytorch_batched
np.random.seed(2)
plt.rcParams.update({'font.size': 16})
plt.rc('font', family='Times New Roman')

# ==============================================================================
# 1. 加载系统配置和测试数据
# ==============================================================================
TEST_DATA = np.load("Data_generation/dataset.npy")
data = np.load("Data_generation/dataset_split.npz")
TEST_DATA = data['X_val_raw']
N_TEST_SAMPLES = TEST_DATA.shape[0]
print(f"Loaded test dataset: {N_TEST_SAMPLES} samples")

SYSTEM_DATA = case()
branch = np.array(SYSTEM_DATA['branch'])
bus = np.array(SYSTEM_DATA['bus'])
PV_bus, PV_capacity = PV_bus_define()
R_ij_matrix, X_ij_matrix, r_x_ratio, branch_max = Y_bus_matrix()

n_bus = bus.shape[0]
n_branches = branch.shape[0]
n_pv = len(PV_bus)
print(f"System: {n_bus} Buses, {n_branches} Branches, {n_pv} PVs")

V_MAX, V_MIN = 1.05, 0.95
S_BASE_MVA = 10.0
V_BASE_KV = 12.66

ACTIVE_LOAD_ALL = TEST_DATA[:, :n_bus].T
REACTIVE_LOAD_ALL = TEST_DATA[:, n_bus:2*n_bus].T
PV_P_POWER_ALL = TEST_DATA[:, 2*n_bus:2*n_bus+n_pv]


# ==============================================================================
# 2. 加载仿射系数
# ==============================================================================
try:
    affine_coef = np.load('System_data/robust_affine_coefficients.npz')
    M_yPVq = affine_coef['M_yPVq']
    m_yPVq = affine_coef['m_yPVq']
    M_yPVp = affine_coef['M_yPVp']
    m_yPVp = affine_coef['m_yPVp']
    print(f"Loaded affine coefficients:")
    print(f"  M_yPVq shape: {M_yPVq.shape}, m_yPVq shape: {m_yPVq.shape}")
    print(f"  M_yPVp shape: {M_yPVp.shape}, m_yPVp shape: {m_yPVp.shape}")
except FileNotFoundError:
    print("Error: Affine coefficients not found. Run Robust_Inner_convex_Distflow.py first.")
    sys.exit(1)

N_X = 2 * n_branches + n_pv
print(f"Uncertainty dimension N_X: {N_X}")

# ==============================================================================
# 3. 使用仿射系数计算所有测试样本的 PV 出力
# ==============================================================================
y_PVq_all = np.zeros((N_TEST_SAMPLES, n_pv))
y_PVp_all = np.zeros((N_TEST_SAMPLES, n_pv))

for i in range(N_TEST_SAMPLES):
    x = np.concatenate([
        ACTIVE_LOAD_ALL[1:, i],
        REACTIVE_LOAD_ALL[1:, i],
        PV_P_POWER_ALL[i, :]
    ])
    y_PVq_all[i, :] = M_yPVq @ x + m_yPVq
    y_PVp_all[i, :] = M_yPVp @ x + m_yPVp

print(f"\nCompleted affine mapping for {N_TEST_SAMPLES} samples")

V_sq_nn, l_sq_nn, _, _ = run_powerflow_pytorch_batched(ACTIVE_LOAD_ALL.T, REACTIVE_LOAD_ALL.T, y_PVp_all, y_PVq_all, PV_bus)
if np.max(V_sq_nn) <= V_MAX**2 and np.min(V_sq_nn) >= V_MIN**2:
    print("pass voltage limits")
if np.max(l_sq_nn) < branch_max**2:
    print("pass ")