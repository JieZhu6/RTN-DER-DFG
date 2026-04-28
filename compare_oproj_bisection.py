"""
对比 NN_penalty_oproj 与 NN_bisection 方法的最优性比例和不可行解投影时间

图示要求:
    1) 一行两列的图
    2) 左图: 箱线图对比最优性比例 (objective / optimal)
    3) 右图: 箱线图对比不可行解的求解(投影)时间

使用方法:
    python compare_oproj_bisection.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt


plt.rcParams.update({'font.size': 14})
plt.rc('font', family='Times New Roman')



# ==============================================================================
# 3. 加载两种方法的结果
# ==============================================================================
res_b = np.load("Test_result/results_B_NN.npz")
res_o = np.load("Test_result/results_O_NN.npz")
label_objective_values = res_o['label_objective_values']  # 两个文件中 label_objective_values 是一样的
# 最优性比例: method_obj / optimal_obj  (越接近 1 越好)
obj_b = res_b['objective_values']
obj_o = res_o['objective_values']
ratio_b = (obj_b - label_objective_values) / label_objective_values * 100
ratio_o = (obj_o - label_objective_values) / label_objective_values * 100

# 不可行解的投影时间
feasible_before_b = res_b['feasible_before']
feasible_before_o = res_o['feasible_before']

proj_time_b = res_b['projection_times'][~feasible_before_b]
proj_time_o = res_o['projection_times'][~feasible_before_o]

print("=" * 70)
print("对比统计")
print("=" * 70)
print(f"Bisection  不可行样本数: {np.sum(~feasible_before_b)}")
print(f"Oproj      不可行样本数: {np.sum(~feasible_before_o)}")
print(f"\n最优性比例 (method / optimal):")
print(f"  Bisection  mean={np.mean(ratio_b):.6f}, median={np.median(ratio_b):.6f}")
print(f"  Oproj       mean={np.mean(ratio_o):.6f}, median={np.median(ratio_o):.6f}")
print(f"\n不可行解投影时间 (s):")
print(f"  Bisection  mean={np.mean(proj_time_b):.4f}, median={np.median(proj_time_b):.4f}")
print(f"  Oproj       mean={np.mean(proj_time_o):.4f}, median={np.median(proj_time_o):.4f}")

# ==============================================================================
# 4. 画图: 一行两列箱线图
# ==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# --- 左图: 最优性比例 ---
ax1 = axes[0]
bp1 = ax1.boxplot(
    [ratio_b, ratio_o],
    tick_labels=['Bisection', 'Oproj'],
    patch_artist=True,
    widths=0.5,
    showfliers=True,
    flierprops=dict(marker='o', markersize=3, alpha=0.5)
)
colors = ['#8ECFC9', '#FFBE7A']
for patch, color in zip(bp1['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax1.set_ylabel('Optimality Ratio', fontsize=14)
ax1.set_title('(a) Optimality Ratio Comparison', fontsize=14)
ax1.axhline(1.0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Optimal')
ax1.legend(loc='upper left', fontsize=11)
ax1.grid(axis='y', linestyle='--', alpha=0.5)
# ax1.set_ylim(0.98, 1.3)

# --- 右图: 不可行解投影时间 ---
ax2 = axes[1]
bp2 = ax2.boxplot(
    [proj_time_b, proj_time_o],
    tick_labels=['Bisection', 'Oproj'],
    patch_artist=True,
    widths=0.5,
    showfliers=True,
    flierprops=dict(marker='o', markersize=3, alpha=0.5)
)
for patch, color in zip(bp2['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax2.set_ylabel('Projection Time (s)', fontsize=14)
ax2.set_title('(b) Infeasible Sample Projection Time', fontsize=14)
ax2.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('Test_result/compare_oproj_bisection.png', dpi=300, bbox_inches='tight')
print("\n图片已保存到: Test_result/compare_oproj_bisection.png")
plt.show()
