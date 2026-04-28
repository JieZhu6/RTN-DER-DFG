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
res_b_33 = np.load("33_bus/Test_result/results_B_NN.npz")
res_o_33 = np.load("33_bus/Test_result/results_O_NN.npz")

res_b_129 = np.load("129_bus/Test_result/results_B_NN.npz")
res_o_129 = np.load("129_bus/Test_result/results_O_NN.npz")

# 不可行解的投影时间
feasible_before_b_33 = res_b_33['feasible_before']
feasible_before_o_33 = res_o_33['feasible_before']

proj_time_b_33 = res_b_33['projection_times'][~feasible_before_b_33]
proj_time_o_33 = res_o_33['projection_times'][~feasible_before_o_33]
print(np.mean(proj_time_b_33), np.mean(proj_time_o_33))

feasible_before_b_129 = res_b_129['feasible_before']
feasible_before_o_129 = res_o_129['feasible_before']
proj_time_b_129 = res_b_129['projection_times'][~feasible_before_b_129]
proj_time_o_129 = res_o_129['projection_times'][~feasible_before_o_129]

# ==============================================================================
# 4. 画图: 一行两列箱线图
# ==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.5))

# --- 左图: 最优性比例 ---
ax1 = axes[0]
bp1 = ax1.boxplot(
    [proj_time_b_33 / 1e-3, proj_time_o_33 / 1e-3],
    tick_labels=['B-NN', 'O-NN'],
    patch_artist=True,
    widths=0.5,
    showfliers=True,
    flierprops=dict(marker='o', markersize=3, alpha=0.5)
)
colors = ["#8EAFCF", '#FFBE7A']
for patch, color in zip(bp1['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax1.set_xlabel('(a) 33-bus System', fontsize=14)
ax1.set_ylabel('Projection Time (×10⁻³ s)', fontsize=14)
ax1.grid(axis='y', linestyle='--', alpha=0.5)

# --- 右图: 不可行解投影时间 ---
ax2 = axes[1]
bp2 = ax2.boxplot(
    [proj_time_b_129 / 1e-3, proj_time_o_129 / 1e-3],
    tick_labels=['B-NN', 'O-NN'],
    patch_artist=True,
    widths=0.5,
    showfliers=True,
    flierprops=dict(marker='o', markersize=3, alpha=0.5)
)
for patch, color in zip(bp2['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax2.set_xlabel('(b) 129-bus System', fontsize=14)
# ax2.set_ylabel('Projection Time (×10⁻³ s)', fontsize=14)
ax2.grid(axis='y', linestyle='--', alpha=0.5)

# 统一两个子图的纵坐标范围
ymin = min(ax1.get_ylim()[0], ax2.get_ylim()[0])
ymax = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
ax1.set_ylim(ymin, ymax)
ax2.set_ylim(ymin, ymax)

plt.tight_layout()
plt.savefig('compare_oproj_bisection.png', dpi=300, bbox_inches='tight')
print("\n图片已保存到: compare_oproj_bisection.png")
plt.show()
