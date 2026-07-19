import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
import warnings
# 抑制 pandapower 内部的数值警告（如除零）
# warnings.filterwarnings('ignore', category=RuntimeWarning, module='pandapower')
# warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')
import pandapower as pp
import torch
from System_data.create_pandapower_network import load_network
from System_data.system_config import Y_bus_matrix, case

# 系统基准电压
V_BASE_KV = 12.66  # kV
s_base_mva = 10.0
R_ij_matrix, X_ij_matrix, r_x_ratio, branch_max = Y_bus_matrix()

# 获取系统拓扑信息
SYSTEM_DATA = case()
branch = np.array(SYSTEM_DATA['branch'])
bus = np.array(SYSTEM_DATA['bus'])
n_bus_total = bus.shape[0]
n_branch_total = branch.shape[0]
branch_list = branch[:, :4]
branch_list[:,2:4] = branch[:, 2:4]* r_x_ratio
R_vector = branch[:, 2] * r_x_ratio  # 调整后的电阻值，单位 p.u.，长度为 n_branches
X_vector = branch[:, 3] * r_x_ratio  # 调整后的电抗值，单位 p.u.，长度为 n_branches

Feasibility_tolerance = 1e-5  # 可行性检查的数值容差

import torch
import numpy as np

def verify_powerflow_solvers(P_load, Q_load, P_pv, Q_pv, pv_bus):
    """
    验证 PyTorch 批处理潮流求解器与 PandaPower 的结果是否一致
    """
    print("="*50)
    print("开始验证潮流求解器精度...")
    print("="*50)
    
    batch_size = P_load.shape[0]
    device = P_load.device
    
    # ---------------------------------------------------------
    # 方法 1: 使用 PandaPower (基准线 Ground Truth)
    # ---------------------------------------------------------
    P_load_np = P_load.detach().cpu().numpy()
    Q_load_np = Q_load.detach().cpu().numpy()
    P_pv_np = P_pv.detach().cpu().numpy()
    Q_pv_np = Q_pv.detach().cpu().numpy()
    
    V_sq_list, l_sq_list, P_branch_list, Q_branch_list = [], [], [], []
    
    for b in range(batch_size):
        # 调用你原来的 pandapower 包装函数
        V_sq_b, l_sq_b, P_branch_b, Q_branch_b = run_powerflow_pandapower(
            P_load_np[b], Q_load_np[b], 
            P_pv_np[b], Q_pv_np[b], 
            pv_bus
        )
        V_sq_list.append(V_sq_b)
        l_sq_list.append(l_sq_b)
        P_branch_list.append(P_branch_b)
        Q_branch_list.append(Q_branch_b)
        
    # 转换为 PyTorch Tensor 以便比对
    V_sq_pp = torch.tensor(np.array(V_sq_list), dtype=torch.float32, device=device)
    l_sq_pp = torch.tensor(np.array(l_sq_list), dtype=torch.float32, device=device)
    P_branch_pp = torch.tensor(np.array(P_branch_list), dtype=torch.float32, device=device)
    Q_branch_pp = torch.tensor(np.array(Q_branch_list), dtype=torch.float32, device=device)

    # ---------------------------------------------------------
    # 方法 2: 使用自定义的 PyTorch 并行前推回代法
    # ---------------------------------------------------------
    V_sq_pt, l_sq_pt, P_branch_pt, Q_branch_pt = run_powerflow_pytorch_batched(
        P_load, Q_load, P_pv, Q_pv, pv_bus
    )
    
    # ---------------------------------------------------------
    # 计算并打印最大绝对误差 (Max Absolute Error)
    # ---------------------------------------------------------
    diff_V = torch.max(torch.abs(V_sq_pp - V_sq_pt)).item()
    diff_l = torch.max(torch.abs(l_sq_pp - l_sq_pt)).item()
    diff_P = torch.max(torch.abs(P_branch_pp - P_branch_pt)).item()
    diff_Q = torch.max(torch.abs(Q_branch_pp - Q_branch_pt)).item()
    
    print(f"最大电压平方误差 (V_sq)     : {diff_V:.6e} pu")
    print(f"最大电流平方误差 (l_sq)     : {diff_l:.6e} pu")
    print(f"最大有功支路潮流误差 (P_br) : {diff_P:.6e} pu")
    print(f"最大无功支路潮流误差 (Q_br) : {diff_Q:.6e} pu")
    print("="*50)
    
    # 设置一个合理的容差阈值 (比如 1e-4)
    tolerance = 1e-4
    if diff_V < tolerance and diff_l < tolerance and diff_P < tolerance and diff_Q < tolerance:
        print("✅ 验证通过！自定义 PyTorch 求解器与 PandaPower 结果高度一致！")
        return True
    else:
        print("❌ 验证失败！结果存在较大差异，请检查基准容量、单位转换或拓扑参数。")
        return False

# 测试调用示例 (你需要传入你的实际数据 tensor)
# is_match = verify_powerflow_solvers(P_load_batch, Q_load_batch, P_pv_batch, Q_pv_batch, pv_bus, branch_list)

def get_topology_info(n_bus):
    """
    辅助函数：只在初始化时运行一次，预处理拓扑，生成网络前推回代的安全顺序
    建议在 DifferentiablePowerFlow 的外部预处理好，或者在 __init__ 中保存，避免每次 forward 都重新算
    """
    n_branch = len(branch_list)
    out_branches = {i: [] for i in range(n_bus)} 
    in_degree = {i: 0 for i in range(n_bus)}     
    branch_info = {}                             
    
    for b_idx, (f_bus, t_bus, r, x) in enumerate(branch_list):
        f_bus_idx = int(f_bus) - 1
        t_bus_idx = int(t_bus) - 1
        out_branches[f_bus_idx].append(b_idx)
        branch_info[b_idx] = (f_bus_idx, t_bus_idx, r, x)
        in_degree[t_bus_idx] += 1
        
    root_nodes = [i for i in range(n_bus) if in_degree[i] == 0]
    
    topo_order = []  
    queue = root_nodes.copy()
    
    # BFS 生成拓扑排序
    while queue:
        curr_bus = queue.pop(0)
        for child_b_idx in out_branches[curr_bus]:
            topo_order.append(child_b_idx)
            _, next_bus, _, _ = branch_info[child_b_idx]
            queue.append(next_bus)
            
    reverse_topo_order = list(reversed(topo_order))
    return topo_order, reverse_topo_order, out_branches, branch_info

topo_order, reverse_topo_order, out_branches, branch_info = get_topology_info(n_bus_total)

def run_powerflow_pytorch_batched(P_load, Q_load, P_pv, Q_pv, pv_bus, max_iter=50, tol = Feasibility_tolerance):
    """
    支持 Numpy/Tensor 混合输入的纯 PyTorch 并行潮流求解器。
    如果输入是 numpy 数组，会自动转换为 Tensor 计算，并在返回时转回 numpy。
    """
    # ==========================================
    # 1. 格式检验与转换 (Numpy -> Tensor)
    # ==========================================
    is_numpy = isinstance(P_load, np.ndarray)
    
    if is_numpy:
        # 如果是 Numpy，优先使用 GPU 进行内部计算加速
        compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        P_load_t = torch.tensor(P_load, dtype=torch.float32, device=compute_device)
        Q_load_t = torch.tensor(Q_load, dtype=torch.float32, device=compute_device)
        P_pv_t   = torch.tensor(P_pv, dtype=torch.float32, device=compute_device)
        Q_pv_t   = torch.tensor(Q_pv, dtype=torch.float32, device=compute_device)
    else:
        # 如果已经是 Tensor，直接沿用，并获取当前设备
        P_load_t, Q_load_t, P_pv_t, Q_pv_t = P_load, Q_load, P_pv, Q_pv
        compute_device = P_load_t.device
        
    batch_size = P_load_t.shape[0]
    n_bus = P_load_t.shape[1]
    
    # 注意：这里默认 branch_list, branch_info 等拓扑变量是你脚本里的全局变量
    n_branch = len(branch_list) 
    dtype = P_load_t.dtype
    
    # ==========================================
    # 2. 计算各节点的净消耗功率
    # ==========================================
    P_net = P_load_t.clone()
    Q_net = Q_load_t.clone()
    for i, bus_idx in enumerate(pv_bus):
        idx_0 = int(bus_idx) - 1
        P_net[:, idx_0] -= P_pv_t[:, i]
        Q_net[:, idx_0] -= Q_pv_t[:, i]
        
    # ==========================================
    # 3. 初始化状态变量张量
    # ==========================================
    V_sq = torch.ones((batch_size, n_bus), device=compute_device, dtype=dtype)
    l_sq = torch.zeros((batch_size, n_branch), device=compute_device, dtype=dtype)
    P_branch = torch.zeros((batch_size, n_branch), device=compute_device, dtype=dtype)
    Q_branch = torch.zeros((batch_size, n_branch), device=compute_device, dtype=dtype)
    
    # ==========================================
    # 4. 开始前推回代迭代求解
    # ==========================================
    for iteration in range(max_iter):
        V_sq_prev = V_sq.clone()
        
        # --- 回代过程 (Backward Sweep) ---
        for b_idx in reverse_topo_order:
            f_bus, t_bus, r, x = branch_info[b_idx]
            
            p_val = P_net[:, t_bus] + r * l_sq[:, b_idx]
            q_val = Q_net[:, t_bus] + x * l_sq[:, b_idx]
            
            for child_b_idx in out_branches[t_bus]:
                p_val = p_val + P_branch[:, child_b_idx]
                q_val = q_val + Q_branch[:, child_b_idx]
                
            P_branch[:, b_idx] = p_val
            Q_branch[:, b_idx] = q_val
            
        # --- 前推过程 (Forward Sweep) ---
        for b_idx in topo_order:
            f_bus, t_bus, r, x = branch_info[b_idx]
            
            V_sq[:, t_bus] = V_sq[:, f_bus] - 2.0 * (r * P_branch[:, b_idx] + x * Q_branch[:, b_idx]) \
                             + (r**2 + x**2) * l_sq[:, b_idx]
            
            V_f_clamped = torch.clamp(V_sq[:, f_bus], min=1e-4)
            l_sq[:, b_idx] = (P_branch[:, b_idx]**2 + Q_branch[:, b_idx]**2) / V_f_clamped
            
        # --- 检查收敛性 ---
        max_error = torch.max(torch.abs(V_sq - V_sq_prev))
        if max_error < 1e-4:
            break
            
    if iteration == max_iter - 1 and max_error >= 1e-4:
        print(f"[Warning] Batched Power Flow did not converge! Max error: {max_error.item():.6f}")
        
    # ==========================================
    # 5. 格式还原 (如果是 Numpy 进来，就 Numpy 出去)
    # ==========================================
    if is_numpy:
        V_sq = V_sq.cpu().numpy()
        l_sq = l_sq.cpu().numpy()
        P_branch = P_branch.cpu().numpy()
        Q_branch = Q_branch.cpu().numpy()
        
    return V_sq, l_sq, P_branch, Q_branch

def run_powerflow_numpy_single(P_load, Q_load, P_pv, Q_pv, pv_bus, max_iter=50, tol=Feasibility_tolerance, verify=False):
    """
    纯 NumPy 单样本前推回代潮流求解器。
    输入为一维 numpy 数组，输出也为 numpy 数组，无 PyTorch 依赖。
    
    参数:
        verify: bool - 是否在返回前与 PandaPower 结果进行比对验证（默认 False）
    """
    # ==========================================
    # 1. 计算各节点的净消耗功率
    # ==========================================
    P_net = P_load.copy()
    Q_net = Q_load.copy()
    for i, bus_idx in enumerate(pv_bus):
        idx_0 = int(bus_idx) - 1
        P_net[idx_0] -= P_pv[i]
        Q_net[idx_0] -= Q_pv[i]

    n_bus = P_load.shape[0]
    n_branch = len(branch_list)

    # ==========================================
    # 2. 初始化状态变量
    # ==========================================
    V_sq = np.ones(n_bus, dtype=np.float32)
    l_sq = np.zeros(n_branch, dtype=np.float32)
    P_branch = np.zeros(n_branch, dtype=np.float32)
    Q_branch = np.zeros(n_branch, dtype=np.float32)

    # ==========================================
    # 3. 前推回代迭代求解
    # ==========================================
    for iteration in range(max_iter):
        V_sq_prev = V_sq.copy()

        # --- 回代过程 (Backward Sweep) ---
        for b_idx in reverse_topo_order:
            f_bus, t_bus, r, x = branch_info[b_idx]

            p_val = P_net[t_bus] + r * l_sq[b_idx]
            q_val = Q_net[t_bus] + x * l_sq[b_idx]

            for child_b_idx in out_branches[t_bus]:
                p_val += P_branch[child_b_idx]
                q_val += Q_branch[child_b_idx]

            P_branch[b_idx] = p_val
            Q_branch[b_idx] = q_val

        # --- 前推过程 (Forward Sweep) ---
        for b_idx in topo_order:
            f_bus, t_bus, r, x = branch_info[b_idx]

            V_sq[t_bus] = (V_sq[f_bus]
                           - 2.0 * (r * P_branch[b_idx] + x * Q_branch[b_idx])
                           + (r**2 + x**2) * l_sq[b_idx])

            V_f_clamped = max(V_sq[f_bus], 1e-4)
            l_sq[b_idx] = (P_branch[b_idx]**2 + Q_branch[b_idx]**2) / V_f_clamped

        # --- 检查收敛性 ---
        max_error = np.max(np.abs(V_sq - V_sq_prev))
        if max_error < tol:
            break

    if iteration == max_iter - 1 and max_error >= tol:
        print(f"[Warning] Single-sample Power Flow did not converge! Max error: {max_error:.6f}")

    # ==========================================
    # 4. 可选：与 PandaPower 结果进行比对验证
    # ==========================================
    if verify:
        print("="*50)
        print("开始验证 NumPy 单样本潮流求解器精度...")
        print("="*50)
        
        V_sq_pp, l_sq_pp, P_branch_pp, Q_branch_pp = run_powerflow_pandapower(
            P_load, Q_load, P_pv, Q_pv, pv_bus
        )
        
        diff_V = np.max(np.abs(V_sq_pp - V_sq))
        diff_l = np.max(np.abs(l_sq_pp - l_sq))
        diff_P = np.max(np.abs(P_branch_pp - P_branch))
        diff_Q = np.max(np.abs(Q_branch_pp - Q_branch))
        
        print(f"最大电压平方误差 (V_sq)     : {diff_V:.6e} pu")
        print(f"最大电流平方误差 (l_sq)     : {diff_l:.6e} pu")
        print(f"最大有功支路潮流误差 (P_br) : {diff_P:.6e} pu")
        print(f"最大无功支路潮流误差 (Q_br) : {diff_Q:.6e} pu")
        print("="*50)
        
        tolerance = 1e-4
        if diff_V < tolerance and diff_l < tolerance and diff_P < tolerance and diff_Q < tolerance:
            print("✅ 验证通过！NumPy 单样本求解器与 PandaPower 结果高度一致！")
        else:
            print("❌ 验证失败！结果存在较大差异，请检查算法实现。")
    
    return V_sq, l_sq, P_branch, Q_branch

def check_feasibility(V_sq, l_sq, pv_p, pv_q, p_available, pv_capacity, v_min=0.95, v_max=1.05, tol=Feasibility_tolerance):
    """
    检查约束可行性，自适应支持 batch 输入 (2D) 和单个样本 (1D)
    
    输入:
        - 2D 数组: (batch_size, features) -> 返回 (batch_size,) 的布尔数组
        - 1D 数组: (features,) -> 返回标量 bool
    """
    # 记录原始输入维度，用于后续决定返回格式
    is_single_sample = (V_sq.ndim == 1)
    
    # 统一转换为 2D 处理
    V_sq = np.atleast_2d(V_sq)
    l_sq = np.atleast_2d(l_sq)
    pv_p = np.atleast_2d(pv_p)
    pv_q = np.atleast_2d(pv_q)
    p_available = np.atleast_2d(p_available)
    
    # 1. 逆变器容量约束: P^2 + Q^2 <= S^2
    cap_sq = pv_capacity**2
    S_sq = pv_p**2 + pv_q**2
    cap_valid = np.all(S_sq <= cap_sq + tol, axis=1)
    
    # 2. PV 出力上限: P_pv <= P_available
    pv_ub_valid = np.all(pv_p <= p_available + tol, axis=1)
    
    # 3. 电压约束 (使用平方边界)
    v_min_sq = v_min**2 - tol
    v_max_sq = v_max**2 + tol
    v_valid = np.all((V_sq >= v_min_sq) & (V_sq <= v_max_sq), axis=1)
    
    # 4. 电流约束 (使用平方边界，branch_max 需为标幺值)
    l_max_sq = branch_max**2 + tol
    l_valid = np.all(l_sq <= l_max_sq, axis=1)
    
    # 全部满足才算 Feasible
    is_feasible = cap_valid & pv_ub_valid & v_valid & l_valid
    
    # 单个样本时返回标量 bool，batch 时返回数组
    if is_single_sample:
        return bool(is_feasible[0])
    return is_feasible


def check_feasibility_torch(V_sq, l_sq, pv_p, pv_q, p_available, pv_capacity, v_min=0.95, v_max=1.05, tol=1e-4):
    """
    PyTorch 版本的可行性检查，避免 numpy 转换开销
    
    输入:
        - V_sq: [batch, n_bus] 或 [n_bus] - torch.Tensor
        - l_sq: [batch, n_branch] 或 [n_branch] - torch.Tensor
        - pv_p, pv_q: [batch, n_pv] 或 [n_pv] - torch.Tensor
        - p_available: [batch, n_pv] 或 [n_pv] - torch.Tensor
    输出:
        - is_feasible: [batch] bool tensor 或 标量 bool
    """
    import torch
    
    # 记录是否为单个样本
    is_single_sample = (V_sq.dim() == 1)
    
    # 统一转换为 2D
    if is_single_sample:
        V_sq = V_sq.unsqueeze(0)
        l_sq = l_sq.unsqueeze(0)
        pv_p = pv_p.unsqueeze(0)
        pv_q = pv_q.unsqueeze(0)
        p_available = p_available.unsqueeze(0)
    
    # 1. 逆变器容量约束: P^2 + Q^2 <= S^2
    cap_sq = pv_capacity ** 2
    S_sq = pv_p ** 2 + pv_q ** 2
    cap_valid = torch.all(S_sq <= cap_sq + tol, dim=1)
    
    # 2. PV 出力上限: P_pv <= P_available
    pv_ub_valid = torch.all(pv_p <= p_available + tol, dim=1)
    
    # 3. 电压约束
    v_min_sq = v_min ** 2 - tol
    v_max_sq = v_max ** 2 + tol
    v_valid = torch.all((V_sq >= v_min_sq) & (V_sq <= v_max_sq), dim=1)
    
    # 4. 电流约束
    l_max_sq = branch_max ** 2 + tol
    l_valid = torch.all(l_sq <= l_max_sq, dim=1)
    
    # 全部满足
    is_feasible = cap_valid & pv_ub_valid & v_valid & l_valid
    
    if is_single_sample:
        return bool(is_feasible[0].item())
    return is_feasible

def run_powerflow_pandapower(p_load, q_load, pv_p_actual, pv_q_power, pv_bus):
    """使用 pandapower 运行潮流计算，返回 numpy 数组"""
    net = load_network()
    bus_shape = p_load.shape[0]
    
    for i in range(bus_shape):
        p_mw = p_load[i] * s_base_mva
        q_mvar = q_load[i] * s_base_mva
        if abs(p_mw) > 1e-9 or abs(q_mvar) > 1e-9:
            pp.create_load(net, bus=i, p_mw=p_mw, q_mvar=q_mvar, name=f"Load {i+1}")

    count_PV = 0
    for i in range(bus_shape):
        if np.isin(i+1, pv_bus):            
            p_mw = pv_p_actual[count_PV] * s_base_mva
            q_mvar = pv_q_power[count_PV] * s_base_mva
            pp.create_sgen(net, bus=i, p_mw=p_mw, q_mvar=q_mvar, name=f"PV {i+1}")
            count_PV += 1

    pp.runpp(net, algorithm='bfsw', calculate_voltage_angles=False, init='dc')

    v_pu = net.res_bus.vm_pu.values
    I_base_ka = s_base_mva / (np.sqrt(3) * V_BASE_KV)

    if len(net.res_line) > 0:
        i_from_ka = net.res_line['i_from_ka'].values
        i_to_ka = net.res_line['i_to_ka'].values
        i_actual_ka = np.maximum(i_from_ka, i_to_ka)
        i_pu = i_actual_ka / I_base_ka
        
        # [新增] 读取线路起始端(from_bus)注入的功率，并转换为标幺值
        # 在 DistFlow 模型中，P_ij 代表从节点 i 流向 j 的功率，正好对应 p_from
        p_branch_pu = net.res_line['p_from_mw'].values / s_base_mva
        q_branch_pu = net.res_line['q_from_mvar'].values / s_base_mva
    else:
        i_pu = np.zeros(n_branch_total)
        p_branch_pu = np.zeros(n_branch_total)
        q_branch_pu = np.zeros(n_branch_total)

    return v_pu**2, i_pu**2, p_branch_pu, q_branch_pu


class DifferentiablePowerFlow(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, P_load, Q_load, P_pv, Q_pv, pv_bus):    
        # is_match = verify_powerflow_solvers(P_load, Q_load, P_pv, Q_pv, pv_bus, branch_list) 检查自己写的前推回代法计算是否准确
            
        # 1. 直接输入张量，直接吐出张量，内部全并行，没有 for b in range(batch_size): 循环了！
        V_sq, l_sq, P_branch, Q_branch = run_powerflow_pytorch_batched(
            P_load, Q_load, P_pv, Q_pv, pv_bus
        )
        
        # 2. 保存上下文
        ctx.save_for_backward(P_pv, Q_pv, V_sq, l_sq, P_branch, Q_branch)
        ctx.pv_bus = pv_bus
        ctx.branch_list = branch_list
        
        return V_sq, l_sq
    
    @staticmethod
    def backward(ctx, grad_V_sq, grad_l_sq):
        # [修改] 读取保存的支路功率。注意这里不再需要保存/读取 P_load 和 Q_load，因为不需要重算了
        P_pv, Q_pv, V_sq, l_sq, P_branch, Q_branch = ctx.saved_tensors
        pv_bus = ctx.pv_bus
        branch_list = ctx.branch_list
        
        batch_size = P_pv.shape[0]
        n_pv = P_pv.shape[1]
        n_branch = len(branch_list)
        n_bus = len(V_sq[0])
        n_state = 3 * n_branch + n_bus 
        
        device = grad_V_sq.device
        grad_P = torch.zeros((batch_size, n_branch), device=device)
        grad_Q = torch.zeros((batch_size, n_branch), device=device)
        
        grad_z = torch.cat([grad_P, grad_Q, grad_V_sq, grad_l_sq], dim=1)
        
        # 批量构建雅可比矩阵 (batch_size, n_state, n_state) 和 (batch_size, n_state, 2*n_pv)
        dh_dz, dh_df = build_distflow_jacobians(
            V_sq.detach().cpu().numpy(),
            l_sq.detach().cpu().numpy(),
            P_branch.detach().cpu().numpy(),
            Q_branch.detach().cpu().numpy(),
            pv_bus
        )
        
        dh_dz_tensor = torch.tensor(dh_dz, dtype=V_sq.dtype, device=device)
        dh_df_tensor = torch.tensor(dh_df, dtype=V_sq.dtype, device=device)
        
        try:
            # 批量求解: (batch, n_state, n_state) 和 (batch, n_state, 1) -> (batch, n_state, 1)
            eps = 1e-7 * torch.eye(n_state, device=device).unsqueeze(0).expand(batch_size, -1, -1)
            # grad_z: (batch, n_state), 需要转置为 (batch, n_state, 1)
            grad_z_t = grad_z.unsqueeze(-1)  # (batch, n_state, 1)
            # 求解 dh_dz^T * lambda = grad_z
            lambda_vec = torch.linalg.solve(dh_dz_tensor.transpose(-2, -1) + eps, grad_z_t)  # (batch, n_state, 1)
            # 计算 grad_f = -dh_df^T * lambda
            grad_f = -torch.matmul(dh_df_tensor.transpose(-2, -1), lambda_vec).squeeze(-1)  # (batch, 2*n_pv)
            
            grad_P_pv = grad_f[:, :n_pv]
            grad_Q_pv = grad_f[:, n_pv:]
        except RuntimeError as e:
            print(f"[WARNING] Batch Jacobian Solver failed: {e}")
            # 失败时回退到逐个处理
            grad_P_pv = torch.zeros_like(P_pv)
            grad_Q_pv = torch.zeros_like(Q_pv)
            for b in range(batch_size):
                try:
                    dh_dz_b = dh_dz_tensor[b]
                    dh_df_b = dh_df_tensor[b]
                    eps = 1e-7 * torch.eye(n_state, device=device)
                    lambda_vec = torch.linalg.solve(dh_dz_b.T + eps, grad_z[b].unsqueeze(-1)).squeeze(-1)
                    grad_f = -torch.matmul(dh_df_b.T, lambda_vec)
                    grad_P_pv[b] = grad_f[:n_pv]
                    grad_Q_pv[b] = grad_f[n_pv:]
                except RuntimeError as e2:
                    print(f"[WARNING] Individual solver failed for batch {b}: {e2}")
                
        # 返回 None 的数量严格匹配 forward 的输入参数数量
        return None, None, grad_P_pv, grad_Q_pv, None

def build_distflow_jacobians(V_sq, l_sq, P_branch, Q_branch, pv_bus):
    """
    接收直接由前向计算好的 P_branch 和 Q_branch 构建精确雅可比矩阵
    (已修正符号，严格遵循标准 DistFlow 物理定律)
    
    支持批量输入: 
    - 单样本: V_sq (n_bus,), l_sq (n_branch,), P_branch (n_branch,), Q_branch (n_branch,)
    - 批量: V_sq (batch, n_bus), l_sq (batch, n_branch), P_branch (batch, n_branch), Q_branch (batch, n_branch)
    """
    # 判断是否为批量输入
    is_batched = V_sq.ndim == 2
    
    if is_batched:
        batch_size = V_sq.shape[0]
        n_bus = V_sq.shape[1]
    else:
        batch_size = 1
        n_bus = len(V_sq)
        # 扩展为批量维度以便统一处理
        V_sq = V_sq[np.newaxis, :]
        l_sq = l_sq[np.newaxis, :]
        P_branch = P_branch[np.newaxis, :]
        Q_branch = Q_branch[np.newaxis, :]
    
    n_branch = len(branch_list)
    n_pv = len(pv_bus)
    
    n_state = 3 * n_branch + n_bus 
    n_var = 2 * n_pv
    
    # 初始化批量雅可比矩阵
    dh_dz = np.zeros((batch_size, n_state, n_state))
    dh_df = np.zeros((batch_size, n_state, n_var))
    
    # 雅各比矩阵的列索引
    idx_P = 0
    idx_Q = n_branch
    idx_V = 2 * n_branch
    idx_l = 2 * n_branch + n_bus

    # 雅各比矩阵的行索引
    row_V0 = 0                   
    row_P = 1                    
    row_Q = 1 + n_branch         
    row_V = 1 + 2 * n_branch     
    row_l = 1 + 3 * n_branch     
    
    # 所有 batch 共享的常数部分
    dh_dz[:, row_V0, idx_V + 0] = 1.0 
    
    # 预处理子节点映射字典 (所有 batch 共享相同拓扑)
    children_map = {i: [] for i in range(n_bus)}
    for b_idx, (f_bus, t_bus, _, _) in enumerate(branch_list):
        children_map[int(f_bus) - 1].append((b_idx, int(t_bus) - 1))
    
    # 处理每条支路
    for b_idx, (f_bus, t_bus, r, x) in enumerate(branch_list):
        f_bus = int(f_bus) - 1
        t_bus = int(t_bus) - 1
        
        # ==========================================
        # 1. 有功平衡方程 (h_P) 修正
        # 公式: P_ij - sum(P_jk) - r_ij * l_ij + P_j^PV - P_j^L = 0
        # ==========================================
        dh_dz[:, row_P + b_idx, idx_P + b_idx] = 1.0                           
        dh_dz[:, row_P + b_idx, idx_l + b_idx] = -r
        for child_b_idx, _ in children_map[t_bus]:
            dh_dz[:, row_P + b_idx, idx_P + child_b_idx] = -1.0                
            
        for pv_i, pv_b in enumerate(pv_bus):
            if (int(pv_b) - 1) == t_bus:
                dh_df[:, row_P + b_idx, pv_i] = 1.0
                
        # ==========================================
        # 2. 无功平衡方程 (h_Q) 修正
        # 公式: Q_ij - sum(Q_jk) - x_ij * l_ij + Q_j^PV - Q_j^L = 0
        # ==========================================
        dh_dz[:, row_Q + b_idx, idx_Q + b_idx] = 1.0                           
        dh_dz[:, row_Q + b_idx, idx_l + b_idx] = -x
        for child_b_idx, _ in children_map[t_bus]:
            dh_dz[:, row_Q + b_idx, idx_Q + child_b_idx] = -1.0                
            
        for pv_i, pv_b in enumerate(pv_bus):
            if (int(pv_b) - 1) == t_bus:
                dh_df[:, row_Q + b_idx, n_pv + pv_i] = 1.0
                
        # ==========================================
        # 3. 电压降落方程 (h_V) 修正
        # 公式: V_j - V_i + 2(r P_ij + x Q_ij) - (r^2+x^2) l_ij = 0
        # ==========================================
        dh_dz[:, row_V + b_idx, idx_V + t_bus] = 1.0                           
        dh_dz[:, row_V + b_idx, idx_V + f_bus] = -1.0                          
        dh_dz[:, row_V + b_idx, idx_P + b_idx] = 2.0 * r
        dh_dz[:, row_V + b_idx, idx_Q + b_idx] = 2.0 * x
        dh_dz[:, row_V + b_idx, idx_l + b_idx] = -(r**2 + x**2)
        
        # ==========================================
        # 4. 支路电流方程 (h_l)
        # 公式: l_ij V_i - P_ij^2 - Q_ij^2 = 0
        # 注意: V_sq[f_bus] 和 l_sq[b_idx] 是 batch 相关的
        # ==========================================
        dh_dz[:, row_l + b_idx, idx_l + b_idx] = V_sq[:, f_bus]                   
        dh_dz[:, row_l + b_idx, idx_V + f_bus] = l_sq[:, b_idx]                   
        dh_dz[:, row_l + b_idx, idx_P + b_idx] = -2.0 * P_branch[:, b_idx]        
        dh_dz[:, row_l + b_idx, idx_Q + b_idx] = -2.0 * Q_branch[:, b_idx]        

    # 如果不是批量输入，去掉 batch 维度
    if not is_batched:
        dh_dz = dh_dz[0]
        dh_df = dh_df[0]

    return dh_dz, dh_df


def run_differentiable_powerflow(P_load, Q_load, P_pv, Q_pv, pv_bus):
    """可微分潮流计算的便捷函数"""
    return DifferentiablePowerFlow.apply(P_load, Q_load, P_pv, Q_pv, pv_bus)


