"""
生成并保存 pandapower 网络基础拓扑

此脚本创建 IEEE 33 节点配电网的 pandapower 网络模型，
只包含基础拓扑（母线、线路、外部电网连接），
不包含负荷和PV等时变元件。

生成的网络可以供其他程序加载使用，避免重复建模。
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pandapower as pp
# pp.to_json and pp.from_json are used directly from the pandapower module
import os

# 导入系统配置
from System_data.system_config import case, Y_bus_matrix, PV_bus_define


def create_network():
    """
    创建节点配电网的 pandapower 网络
    
    返回:
        net: pandapower 网络对象（只包含基础拓扑，不含负荷和PV）
    """
    # 获取系统数据
    system_data = case()
    branch = np.array(system_data['branch'])
    bus = np.array(system_data['bus'])
    
    # 获取阻抗参数
    R_ij_matrix, X_ij_matrix, r_x_ratio, branch_max = Y_bus_matrix()
    
    # 系统基准参数
    S_BASE_MVA = 10  # 基准功率 MVA
    V_BASE_KV = 12.66  # 基准电压 kV
    
    # 创建空网络
    net = pp.create_empty_network(sn_mva=S_BASE_MVA, name="IEEE 33 Bus Distribution System")
    
    # 计算阻抗基准值 (ohm)
    Z_base = V_BASE_KV**2 / S_BASE_MVA
    
    # 计算最大电流 (kA)，用于线路参数
    max_i_ka = branch_max * S_BASE_MVA / (np.sqrt(3) * V_BASE_KV)
    
    # ========== 添加母线 ==========
    for i in range(bus.shape[0]):
        pp.create_bus(
            net, 
            vn_kv=V_BASE_KV, 
            name=f"Bus {i+1}",
            type="b" if i > 0 else "n"  # 节点1为平衡节点
        )
    
    # ========== 添加外部电网 (平衡节点) ==========
    pp.create_ext_grid(
        net, 
        bus=0,  # 节点1 (索引0)
        vm_pu=1.0,  # 电压幅值 1.0 p.u.
        va_degree=0.0,  # 电压相角 0度
        name="Grid Connection"
    )
    
    # ========== 添加线路 ==========
    # 注意：branch 数据中的阻抗是标幺值，需要应用 r_x_ratio 缩放
    for i in range(branch.shape[0]):
        from_bus = int(branch[i, 0]) - 1  # 转换为0-based索引
        to_bus = int(branch[i, 1]) - 1
        
        # 读取标幺值阻抗并应用缩放
        r_pu = branch[i, 2] * r_x_ratio
        x_pu = branch[i, 3] * r_x_ratio
        
        # 转换为实际阻抗 (ohm)，假设线路长度为1km
        r_ohm = r_pu * Z_base
        x_ohm = x_pu * Z_base
        
        # 创建线路
        pp.create_line_from_parameters(
            net,
            from_bus=from_bus,
            to_bus=to_bus,
            length_km=1.0,  # 标准化长度为1km
            r_ohm_per_km=r_ohm,
            x_ohm_per_km=x_ohm,
            c_nf_per_km=0,  # 忽略对地电容
            max_i_ka=max_i_ka,
            name=f"Line {int(branch[i, 0])}-{int(branch[i, 1])}"
        )
    
    # 存储系统参数到网络的自定义属性中，方便后续使用
    net._system_params = {
        'S_BASE_MVA': S_BASE_MVA,
        'V_BASE_KV': V_BASE_KV,
        'Z_base': Z_base,
        'r_x_ratio': r_x_ratio,
        'branch_max': branch_max,
        'n_bus': bus.shape[0],
        'n_branch': branch.shape[0],
        'PV_bus': PV_bus_define()[0],
        'PV_capacity': PV_bus_define()[1]
    }
    
    print(f"网络创建完成:")
    print(f"  - 母线数量: {len(net.bus)}")
    print(f"  - 线路数量: {len(net.line)}")
    print(f"  - 外部电网: {len(net.ext_grid)}")
    print(f"  - 基准功率: {S_BASE_MVA} MVA")
    print(f"  - 基准电压: {V_BASE_KV} kV")
    print(f"  - 阻抗基准: {Z_base:.4f} ohm")
    
    return net


def save_network(net, filepath=None):
    """
    保存 pandapower 网络到文件
    
    参数:
        net: pandapower 网络对象
        filepath: 保存路径，默认为 System_data/ieee_network.json
    """
    if filepath is None:
        filepath = os.path.join(os.path.dirname(__file__), "ieee_network.json")
    
    # 使用 pandapower 的 to_json 函数保存
    pp.to_json(net, filepath)
    
    print(f"\n网络已保存到: {filepath}")
    
    # 同时保存一个二进制版本（加载更快）
    pkl_path = filepath.replace('.json', '.pkl')
    import pickle
    with open(pkl_path, 'wb') as f:
        pickle.dump(net, f)
    print(f"网络已保存到: {pkl_path}")
    
    return filepath


def load_network(filepath=None, use_pickle=True):
    """
    加载 pandapower 网络
    
    参数:
        filepath: 网络文件路径，默认为 System_data/ieee33bus_network.pkl 或 .json
        use_pickle: 是否优先使用 pickle 格式（加载更快）
    
    返回:
        net: pandapower 网络对象
    """
    if filepath is None:
        base_path = os.path.join(os.path.dirname(__file__), "ieee_network")
        pkl_path = base_path + ".pkl"
        json_path = base_path + ".json"
        
        if use_pickle and os.path.exists(pkl_path):
            filepath = pkl_path
        elif os.path.exists(json_path):
            filepath = json_path
        else:
            raise FileNotFoundError(
                f"找不到网络文件，请先运行 create_33bus_network() 和 save_network() 创建网络。\n"
                f"查找路径: {pkl_path} 或 {json_path}"
            )
    
    # 根据文件扩展名选择加载方式
    if filepath.endswith('.pkl'):
        import pickle
        with open(filepath, 'rb') as f:
            net = pickle.load(f)
    else:
        net = pp.from_json(filepath)
    
    return net


def get_system_params(net):
    """
    获取网络的系统参数
    
    参数:
        net: pandapower 网络对象
    
    返回:
        dict: 系统参数字典
    """
    if hasattr(net, '_system_params'):
        return net._system_params
    else:
        # 如果没有存储参数，返回默认参数
        return {
            'S_BASE_MVA': 10,
            'V_BASE_KV': 12.66,
            'Z_base': 12.66**2 / 10,
            'r_x_ratio': 0.01,
            'branch_max': 2,
            'n_bus': 33,
            'n_branch': 32,
            'PV_bus': [7, 15, 18, 22, 25, 27, 33],
            'PV_capacity': 1.2
        }


def add_loads_to_network(net, active_load_pu, reactive_load_pu):
    """
    向网络添加负荷
    
    参数:
        net: pandapower 网络对象
        active_load_pu: 有功负荷，单位 p.u. (长度为 n_bus 的数组)
        reactive_load_pu: 无功负荷，单位 p.u. (长度为 n_bus 的数组)
    
    返回:
        net: 更新后的网络对象
    """
    params = get_system_params(net)
    S_BASE = params['S_BASE_MVA']
    
    # 清除现有负荷
    if len(net.load) > 0:
        net.load.drop(net.load.index, inplace=True)
    
    # 添加新负荷
    for i in range(len(active_load_pu)):
        p_mw = active_load_pu[i] * S_BASE
        q_mvar = reactive_load_pu[i] * S_BASE
        
        if abs(p_mw) > 1e-9 or abs(q_mvar) > 1e-9:
            pp.create_load(
                net,
                bus=i,
                p_mw=p_mw,
                q_mvar=q_mvar,
                name=f"Load {i+1}"
            )
    
    return net


def add_pv_to_network(net, pv_bus_indices, pv_p_pu, pv_q_pu):
    """
    向网络添加 PV 发电单元
    
    参数:
        net: pandapower 网络对象
        pv_bus_indices: PV 连接的节点编号列表 (1-based)
        pv_p_pu: PV 有功出力，单位 p.u. (与 pv_bus 等长)
        pv_q_pu: PV 无功出力，单位 p.u. (与 pv_bus 等长)
    
    返回:
        net: 更新后的网络对象
    """
    params = get_system_params(net)
    S_BASE = params['S_BASE_MVA']
    
    # 清除现有 sgen (分布式电源)
    if len(net.sgen) > 0:
        net.sgen.drop(net.sgen.index, inplace=True)
    
    # 添加 PV
    for i, bus_idx in enumerate(pv_bus_indices):
        bus_0based = bus_idx - 1  # 转换为 0-based 索引
        p_mw = pv_p_pu[i] * S_BASE
        q_mvar = pv_q_pu[i] * S_BASE
        
        pp.create_sgen(
            net,
            bus=bus_0based,
            p_mw=p_mw,
            q_mvar=q_mvar,
            name=f"PV Bus{bus_idx}"
        )
    
    return net


def clear_dynamic_elements(net):
    """
    清除网络中的动态元件（负荷和PV），保留基础拓扑
    
    参数:
        net: pandapower 网络对象
    
    返回:
        net: 更新后的网络对象
    """
    # 清除负荷
    if len(net.load) > 0:
        net.load.drop(net.load.index, inplace=True)
    
    # 清除分布式电源 (sgen)
    if len(net.sgen) > 0:
        net.sgen.drop(net.sgen.index, inplace=True)
    
    # 清除发电机 (gen)
    if len(net.gen) > 0:
        net.gen.drop(net.gen.index, inplace=True)
    
    # 清除储能 (storage)
    if len(net.storage) > 0:
        net.storage.drop(net.storage.index, inplace=True)
    
    return net


def run_power_flow(net, algorithm='nr', init='dc', max_iteration=50):
    """
    运行潮流计算，带错误处理
    
    参数:
        net: pandapower 网络对象
        algorithm: 算法类型，默认 'nr' (Newton-Raphson)
        init: 初始化方法，默认 'dc' (DC潮流)
        max_iteration: 最大迭代次数
    
    返回:
        success: 是否收敛
        net: 更新后的网络对象
    """
    try:
        pp.runpp(net, algorithm=algorithm, init=init, max_iteration=max_iteration)
        return True, net
    except pp.powerflow.LoadflowNotConverged:
        # 尝试平坦启动
        try:
            pp.runpp(net, algorithm=algorithm, init='flat', max_iteration=100)
            return True, net
        except pp.powerflow.LoadflowNotConverged:
            return False, net


if __name__ == "__main__":
    # 创建网络
    print("=" * 60)
    print("创建 I配电网 pandapower 模型")
    print("=" * 60)
    
    net = create_network()
    
    # 保存网络
    save_network(net)
    
    # 测试加载
    print("\n测试加载网络...")
    net_loaded = load_network()
    print(f"加载成功!")
    print(f"  - 母线数量: {len(net_loaded.bus)}")
    print(f"  - 线路数量: {len(net_loaded.line)}")
    
    # 显示系统参数
    params = get_system_params(net_loaded)
    print(f"\n系统参数:")
    for key, value in params.items():
        print(f"  - {key}: {value}")
    
    print("\n" + "=" * 60)
    print("网络创建和保存完成!")
    print("其他程序可以使用 load_network() 加载此网络")
    print("=" * 60)
