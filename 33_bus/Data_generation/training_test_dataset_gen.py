# 生成测试集和训练集的负荷和光伏数据
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from System_data.system_config import case,PV_bus_define
import numpy as np
import pandas as pd
import os
np.random.seed(2)

# 获取 System_data 目录的绝对路径（脚本所在目录的上级目录）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SYSTEM_DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "System_data")

def res_forecast_data():
    RESDATE = pd.read_excel(os.path.join(SYSTEM_DATA_DIR, "forecastdata.xlsx"))  #可再生能源数据
    return np.array(RESDATE.iloc[3,1:])

def forecast_load():
    # 标准算例中各节点负荷的比例
    bus = np.array(case()['bus'])
    LOADDATE = np.array(pd.read_excel(os.path.join(SYSTEM_DATA_DIR, "baseload.xlsx")))[:,1:]
    load_ratio = bus[:,2]/np.sum(bus[:,2])
    active_load,reactive_load = np.zeros((LOADDATE.shape[1],bus.shape[0])),np.zeros((LOADDATE.shape[1],bus.shape[0]))
    for i in range(LOADDATE.shape[1]):
        active_load[i] = load_ratio*LOADDATE[0][i]
        reactive_load[i] = load_ratio*LOADDATE[1][i]
    rateLOAD = 0.5 #调整负荷水平
    return active_load.T*rateLOAD,reactive_load.T*rateLOAD

# 获取基础负荷数据
ACTIVE_LOAD, REACTIVE_LOAD = np.array(forecast_load())  # 转化为标幺值

# 定义光伏的连接节点
PV_bus,PV_capacity = PV_bus_define()
PV_p_power = res_forecast_data()

# 计算有功无功均值（按节点平均）
scale = 0.8
active_load_mean = np.mean(ACTIVE_LOAD, axis=1) * scale # 每个节点的有功均值
reactive_load_mean = np.mean(REACTIVE_LOAD, axis=1)  * scale  # 每个节点的无功均值
print(np.sum(active_load_mean), np.sum(reactive_load_mean))

# 计算最大出力作为标准值
max_PV_power = np.max(PV_p_power)

# ==================== 数据集生成参数配置 ====================
# 1. 数据集数量
N_SAMPLES = 7000  # 总样本数量

# 3. 负荷采样范围（±25%）
LOAD_SAMPLE_RANGE = 0.25 

# 4. 光伏采样范围
PV_SAMPLE_RANGE = [0.55, 1.05]  # 波动范围25%

def generate_dataset(n_samples=N_SAMPLES):
    n_bus = len(active_load_mean)  # 节点数
    n_pv = len(PV_bus)  # 光伏节点数
    
    # 初始化数据集
    dataset = np.zeros((n_samples, n_bus + n_bus + n_pv))
    
    for i in range(n_samples):
        load_factor = np.random.uniform(
            1 - LOAD_SAMPLE_RANGE, 1 + LOAD_SAMPLE_RANGE, n_bus
        )
        
        # 1. 有功负荷采样：
        active_load_sample = active_load_mean * load_factor
        
        # 2. 无功负荷采样：与有功使用同一因子
        reactive_load_sample = reactive_load_mean * load_factor
        
        # 3. 光伏出力采样：
        # 基于一个共同的出力因子（因为装机容量相同），各节点只有微小差异
        base_pv_factor = np.random.uniform(PV_SAMPLE_RANGE[0], PV_SAMPLE_RANGE[1])
        # 各节点微小扰动
        pv_variation = np.random.uniform(0.98, 1.02, n_pv)
        pv_sample = max_PV_power * base_pv_factor * pv_variation
        
        # 组合成一条样本
        dataset[i, :n_bus] = active_load_sample
        dataset[i, n_bus:2*n_bus] = reactive_load_sample
        dataset[i, 2*n_bus:] = pv_sample
    
    # 随机打乱数据集
    np.random.shuffle(dataset)
    
    return dataset



if __name__ == "__main__":
    import os
    
    # 获取脚本所在目录，确保文件保存在脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 生成数据集
    dataset = generate_dataset(N_SAMPLES)
    
    # 保存数据集
    test_path = os.path.join(script_dir, "dataset.npy")
    np.save(test_path, dataset)
    print(f"数据集已保存到: {test_path}")
