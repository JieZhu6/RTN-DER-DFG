#数据生成模块
#生成可再生能源的数据
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os

# 获取 System_data 目录的绝对路径（脚本所在目录的上级目录）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SYSTEM_DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "System_data")

def case():
    bus = np.array(pd.read_excel(os.path.join(SYSTEM_DATA_DIR, "bus.xlsx")))
    branch = np.array(pd.read_excel(os.path.join(SYSTEM_DATA_DIR, "branch.xlsx")))
    ppc = {"version": '2'}
    ppc["bus"] = bus
    ppc["branch"] = branch
    return ppc

def Y_bus_matrix():
    SYSTEM_DATA = case()
    #节点和系统参数
    branch = np.array(SYSTEM_DATA['branch'])
    bus = np.array(SYSTEM_DATA['bus'])
    #生成阻抗矩阵 
    r_x_ratio = 0.01 # 阻抗调整
    branch_r_x = np.zeros((branch.shape[0],4))
    branch_r_x[:,0:2] = branch [:,0:2]
    branch_r_x[:,2] = branch [:,2]
    branch_r_x[:,3] = branch [:,3]
    branch_r_x[:,2:4] = branch_r_x[:,2:4]*r_x_ratio
    R_ij_matrix = np.inf*np.ones((bus.shape[0],bus.shape[0]))
    X_ij_matrix = np.inf*np.ones((bus.shape[0],bus.shape[0]))
    for i in range(branch_r_x.shape[0]):
        R_ij_matrix[int(branch_r_x[i,0])-1,int(branch_r_x[i,1])-1] = branch_r_x[i,2]
        R_ij_matrix[int(branch_r_x[i,1]-1),int(branch_r_x[i,0])-1] = branch_r_x[i,2]
        X_ij_matrix[int(branch_r_x[i,0]-1),int(branch_r_x[i,1])-1] = branch_r_x[i,3]
        X_ij_matrix[int(branch_r_x[i,1]-1),int(branch_r_x[i,0])-1] = branch_r_x[i,3]
    branch_max = 2 # 定义支路电流上限
    return R_ij_matrix,X_ij_matrix,r_x_ratio,branch_max  

def PV_bus_define():
    PV_bus = [7,15,18,22,25,27,33]
    PV_capacity = 1.2 # 光伏的装机容量
    return PV_bus, PV_capacity

