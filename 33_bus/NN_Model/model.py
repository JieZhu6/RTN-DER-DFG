import torch
import torch.nn as nn

class DERDispatchNet(nn.Module):
    def __init__(self, n_bus, n_pv, hidden_dims=[512, 512, 256]):
        super().__init__()
        self.n_bus = n_bus
        self.n_pv = n_pv
        
        input_dim = n_bus * 2 + n_pv
        output_dim = n_pv * 2
        
        # 纯 MLP，无 BN，无 Dropout，无 Sigmoid
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.ReLU(),
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
        
        # Kaiming 初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        raw_output = self.net(x)
        # 4. 分离并约束 P_pv 到 [0, 1]（弃光系数）
        P_pv_raw = raw_output[:, :self.n_pv]
        Q_pv = raw_output[:, self.n_pv:]
        P_pv = torch.sigmoid(P_pv_raw)  # 弃光系数范围 [0, 1]
        f_NN = torch.cat([P_pv, Q_pv], dim=1)
        return f_NN

