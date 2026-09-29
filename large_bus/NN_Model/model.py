import torch
import torch.nn as nn


class DERDispatchNet(nn.Module):
    """根据 12 时段负荷与可用光伏，联合预测 12 时段 PV 调度。"""

    def __init__(
        self,
        n_bus,
        n_pv,
        hidden_dims=None,
        n_periods=12,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 512, 256]

        self.n_bus = n_bus
        self.n_pv = n_pv
        self.n_periods = n_periods
        self.features_per_period = 2 * n_bus + n_pv
        self.outputs_per_period = 2 * n_pv
        self.input_dim = n_periods * self.features_per_period
        self.output_dim = n_periods * self.outputs_per_period

        layers = []
        prev_dim = self.input_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU()])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, self.output_dim))
        self.net = nn.Sequential(*layers)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        """输入和输出均按时段优先展平。

        输入:  (batch, 12 * (2*n_bus+n_pv))，也接受对应的三维张量。
        输出:  (batch, 12 * 2*n_pv)，每时段依次为 PV 有功比例和无功。
        """
        if x.ndim == 3:
            expected = (self.n_periods, self.features_per_period)
            if tuple(x.shape[1:]) != expected:
                raise ValueError(f"三维输入期望 (*, {expected})，实际 {tuple(x.shape)}")
            x = x.reshape(x.shape[0], -1)
        elif x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(
                f"二维输入期望 (*, {self.input_dim})，实际 {tuple(x.shape)}"
            )

        raw_output = self.net(x).reshape(
            x.shape[0], self.n_periods, self.outputs_per_period
        )
        p_pv_ratio = torch.sigmoid(raw_output[:, :, : self.n_pv])
        q_pv = raw_output[:, :, self.n_pv :]
        dispatch = torch.cat([p_pv_ratio, q_pv], dim=2)
        return dispatch.reshape(x.shape[0], self.output_dim)
