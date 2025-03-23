import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.ReLU(),
            nn.Linear(4 * dim, dim)
        )

    def forward(self, x):
        return self.net(x)


class MoE(nn.Module):
    def __init__(self, num_experts=4, dim=512, top_k=2):
        super().__init__()
        self.experts = nn.ModuleList([Expert(dim) for _ in range(num_experts)])
        self.gate = nn.Linear(dim, num_experts)
        self.top_k = top_k

    def forward(self, x):
        # 路由计算
        gate_logits = self.gate(x)  # [batch_size, num_experts]
        weights = F.softmax(gate_logits, dim=-1)
        top_weights, top_indices = torch.topk(weights, self.top_k, dim=-1)  # [batch_size, top_k]

        # 稀疏激活专家
        output = torch.zeros_like(x)
        for i in range(self.top_k):
            expert_idx = top_indices[:, i]  # [batch_size]
            expert_weight = top_weights[:, i].unsqueeze(-1)  # [batch_size, 1]

            # 对每个样本选择对应的专家
            expert_output = torch.stack([
                self.experts[idx](x[batch_idx].unsqueeze(0))  # 选择专家并计算输出
                for batch_idx, idx in enumerate(expert_idx)
            ]).squeeze(1)  # [batch_size, dim]

            # 加权求和
            output += expert_weight * expert_output
        return output


# 使用示例
moe = MoE(num_experts=4, dim=512, top_k=2)
x = torch.randn(32, 512)  # 假设输入维度为512，批量大小32
out = moe(x)  # 仅激活2个专家
print(out.shape)  # 输出形状应为 [32, 512]