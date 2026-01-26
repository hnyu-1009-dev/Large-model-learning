"""

"""
import math

import torch

torch.set_printoptions(sci_mode=False)
torch.manual_seed(42)
seq_len = 10
dk = 512

Q = torch.randn([seq_len, dk], requires_grad=True)
K = torch.randn([seq_len, dk], requires_grad=True)

scores = Q @ K.T / math.sqrt(dk)
# dim=-1 表示沿第 -1 维做 softmax（也就是固定其他维，只变化列）
wights = torch.softmax(scores, dim=-1)
print(wights)

loss = wights[0, 0] - 0

loss.backward()
print(Q.grad.norm().item())
