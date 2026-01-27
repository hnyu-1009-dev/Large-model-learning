import torch
from torch import nn
import config


class ReviewAnalyzeModel(nn.Module):
    def __init__(self, vocab_size, padding_index):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size,
            config.EMBEDDING_DIM,
            padding_idx=padding_index,  # 记录填充token的索引， 如果不指定会随机初始化 ，指定后前向传播和反向传播的padding位置都不会计算且在embedding中全为0
        )
        """
        self.lstm = nn.LSTM(...) 这行代码是初始化了一个 LSTM 层，
        并将这个 LSTM 层赋值给 self.lstm。
        这个 LSTM 层是一个可调用的对象，也就是说，可以通过 self.lstm 来调用它并传递输入数据进行前向传播。

        在调用 self.lstm 时，默认情况下，只需要传入输入数据（input），
        LSTM 层会自动使用默认的初始隐藏状态和记忆单元（即全零的状态）。
        也可以手动提供初始的隐藏状态和记忆单元（h_0 和 c_0），你也可以通过额外传递这两个参数。
        """
        self.gru = nn.GRU(
            input_size=config.EMBEDDING_DIM,
            hidden_size=config.HIDDEN_SIZE,  # hidden_size 也是隐藏单元的大小
            batch_first=True,
        )
        self.linear = nn.Linear(
            config.HIDDEN_SIZE,
            1,
        )

    def forward(self, x):
        # x.shape: [batch_size, seq_len, embedding]

        embed = self.embedding(x)
        # embed.shape: [batch_size, seq_len, embedding_dim]

        output, _ = self.gru(embed)
        # output.shape[batch_size, seq_len, num_directions*hidden_size]

        # 获取每个样本的最后的真实隐藏状态
        batch_indexes = torch.arange(0, output.shape[0])
        # 如果 output 在 GPU 上，而 torch.arange(...) 默认在 CPU，会报 device mismatch。应写成：
        # batch_indexes = torch.arange(output.size(0), device=output.device)
        """
        (x != self.embedding.padding_idx) 这部分的作用是生成一个布尔矩阵，表示哪些位置是非填充 token。
        
        x != self.embedding.padding_idx：这个表达式会返回一个布尔张量，对于每个 token，如果它不等于填充索引 padding_idx，则该位置为 True，表示它是有效的 token；否则为 False，表示它是填充 token。
        
        sum必须操作布尔值
        """
        lengths = (x != self.embedding.padding_idx).sum(dim=1)
        """
        GPT优化
        batch_idx = torch.arange(x.size(0), device=x.device)
        last_hidden = output[batch_idx, lengths - 1, :]  # [B, H]
        """
        last_hidden = output[batch_indexes, lengths - 1]
        # last_hidden.shape:[batch_size,hidden_size]

        output = self.linear(last_hidden).squeeze(-1)
        # output.shape:[ batch_size]

        return output
