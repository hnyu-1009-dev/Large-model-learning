import torch
import numpy as np
from torch import nn

# 1. 字符输入
text = "hey how are you"
# 2.
# 3. 数据划分
input_seq = []
output_seq = []
window = 5
for i in range(0, len(text) - window, 1):
    input_seq.append(text[i:i + window])
    output_seq.append(text[i + window])
print("input_seq:", input_seq)
print("output_seq:", output_seq)
# 4. 数据编码：one-hot
chars = set(text)
chars = sorted(chars)
print("chars", chars)
char2int = {char: ind for ind, char in enumerate(chars)}
print("char2int:", char2int)
int2char = dict(enumerate(chars))
print("int2char:", int2char)
# 将字符转换成数字编码
input_seq = [[char2int[char] for char in seq] for seq in input_seq]
output_seq = [char2int[char] for char in output_seq]
# output_seq = [[char2int[char] for char in seq] for seq in output_seq]
print("input_seq:", input_seq)
print("output_seq:", output_seq)
# one hot 编码
features = np.zeros(
    (len(input_seq), len(chars)),
    dtype=np.float32,
)
for i, seq in enumerate(input_seq):
    features[i, seq] = 1.0
input_seq = torch.tensor(features, dtype=torch.float32)
features = np.zeros(
    (len(output_seq), len(chars)),
    dtype=np.float32
)
for i, seq in enumerate(output_seq):
    features[i, seq] = 1.0
print(features)
output_seq = torch.tensor(features, dtype=torch.float32)
print(output_seq)


# 5. 定义前向模型
class Model(nn.Module):
    def __init__(self, input_size, hidden_size, out_size):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = Model(len(chars), 32, len(chars))
# 6. 定义损失函数和优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
# 7. 开始迭代
epochs = 1000
for epoch in range(1, epochs + 1):
    output = model(input_seq)
    loss = criterion(output, output_seq)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # 8. 显示频率设置
    if epoch == 0 or epochs % 50 == 0:
        print(f"Epoch[{epoch}/{epochs}],loss{loss:.4f}")


# 5.数据编码，将输入和输出数据进行one-hot编码
def one_hot_encode(sequence, dict_size):
    # 先申请一个全零的列为dict_size 行为sequence长度的全零矩阵
    features = np.zeros((len(sequence), dict_size), dtype=np.float32)
    # 对应的位置置为1，其他位置不变,且将句子中的字符进行合并操作
    for i, seq in enumerate(sequence):
        features[i, seq] = 1.0
    return features


# 定义预测函数
def predict_next_char(model, input_text):
    input_seq = [char2int[char] for char in input_text]
    input_seq = torch.from_numpy(one_hot_encode([input_seq], len(chars))).float()
    output = model(input_seq)
    predicted_char_index = torch.argmax(output).item()
    predicted_char = int2char[predicted_char_index]
    return predicted_char


# 预测下一个字符
input_text = "ey ho"
# 预测 结果是w
predicted_char = predict_next_char(model, input_text)
print(f'Input: {input_text}')
print(f'Output: {predicted_char}')
