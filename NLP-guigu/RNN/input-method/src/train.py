import time

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from dataset import get_dataloader
from model import InputMethodModel
import config
from tokenizer import JiebaTokenizer

def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    """
    训练一个轮次
    :param model: 模型
    :param dataloader: 数据集
    :param loss_fn: 损失函数
    :param optimizer: 优化器
    :param device: 设备
    :return: 当前epoch的平均loss
    """
    # 训练模式不同于测试模式，会有Dropout/BatchNorm之类的
    model.train()
    total_loss = 0
    for inputs, targets in tqdm(dataloader, desc='开始训练'):
        # DataLoader 在底层是通过实现 Python 的 迭代器协议 来工作的
        # 每次循环inputs和targets都会是一个batch的数据
        # 如果 inputs 不是 Long（比如是 float），就会报错（或行为不符合预期）
        inputs = inputs.to(device).long()
        # inputs.shape = [batch_size,seq_len]
        targets = targets.to(device).long()
        # targets.shape = [batch_size]

        # 前向传播
        outputs = model(inputs)
        # outputs.shape [batch_size,vocab_size]
        loss = loss_fn(outputs, targets)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def train():
    # 1.确定设备
    device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu'
    )

    # 2.数据集
    dataloader = get_dataloader()

    # 3.加载词表
    tokenizer = JiebaTokenizer.from_vocab(config.MODELS_DIR / "vocab.txt")

    # 4.模型
    model = InputMethodModel(vocab_size=tokenizer.vocab_size).to(device)

    # 5.损失函数
    loss_fn = torch.nn.CrossEntropyLoss()

    # 6.优化器
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
    )
    # 7. tensorboard writer
    writer = SummaryWriter(
        log_dir=config.LOGS_DIR /time.strftime("%Y-%M-%D_%H-%M-%S")
    )

    # 开始训练
    best_loss = float('inf')  # 正无穷
    for epoch in range(1, 1 + config.EPOCHS):
        print('=' * 10, f" Epoch : {epoch}", '=' * 10)
        # 训练一个epoch的逻辑
        loss = train_one_epoch(
            model,
            dataloader,
            loss_fn,
            optimizer,
            device,
        )
        print(f"loss:{loss}")
        # 记录训练结果
        writer.add_scalar('loss', loss, epoch)

        # 保存模型
        if loss < best_loss:
            best_loss = loss
            torch.save(
                model.state_dict(),
                config.MODELS_DIR / 'best.pth',
            )
            print("模型保存成功")

    writer.close()


if __name__ == '__main__':
    train()
