import config
import torch
from model import InputMethodModel
from dataset import get_dataloader
from predict import predict_batch


def evaluate(model, test_dataloader, device):
    top1_acc_count = 0
    top5_acc_count = 0
    total_count = 0
    for inputs, target in test_dataloader:
        inputs = inputs.to(device)
        # inputs_shape :[batch_size,seq_len]
        targets = target.tolist()
        # targets_shape:[batch_size] e.g.[1,3,5] 假设batch_size = 3
        top5_indexes_list = predict_batch(model, inputs)
        # top5_indexes_list.shape:[batch_size,5]  e.g. [[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]]
        for target, top5_indexes in zip(targets, top5_indexes_list):
            total_count += 1
            if target == top5_indexes[0]:
                top1_acc_count += 1
            if target in top5_indexes:
                top5_acc_count += 1

    return top1_acc_count / total_count, top5_acc_count / total_count


def run_evaluate():
    # 使用Top1和Top5准确率进行评估
    # =====资源准备=====
    # 1.确定设备
    device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu'
    )

    # 2.词表
    with open(config.MODELS_DIR / 'vocab.txt', encoding='utf-8') as f:
        # 一次读取一行，但会读入\n
        vocab_list = [line.strip() for line in f.readlines()]
    word2index = {word: index for index, word in enumerate(vocab_list)}
    index2word = {index: word for index, word in enumerate(vocab_list)}

    # 3.模型
    model = InputMethodModel(vocab_size=len(vocab_list)).to(device)
    model.load_state_dict(torch.load((config.MODELS_DIR / 'best.pth')))

    # 4.数据集
    test_dataloader = get_dataloader(train=False)

    # 5.评估逻辑
    top1_acc, top5_acc = evaluate(model, test_dataloader, device)
    print("评估结果")
    print(f"top1_acc：{top1_acc}")
    print(f"top5_acc：{top5_acc}")


if __name__ == '__main__':
    run_evaluate()
