import config
import torch
from model import ReviewAnalyzeModel
from dataset import get_dataloader
from predict import predict_batch
from tokenizer import JiebaTokenizer


def evaluate(model, test_dataloader, device):
    correct_count = 0
    total_count = 0
    for inputs, target in test_dataloader:
        inputs = inputs.to(device)
        # inputs_shape :[batch_size,seq_len]
        targets = target.tolist()
        # targets_shape:[batch_size] e.g.[0,1,0,1]
        batch_result = predict_batch(model, inputs)
        # top5_indexes_list.shape:[batch_size]  e.g. [0.1.0.2.0.9.]
        """
        当多个可迭代对象的长度不同时，zip() 会停止于最短的那个对象，忽略掉较长对象中多余的部分。
        """
        for result, target in zip(batch_result, targets):
            result = 1 if result > 0.5 else 0
            if result == target:
                correct_count += 1
            total_count += 1

    return correct_count / total_count


def run_evaluate():
    # 使用Top1和Top5准确率进行评估
    # =====资源准备=====
    # 1.确定设备
    device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu'
    )

    # 2.词表
    tokenizer = JiebaTokenizer.from_vocab(config.MODELS_DIR / "vocab.txt")

    # 3.模型
    model = ReviewAnalyzeModel(
        vocab_size=tokenizer.vocab_size,
        padding_index=tokenizer.pad_token_index,
    ).to(device)
    model.load_state_dict(torch.load((config.MODELS_DIR / 'best.pt')))

    # 4.数据集
    test_dataloader = get_dataloader(train=False)

    # 5.评估逻辑
    acc = evaluate(model, test_dataloader, device)
    print("评估结果")
    print(f"acc：{acc}")


if __name__ == '__main__':
    run_evaluate()
