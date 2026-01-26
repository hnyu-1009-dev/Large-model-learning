import torch
from sympy.codegen.fnodes import cmplx

import config
import jieba
from model import ReviewAnalyzeModel
from tokenizer import JiebaTokenizer


def predict_batch(model, inputs):
    """
    批量预测
    :param model: 模型
    :param inputs: 输入：shape [batch_size,seq_len]
    :return: 预测结果，shape[batch_size,5]
    """
    model.eval()
    with torch.no_grad():
        output = model(inputs)
        # output.shape:[batch_size]
        batch_result = torch.sigmoid(output)
    return batch_result


def predict(text, model, device, tokenizer):
    # 4.处理输入
    indexes = tokenizer.encode(text)
    input_tensor = torch.tensor([indexes], dtype=torch.long).to(device)

    # 5.预测逻辑
    top5_indexes_list = predict_batch(model, input_tensor)
    top5_tokens = [tokenizer.index2word[index] for index in top5_indexes_list[0]]
    return top5_tokens


def run_predict():
    # =====资源准备=====
    # 1.确定设备
    device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu'
    )

    # 2.词表
    tokenizer = JiebaTokenizer.from_vocab(config.MODELS_DIR / "vocab.txt")

    # 3.模型
    model = ReviewAnalyzeModel(vocab_size=tokenizer.vocab_size).to(device)
    model.load_state_dict(torch.load((config.MODELS_DIR / 'best.pth')))
    print("欢迎使用情感分析模型(输入q或quit退出)")
    while True:
        user_input = input("》输入。。。")
        if user_input in ['q', 'quit']:
            break
        if user_input.strip() == '':
            print("hello")
            continue
        result = predict(user_input, model, device, tokenizer)
        if result > 0.5:
            print("正向")
        else:
            print("负向")


if __name__ == '__main__':
    run_predict()
