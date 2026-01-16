import torch
from sympy.codegen.fnodes import cmplx

import config
import jieba
from model import InputMethodModel


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
        # output.shape:[batch_size, vocab_size]
    top5_indexes = torch.topk(output, k=5).indices  # dim默认维度为-1，即选取最后一维vocab_size，每次从vocab_size这一维度中选取数值
    # top5_indexes.shape :[batch_size,5]

    top5_indexes_list = top5_indexes.tolist()

    return top5_indexes_list


def predict(text, model, device, word2index, index2word):
    # 4.处理输入
    tokens = jieba.lcut(text)
    indexes = [word2index.get(token, 0) for token in tokens]
    input_tensor = torch.tensor([indexes], dtype=torch.long).to(device)

    # 5.预测逻辑
    top5_indexes_list = predict_batch(model, input_tensor)
    top5_tokens = [index2word[index] for index in top5_indexes_list[0]]
    return top5_tokens


def run_predict():
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
    print("hello(输入q或quit退出)")
    while True:
        user_input = input("》输入。。。")
        input_history = ''
        if user_input in ['q', 'quit']:
            break
        if user_input.strip() == '':
            print("hello")
            continue
        input_history += user_input
        print(f'输入历史：{input_history}')
        top5_tokens = predict(input_history, model, device, word2index, index2word)
        print(f'预测结果：{top5_tokens}')


if __name__ == '__main__':
    # top_tokens = predict("我们团队")
    # print(top_tokens)
    run_predict()
