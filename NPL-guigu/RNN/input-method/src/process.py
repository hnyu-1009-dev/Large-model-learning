import jieba
import pandas as pd
from datasets import tqdm
from sklearn.model_selection import train_test_split

import config


def process():
    print("开始处理数据")
    # 1.读取文件
    df = pd.read_json(
        config.RAW_DATA_DIR / "synthesized_.jsonl",
        lines=True,
        orient="records"
    ).sample(frac=0.1, random_state=42)  # 测试只抽取百分之一
    print(df.head())

    # 2.提取句子
    sentences = []
    for dialog in df['dialog']:
        for sentence in dialog:
            sentences.append(sentence.split('：')[1])
    print(sentences[0:10])
    print(f'句子总数：{len(sentences)}')

    # 3.划分数据集
    train_sentences, test_sentences = train_test_split(sentences, test_size=0.2, random_state=42)

    # 4.构建词表
    vocab_set = set()
    for sentence in tqdm(train_sentences, desc='构建词表'):
        # 使用set去重，但是无法保证有序
        vocab_set.update(jieba.lcut(sentence))

    # 转化成list，就可以让数据有序，同时set无法通过索引便利
    vocab_list = ['<unk>'] + list(vocab_set)
    print(f"此表大小{len(vocab_list)}")
    print(vocab_list[0:10])
    # 5.保存词表
    with open(config.MODELS_DIR / 'vocab.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(vocab_list))

        print("数据处理完成")

    # 6.构建训练集


    # 7.保存训练集


if __name__ == '__main__':
    process()
