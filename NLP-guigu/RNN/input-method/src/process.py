import jieba
import pandas as pd
from datasets import tqdm
from sklearn.model_selection import train_test_split

import config


def build_dataset(sentences, word2index, describe: str):
    indexed_sentences = [
        [word2index.get(token, 0) for token in jieba.lcut(sentence)] for sentence in sentences
    ]

    dataset = []
    # [ {'input':[1,2,3,4,5],'target':5}, {'input':[2,3,4,5，6],'target':7} ]
    for sentence in tqdm(indexed_sentences, desc=f'{describe}构建数据集'):
        for i in range(len(sentence) - config.SEQ_LEN):
            input = sentence[i:i + config.SEQ_LEN]
            target = sentence[i + config.SEQ_LEN]
            dataset.append({'input': input, 'target': target})
    print(dataset[0:3])
    return dataset


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
    # tqdm用于长内容加载时的进度条
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
    word2index = {word: index for index, word in enumerate(vocab_list)}

    train_dataset = build_dataset(train_sentences, word2index, 'train_dataset')

    # 7.保存训练集
    pd.DataFrame(train_dataset).to_json(
        config.PROCESSED_DATA_DIR / 'train.jsonl', orient='records', lines=True
    )

    # 8.构建测试集
    test_dataset = build_dataset(test_sentences, word2index, 'test_dataset')

    # 9.保存测试集
    pd.DataFrame(test_dataset).to_json(
        config.PROCESSED_DATA_DIR / 'test.jsonl', orient='records', lines=True
    )


if __name__ == '__main__':
    process()
