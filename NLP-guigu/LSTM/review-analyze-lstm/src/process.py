import pandas as pd
from sklearn.model_selection import train_test_split
from tokenizer import JiebaTokenizer

import config


def process():
    print("开始处理数据")
    # 读取文件
    df = pd.read_csv(
        config.RAW_DATA_DIR / 'online_shopping_10_cats.csv',
        usecols=['label', 'review'],  # 标记使用的列
        encoding='utf-8'
    ).dropna()  # 去除空值

    # 划分数据集
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df['label'],  # 分层抽样，保证在训练集和测试集中的正负情感样本分布相同
    )
    # 基于训练集构建词表
    JiebaTokenizer.build_vocab(
        train_df['review'].tolist(),  # 构建词表的原始句子列表
        config.MODELS_DIR / 'vocab.txt',
    )
    # 创建Tokenizer
    tokenizer = JiebaTokenizer.from_vocab(
        config.MODELS_DIR / 'vocab.txt',
    )
    # 计算序列长度
    # print(train_df['review'].apply(lambda x: len(tokenizer.tokenize(x))).quantile(0.95))

    # 构建训练集
    train_df['review'] = train_df['review'].apply(
        lambda x: tokenizer.encode(x, config.SEQ_LEN))  # 接收一个函数，并将函数应用到前面的数据,并返回新的数据
    # 保存训练集 保存成jsonl数据格式
    train_df.to_json(
        config.PROCESSED_DATA_DIR / 'train.jsonl',
        orient='records',
        lines=True,
    )
    # 构建测试集
    test_df['review'] = test_df['review'].apply(
        lambda x: tokenizer.encode(x, config.SEQ_LEN))  # 接收一个函数，并将函数应用到前面的数据,并返回新的数据

    # 保存测试集
    test_df.to_json(
        config.PROCESSED_DATA_DIR / 'test.jsonl',
        orient='records',
        lines=True,
    )
    print(df.head())
    print('数据处理完成')


if __name__ == '__main__':
    process()
