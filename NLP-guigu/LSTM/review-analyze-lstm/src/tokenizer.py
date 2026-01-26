import config
import jieba
from datasets import tqdm


class JiebaTokenizer:
    """
    封装对词表的操作
    """
    # 在init外的是类属性
    unk_token = '<unk>'
    pad_token = '<pad>'

    def __init__(self, vocab_list):
        # 实例属性
        self.vocab_list = vocab_list
        self.vocab_size = len(vocab_list)
        self.word2index = {
            word: index for index, word in enumerate(vocab_list)
        }
        self.index2word = {
            index: word for index, word in enumerate(vocab_list)
        }
        # 因为是自己维护的词表，所以不使用get
        # self.unk_token_index = self.word2index.get(self.unk_token, 0)
        self.unk_token_index = self.word2index[self.unk_token]
        self.pad_token_index = self.word2index[self.pad_token]

    @staticmethod
    # @staticmethod 装饰器
    def tokenize(text):
        return jieba.lcut(text)

    def encode(self, text, seq_len):
        tokens = self.tokenize(text)
        # 截取 和 填充 到指定的长度
        if len(tokens)  > seq_len:
            tokens = tokens[:seq_len]
        elif len(tokens) < seq_len:
            tokens = tokens + [self.pad_token] * (seq_len - len(tokens))

        return [self.word2index.get(token, self.unk_token_index) for token in tokens]

    @classmethod
    def build_vocab(cls, sentences, vocab_path) -> None:
        vocab_set = set()
        # tqdm用于长内容加载时的进度条
        for sentence in tqdm(sentences, desc='构建词表'):
            # 使用set去重，但是无法保证有序
            vocab_set.update(jieba.lcut(sentence))

        # 转化成list，就可以让数据有序，同时set无法通过索引便利
        vocab_list = [cls.pad_token, cls.unk_token] + [token for token in vocab_set if token.strip() != ' ']
        print(f"此表大小{len(vocab_list)}")
        print(vocab_list[0:10])
        # 5.保存词表
        with open(vocab_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(vocab_list))

            print("数据处理完成")

    @classmethod
    def from_vocab(cls, vocab_path):
        with open(vocab_path, encoding='utf-8') as f:
            vocab_list = [readline.strip() for readline in f.readlines()]

        # 构建出一个对象
        # 相当于 instance = JiebaTokenizer(vocab_list)
        return cls(vocab_list)


if __name__ == '__main__':
    tokenizers = JiebaTokenizer.from_vocab(config.MODELS_DIR / "vocab.txt")
    print(f'此表大小：{tokenizers.vocab_size}')
    print(f'特殊符号：{tokenizers.unk_token}')
    print(tokenizers.encode("集团年天气不错"))
