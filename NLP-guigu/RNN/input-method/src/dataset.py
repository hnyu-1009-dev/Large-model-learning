# 1.定义 Dataset
import pandas as pd
import config
import torch
from torch.utils.data import Dataset, DataLoader


class InputMethodDataset(Dataset):
    def __init__(self, path):
        # 根据指定路径，读入数据，为dataFrame数据
        self.data = pd.read_json(
            path, lines=True, orient='records',
        ).to_dict(
            orient='records',
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_tensor = torch.tensor(
            self.data[index]['input'],
            dtype=torch.long,  # = torch.int64
        )

        target_tensor = torch.tensor(
            self.data[index]['target'],
            dtype=torch.long,
        )

        return input_tensor, target_tensor


# 2.提供一个获取dataloader的方法
def get_dataloader(train=True):
    path = config.PROCESSED_DATA_DIR / ('train.jsonl' if train else 'train.jsonl')
    dataset = InputMethodDataset(path)
    return DataLoader(dataset=dataset, batch_size=config.BATCH_SIZE, shuffle=True)


if __name__ == '__main__':
    train_dataloader = get_dataloader()
    test_dataloader = get_dataloader(train=False)
    print(len(train_dataloader))
    print(len(test_dataloader))

    for input_tensor, target_tensor in train_dataloader:
        print(input_tensor.shape)
        print(target_tensor.shape)
        break
