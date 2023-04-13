from torch import tensor
from torch.utils.data import Dataset as TDataset


class myDataset(TDataset):
    def __init__(self, encodings, labels, tokenizer):
        self.encodings = tokenizer(list(encodings), truncation=True, padding=True)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
