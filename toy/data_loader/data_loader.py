import numpy as np
import pickle
from torch.utils.data import Dataset

def read_pickle(name):
    with open(name, 'rb') as f:
        data = pickle.load(f)
    return data

# Toy Dataset
class ToyDataset(Dataset):
    def __init__(self, pkl, domain_id):
        idx = pkl["domain"] == domain_id
        self.data = pkl["data"][idx].astype(np.float32)
        self.label = pkl["label"][idx].astype(np.int64)
        self.domain = domain_id

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx], self.domain

    def __len__(self):
        return len(self.data)

class SeqToyDataset(Dataset):
    def __init__(self, datasets, size=3*200):
        self.datasets = datasets
        self.size = size
        print(
            "SeqDataset Size {} SubDataset Size {}".format(
                size, [len(ds) for ds in datasets]
            )
        )

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        return [ds[i] for ds in self.datasets]