import torch
import numpy as np
from torch.utils.data import Dataset


class DataSet(Dataset):
    def __init__(self, X, y):
        self.len = X.shape[0]
        self.x_data = torch.FloatTensor(X)
        self.y_data = torch.LongTensor(y)

    def __getitem__(self, index):

        return self.x_data[index], self.y_data[index]

    def __len__(self):

        return self.len
