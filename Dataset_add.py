import os
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset


# raw_path,ref_path对应文件名称集合.txt文件
class MyDataset(Dataset):
    def __init__(self, raw_path, ref_path, normalize=True):
        super(MyDataset, self).__init__()
        self.normalize = normalize
        self.raw_root, self.ref_root = raw_path, ref_path
        f, f1 = open(self.raw_root, 'r'), open(self.ref_root, 'r')
        data, data1 = f.readlines(), f1.readlines()
        raw_signals, ref_signals = [], []

        for line in data:
            word = line.rstrip()
            raw_signals.append(word)

        for line1 in data1:
            word1 = line1.rstrip()
            ref_signals.append(word1)

        self.raw_signals = raw_signals
        self.ref_signals = ref_signals

    def __len__(self):
        return len(self.raw_signals)
        return len(self.ref_signals)

    def __getitem__(self, item):
        raw_signal = self.raw_signals[item]
        ref_signal = self.ref_signals[item]
        raw, label = np.loadtxt(raw_signal), np.loadtxt(ref_signal)
        raw, label = torch.from_numpy(raw), torch.from_numpy(label)
        #归一化
        if self.normalize:
            temp_raw, temp_label = raw.reshape(len(raw), 1, 1), label.reshape(len(label), 1, 1)
            norm_raw = transforms.Normalize(mean=torch.mean(temp_raw), std=torch.std(temp_raw))(temp_raw)
            norm_label = transforms.Normalize(mean=torch.mean(temp_label), std=torch.std(temp_label)+1e-8)(temp_label)
            raw, label = norm_raw[:, :, 0].T, norm_label[:, :, 0].T
            raw, label = raw.float(), label.float()
        return raw, label
