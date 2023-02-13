import random
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.utils.rnn as rnn_utils


def collate_fn_atten(batch):
    batch.sort(key=lambda x: x[2], reverse=True)
    seq, labels, seq_length = zip(*batch)
    data = rnn_utils.pad_sequence(seq, batch_first=True, padding_value=0)
    labels = torch.LongTensor(labels)
    return data, labels, seq_length


class RawFeatures(data.Dataset):
    def __init__(self, txt_path):
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            self.feature_list = [i.split()[0] for i in lines]
            self.label_list = [i.split()[1] for i in lines]

    def __getitem__(self, index):
        feature_path = self.feature_list[index]
        feature = torch.from_numpy(np.load(feature_path, allow_pickle=True).T)
        seq_len = feature.size(0)
        label = int(self.label_list[index])

        return feature, label, seq_len

    def __len__(self):
        return len(self.label_list)

def get_atten_mask(seq_lens, batch_size):
    max_len = seq_lens[0]
    atten_mask = torch.ones([batch_size, max_len, max_len])
    for i in range(batch_size):
        length = seq_lens[i]
        atten_mask[i, :length, :length] = 0
    return atten_mask.bool()
