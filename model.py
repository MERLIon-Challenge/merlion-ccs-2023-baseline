import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_util
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from transformer import *
import conformer as cf
from convolution_module import Conv1dSubampling

class Conformer(nn.Module):
    def __init__(self,input_dim, feat_dim, d_k, d_v, n_heads, d_ff, max_len, dropout, device, n_lang):
        super(Conformer, self).__init__()
        self.conv_subsample = Conv1dSubampling(in_channels=input_dim, out_channels= input_dim)
        self.transform = nn.Linear(input_dim, feat_dim)
        self.dropout = nn.Dropout(dropout)
        self.d_model = feat_dim * n_heads
        self.layernorm1 = LayerNorm(feat_dim)
        self.n_heads = n_heads
        self.attention_block1 = cf.ConformerEncoder(self.d_model, d_k, d_v, d_ff, n_heads, dropout, max_len, device)
        self.attention_block2 = cf.ConformerEncoder(self.d_model, d_k, d_v, d_ff, n_heads, dropout, max_len, device)
        self.attention_block3 = cf.ConformerEncoder(self.d_model, d_k, d_v, d_ff, n_heads, dropout, max_len, device)
        self.attention_block4 = cf.ConformerEncoder(self.d_model, d_k, d_v, d_ff, n_heads, dropout, max_len, device)
        self.fc1 = nn.Linear(self.d_model * 2, self.d_model)
        self.fc2 = nn.Linear(self.d_model, self.d_model)
        self.fc3 = nn.Linear(self.d_model, n_lang)

    def forward(self, x, atten_mask):
        batch_size = x.size(0)
        output = self.transform(x)  # x [B, T, input_dim] => [B, T feat_dim]
        output = self.layernorm1(output)
        output = output.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output, _ = self.attention_block1(output, atten_mask)
        output, _ = self.attention_block2(output, atten_mask)
        output, _ = self.attention_block3(output, atten_mask)
        output, _ = self.attention_block4(output, atten_mask)
        stats = torch.cat((output.mean(dim=1), output.std(dim=1)), dim=1)
        # print(stats.size())
        output = F.relu(self.fc1(stats))
        output = F.relu(self.fc2(output))
        output = self.fc3(output)
        return output

