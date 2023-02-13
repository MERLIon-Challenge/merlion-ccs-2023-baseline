import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from conformer_conv import ConformerConvModule

class PositionalEncoding(nn.Module):
    """
    PE_(pos, 2i)    =  sin(pos / power(10000, 2i / d_model))
    PE_(pos, 2i+1)  =  cos(pos / power(10000, 2i / d_model))
    """
    def __init__(self, d_model: int = 512, max_len: int = 10000) -> None:
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, length: int):
        return self.pe[:, :length]


class LayerNorm(nn.Module):
    def __init__(self, d_hid, eps=1e-6):
        super(LayerNorm, self).__init__()
        # d_hid = feat_dim
        self.gamma = nn.Parameter(torch.ones(d_hid))
        self.beta = nn.Parameter(torch.zeros(d_hid))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True, )
        std = x.std(dim=-1, keepdim=True, )
        ln_out = (x - mean) / (std + self.eps)
        ln_out = self.gamma * ln_out + self.beta
        return ln_out


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.scale_factor = np.sqrt(d_k)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, atten_mask=None):
        # queries: [B, n_head, len_queries, d_k]
        # keys: [B, n_head, len_keys, d_k]
        # values: [B, n_head, len_values, d_v] note: len_keys = len_values
        scores = torch.matmul(q, k.transpose(-1, -2)) / self.scale_factor
        if atten_mask is not None:
            # print(atten_mask.size(),scores.size())
            assert atten_mask.size() == scores.size()
            scores.masked_fill_(atten_mask, -1e9)
        atten = self.dropout(self.softmax(scores))
        context = torch.matmul(atten, v)
        return context, atten


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.xavier_normal_(self.linear.weight)
        if bias:
            init.zeros_(self.linear.bias)

    def forward(self, inputs):
        return self.linear(inputs)


class RelMultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads, dropout):
        super(RelMultiHeadAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.n_heads = n_heads
        self.scale_factor = np.sqrt(d_model)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

        self.w_q = Linear(self.d_model, d_k * n_heads)
        self.w_k = Linear(self.d_model, d_k * n_heads)
        self.w_v = Linear(self.d_model, d_v * n_heads)
        self.pos_proj = Linear(self.d_model, self.d_model, bias=False)

        self.u_bias = nn.Parameter(torch.Tensor(n_heads, d_k))
        self.v_bias = nn.Parameter(torch.Tensor(n_heads, d_k))
        torch.nn.init.xavier_uniform_(self.u_bias)
        torch.nn.init.xavier_uniform_(self.v_bias)

        self.out_proj = Linear(d_model, d_model)

    def _relative_shift(self, pos_score):
        batch_size, num_heads, seq_length1, seq_length2 = pos_score.size()
        zeros = pos_score.new_zeros(batch_size, num_heads, seq_length1, 1)
        padded_pos_score = torch.cat([zeros, pos_score], dim=-1)

        padded_pos_score = padded_pos_score.view(batch_size, num_heads, seq_length2 + 1, seq_length1)
        pos_score = padded_pos_score[:, :, 1:].view_as(pos_score)

        return pos_score

    def forward(self, x, pos_emb, atten_mask):
        batch_size = x.size(0)

        q_ = self.w_q(x).view(batch_size, -1, self.n_heads, self.d_k)
        k_ = self.w_k(x).view(batch_size, -1, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        v_ = self.w_v(x).view(batch_size, -1, self.n_heads, self.d_v).permute(0, 2, 1, 3)
        # q_: [Batch, n_heads, len, d_k]
        # k_: [Batch, n_heads, len, d_k]
        # v_: [Batch, n_heads, len, d_v]
        pos_emb = self.pos_proj(pos_emb).view(batch_size, -1, self.n_heads, self.d_k)

        q_ = q_ + self.u_bias
        content_score = torch.matmul((q_ + self.u_bias).transpose(1,2), k_.transpose(2, 3))
        pos_score = torch.matmul((q_ + self.v_bias).transpose(1,2),pos_emb.permute(0, 2, 3, 1))
        pos_score = self._relative_shift(pos_score)
        score = (content_score+pos_score)/ self.scale_factor
        if atten_mask is not None:
            # [Batch, len, len] -> [Batch, n_heads, len, len]
            atten_mask = atten_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
            assert atten_mask.size() == score.size()
            score.masked_fill_(atten_mask, -1e9)
        atten = self.dropout(self.softmax(score))
        context = torch.matmul(atten, v_)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        context = self.out_proj(context)
        return context, atten


class RelMultiHeadAttentionLayer(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads, dropout, max_len, device):
        '''

        :param d_model:
        :param d_k:
        :param d_v:
        :param n_heads:
        :param dropout:
        :param max_len:
        :param device:
        '''
        super(RelMultiHeadAttentionLayer, self).__init__()
        self.device = device
        self.n_heads = n_heads
        self.multihead_attention = RelMultiHeadAttention(d_model, d_k, d_v, n_heads, dropout)
        self.linear = Linear(n_heads * d_v, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = LayerNorm(d_model)

        self.positional_encoding = PositionalEncoding(d_model, max_len)

    def forward(self, x, atten_mask):
        # x_: [Batch, seq_len, d_model], where d_model = feat_dim * n_heads
        batch_size, seq_len, d_model = x.size()
        pos_emb = self.positional_encoding(seq_len).to(self.device)
        pos_emb = pos_emb.repeat(batch_size, 1, 1)
        residual = x
        x = self.layernorm(x)
        context, atten = self.multihead_attention(x, pos_emb, atten_mask)
        output = self.dropout(self.linear(context))
        output = output + residual
        # output: [Batch, len, feat_dim]
        return output, atten


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = Linear(d_model, d_ff)
        self.fc2 = Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = self.layernorm(x)
        output = self.relu(self.fc1(x))
        output = self.dropout(self.fc2(output))
        output = 0.5*output + residual
        return output


class ConformerEncoder(nn.Module):
    def __init__(self, d_model, d_k, d_v, d_ff, n_heads, dropout, max_len, device):
        super(ConformerEncoder, self).__init__()
        self.self_attention = RelMultiHeadAttentionLayer(d_model, d_k, d_v, n_heads, dropout, max_len, device)
        self.position_wise_ff_in = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.position_wise_ff_out = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.layernorm = LayerNorm(d_model)
        self.conv_module = ConformerConvModule(in_channels=d_model)

    def forward(self, x, atten_mask):
        output = self.position_wise_ff_in(x)
        output, atten = self.self_attention(output, atten_mask)
        output = self.conv_module(output)
        output = self.position_wise_ff_out(output)
        output = self.layernorm(output)
        return output, atten


