from calendar import c
import torch
import torch.nn.functional as F
from torch.multiprocessing import Pool
from torch import nn


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dim_k, dim_v):
        super(ScaledDotProductAttention, self).__init__()
        self.dim_k = dim_k
        self.dim_v = dim_v

    def forward(self, q, k, v):
        print(q.shape, k.transpose(1, 2).shape)

        computed_queries = q.matmul(k.transpose(1, 2))
        computed_queries = F.softmax(computed_queries, dim=0)
        return computed_queries.matmul(v)


class MultiHeadAttention(nn.Module):
    def __init__(self, dim_k=64, dim_v=64, dim_model=512, h=8):
        super(MultiHeadAttention, self).__init__()
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dim_model = dim_model
        self.h = h
        self.attention = ScaledDotProductAttention(dim_k, dim_v)
        self.lq = nn.Linear(dim_model, h*dim_k)
        self.lk = nn.Linear(dim_model, h*dim_k)
        self.lv = nn.Linear(dim_model, h*dim_v)

    def forward(self, q, k, v):
        q = self.lv(q).view(-1, self.h, self.dim_k)
        k = self.lv(k).view(-1, self.h, self.dim_k)
        v = self.lv(v).view(-1, self.h, self.dim_v)
        return self.attention(q, k, v)


class FeedForward(nn.Module):
    def __init__(self, dim_model=512, dim_inner_layer=2048):
        super(FeedForward, self).__init__()
        self.l1 = nn.Linear(dim_model, dim_inner_layer)
        self.l2 = nn.Linear(dim_inner_layer, dim_model)

    def forward(self, x):
        return F.relu(self.l2(self.l1(x)))
