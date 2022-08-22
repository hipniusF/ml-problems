import math

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
        computed_queries = F.softmax(computed_queries, dim=1)
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
        self.l = nn.Linear(h*dim_v, dim_model)

    def forward(self, q, k, v):
        q = self.lv(q).view(-1, self.h, self.dim_k)
        k = self.lv(k).view(-1, self.h, self.dim_k)
        v = self.lv(v).view(-1, self.h, self.dim_v)
        x = self.attention(q, k, v).view(-1, self.h*self.dim_v)
        return self.l(x)


class FeedForward(nn.Module):
    def __init__(self, dim_model=512, dim_inner_layer=2048):
        super(FeedForward, self).__init__()
        self.l1 = nn.Linear(dim_model, dim_inner_layer)
        self.l2 = nn.Linear(dim_inner_layer, dim_model)

    def forward(self, x):
        return F.relu(self.l2(self.l1(x)))


class Embedding(nn.Module):
    def __init__(self, dim_model=512, emb_len=10_000):
        super(Embedding, self).__init__()
        self.emb = nn.Embedding(emb_len, dim_model)

    def forward(self, x):
        return self.emb(x)


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model=512, max_len=10_000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros((max_len, 1, dim_model))
        """
        -- Slow --
        for i in torch.arange(dim_model // 2):
            den = torch.pow(torch.tensor(max_len),
                            torch.tensor(2*i / dim_model))
            for pos in range(max_len):
                pe[pos, i] = torch.sin(torch.tensor(pos)/den)
                pe[pos, i+1] = torch.cos(torch.tensor(pos)/den)
        """
        denominator = torch.exp(torch.arange(0, dim_model, 2)
                                * (-math.log(10000.0) / dim_model))
        pos = torch.arange(max_len).unsqueeze(1)
        pe[:, 0, 0::2] = torch.sin(pos * denominator)
        pe[:, 0, 1::2] = torch.cos(pos * denominator)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]


class ResidualConnectionLayer(nn.Module):
    def __init__(self, model_dim, sublayer):
        super(ResidualConnectionLayer, self).__init__()
        self.norm = nn.LayerNorm(model_dim)
        self.sublayer = sublayer

    def forward(self, x):
        return self.norm(x + self.sublayer(x))


class TokenizerLayer(nn.Module):
    def __init__(self, vocab):
        self.vocab = vocab
        self.word2idx = {x: i for i, x in enumerate(vocab)}
        self.idx2word = [x for x in vocab]

    def tokenize(self, x: str):
        l = [self.word2idx[x_i] for x_i in x.casefold().split(' ')]
        return torch.tensor(l)

    def detokenize(self, x):
        s = [self.idx2word[x_i] for x_i in x]
        return ' '.join(s)


class Transformer(nn.Module):
    def __init__(self, input_vocab, output_vocab, dim_model=512, dim_k=64, dim_v=64, h=8, N=6):
        super(Transformer, self).__init__()
        self.input_tokenizer = TokenizerLayer(input_vocab)
        self.output_tokenizer = TokenizerLayer(output_vocab)
        self.embedding = Embedding(dim_model, 697162)
        self.pos_encoding = PositionalEncoding()
        encoder_layers = []
        for i in range(N):
            encoder_layers.append(ResidualConnectionLayer(
                dim_model, MultiHeadAttention(dim_k, dim_v, dim_model, h)))
            encoder_layers.append(ResidualConnectionLayer(
                dim_model, FeedForward(dim_model)
            ))
        self.encoder = nn.Sequential(*encoder_layers)

    def forward(self, x):
        x = self.input_tokenizer.tokenize(x)
        x = self.embedding(x)
        x = self.pos_encoding(x)
        return x
