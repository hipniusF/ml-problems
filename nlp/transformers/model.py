import math

import torch
import torch.nn.functional as F
from torch.multiprocessing import Pool
from torch import nn


class Attention(nn.Module):
    def __init__(self, dim_k, dim_v, dropout=0.1):
        super(Attention, self).__init__()
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        computed_queries = q.matmul(k.transpose(-2, -1)) / math.sqrt(self.dim_k)
        if mask is not None:
            computed_queries = computed_queries.masked_fill(mask == 0, value=-1e9)
        computed_queries = F.softmax(computed_queries, dim=-1)
        computed_queries = self.dropout(computed_queries)
        return computed_queries.matmul(v)


class MultiHeadAttention(nn.Module):
    def __init__(self, dim_model=512, h=8, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert dim_model % h == 0
        self.dim_k = dim_model // h
        self.dim_model = dim_model
        self.h = h
        self.attention = Attention(self.dim_k, self.dim_k)
        self.lq = nn.Linear(dim_model, dim_model)
        self.lk = nn.Linear(dim_model, dim_model)
        self.lv = nn.Linear(dim_model, dim_model)
        self.l = nn.Linear(dim_model, dim_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        if isinstance(x, tuple):
            q, k, v = x
        elif isinstance(x, torch.Tensor):
            q, k, v = x, x.clone(), x.clone()
        else:
            raise TypeError(
                'Input to MultiHeadAttention excepted to be either Tensor or 3-tuple of Tensors')
        batch_size = q.size(0)
        q = self.lv(q).view(batch_size, -1, self.h, self.dim_k)
        k = self.lv(k).view(batch_size, -1, self.h, self.dim_k)
        v = self.lv(v).view(batch_size, -1, self.h, self.dim_k)
        x = self.attention(q, k, v, mask).contiguous().view(batch_size, -1, self.h*self.dim_k)
        del q
        del k
        del v
        x = self.l(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, dim_model=512, dim_inner_layer=2048, dropout=0.1):
        super(FeedForward, self).__init__()
        self.l1 = nn.Linear(dim_model, dim_inner_layer)
        self.l2 = nn.Linear(dim_inner_layer, dim_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.l2(self.dropout(F.relu(self.l1(x))))


class Embedding(nn.Module):
    def __init__(self, dim_model=512, emb_len=10_000):
        super(Embedding, self).__init__()
        self.emb = nn.Embedding(emb_len, dim_model)
        self.dim_model = dim_model

    def forward(self, x):
        return self.emb(x) + math.sqrt(self.dim_model)


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model=512, dropout=0, max_len=10_000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros((max_len, dim_model))
        denominator = torch.exp(torch.arange(0, dim_model, 2)
                                * (-math.log(10000.0) / dim_model))
        pos = torch.arange(max_len).unsqueeze(1)
        pe[:, 0::2] = torch.sin(pos * denominator)
        pe[:, 1::2] = torch.cos(pos * denominator)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class EncoderResidualLayer(nn.Module):
    def __init__(self, model_dim, sublayer, dropout=0.1):
        super(EncoderResidualLayer, self).__init__()
        self.norm = nn.LayerNorm(model_dim)
        self.sublayer = sublayer
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask=None):
        if mask is not None:
            return self.norm(x + self.dropout(self.sublayer(x, mask=mask)))
        else:
            return self.norm(x + self.dropout(self.sublayer(x)))


class DecoderResidualLayer(nn.Module):
    def __init__(self, model_dim, sublayer, dropout=0.1):
        super(DecoderResidualLayer, self).__init__()
        self.norm = nn.LayerNorm(model_dim)
        self.sublayer = sublayer
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, q, k, v, mask=None):
        if mask is not None:
            return self.norm(q + self.dropout(self.sublayer((q, k, v), mask=mask)))
        else:
            return self.norm(q + self.dropout(self.sublayer((q, k, v))))

class Encoder(nn.Module):
    def __init__(self, N, dim_model=512, d_ff=2048, dropout=0.1):
        super(Encoder, self).__init__()
        self.atts = nn.ModuleList([EncoderResidualLayer(
            dim_model,
            MultiHeadAttention(dim_model, dropout=dropout), dropout=dropout) for _ in range(N)])
        self.ffs = nn.ModuleList([EncoderResidualLayer(
            dim_model,
            FeedForward(dim_model, d_ff, dropout=dropout), dropout=dropout) for _ in range(N)])
    
    def forward(self, x, mask=None):
        for att, ff in zip(self.atts, self.ffs):
            x = att(x, mask=mask)
            x = ff(x)
        return x


class Decoder(nn.Module):
    def __init__(self, N, dim_model=512, d_ff=2048, dropout=0.1):
        super(Decoder, self).__init__()
        self.src_atts = nn.ModuleList([EncoderResidualLayer(
            dim_model,
            MultiHeadAttention(dim_model, dropout=dropout), dropout=dropout) for _ in range(N)])
        self.tgt_atts = nn.ModuleList([DecoderResidualLayer(
            dim_model,
            MultiHeadAttention(dim_model, dropout=dropout), dropout=dropout) for _ in range(N)])
        self.ffs = nn.ModuleList([EncoderResidualLayer(
            dim_model,
            FeedForward(dim_model, d_ff, dropout=dropout), dropout=dropout) for _ in range(N)])
    
    def forward(self, x, memory, tgt_mask=None, src_mask=None):
        for src_att, tgt_att, ff in zip(self.src_atts, self.tgt_atts, self.ffs):
            x = src_att(x, tgt_mask)
            x = tgt_att(x, memory, memory.clone(), src_mask)
            x = ff(x)
        return x


class Generator(nn.Module):
    def __init__(self, dim_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(dim_model, vocab)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        return self.softmax(self.proj(x))


class EncoderDecoder(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, N=6, dim_model=512, d_ff=2048, dropout=0.1):
        super(EncoderDecoder, self).__init__()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.encoder = Encoder(N, dim_model, d_ff, dropout)
        self.decoder = Decoder(N, dim_model, d_ff, dropout)
        self.src_embed = nn.Sequential(Embedding(dim_model, src_vocab), PositionalEncoding(dim_model, dropout))
        self.tgt_embed = nn.Sequential(Embedding(dim_model, tgt_vocab), PositionalEncoding(dim_model, dropout))
        self.generator = Generator(dim_model, tgt_vocab)

        # Initialize parameters with Glorot / fan_avg.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_mask, tgt_mask):
        src = self.src_embed(src)
        tgt = self.tgt_embed(tgt)
        memory = self.encode(src, src_mask)
        x = self.decode(tgt, tgt_mask, memory, src_mask)
        return self.generator(x)
    
    def encode(self, src, mask):
        return self.encoder(src, mask)
    
    def decode(self, tgt, tgt_mask, memory, src_mask):
        return self.decoder(tgt, memory, tgt_mask, src_mask)