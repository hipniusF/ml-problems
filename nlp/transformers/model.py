import math

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import nn
import numpy as np
from datetime import datetime
from tqdm.auto import tqdm
from matplotlib import pyplot as plt


class Attention(nn.Module):
    def __init__(self, dim_k, dim_v, dropout=0.1):
        super(Attention, self).__init__()
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        computed_queries = q.matmul(k.transpose(-2, -1)) / math.sqrt(self.dim_k)
        if mask is not None:
            computed_queries = computed_queries.masked_fill(mask == 0, value=float('-inf'))
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
        q = self.lv(q).view(batch_size, -1, self.h, self.dim_k).transpose(1, 2)
        k = self.lv(k).view(batch_size, -1, self.h, self.dim_k).transpose(1, 2)
        v = self.lv(v).view(batch_size, -1, self.h, self.dim_k).transpose(1, 2)
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
        pos = torch.arange(0, max_len).unsqueeze(1)
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
        self.dim_model = dim_model
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.encoder = Encoder(N, dim_model, d_ff, dropout)
        self.decoder = Decoder(N, dim_model, d_ff, dropout)
        self.src_embed = nn.Sequential(Embedding(dim_model, src_vocab), PositionalEncoding(dim_model, dropout))
        self.tgt_embed = nn.Sequential(Embedding(dim_model, tgt_vocab), PositionalEncoding(dim_model, dropout))
        self.generator = Generator(dim_model, tgt_vocab)

    def forward(self, src, tgt, src_mask, tgt_mask):
        memory = self.encode(src, src_mask)
        x = self.decode(tgt, tgt_mask, memory, src_mask)
        return self.generator(x)
    
    def encode(self, src, mask):
        src = self.src_embed(src)
        return self.encoder(src, mask)
    
    def decode(self, tgt, tgt_mask, memory, src_mask):
        tgt = self.tgt_embed(tgt)
        return self.decoder(tgt, memory, tgt_mask, src_mask)

    def translate(self, src, src_mask, dataset, dev='cpu'):
        ys = torch.zeros(1, 1).fill_(dataset.start_symbol).type_as(src.data)
        memory = self.encode(src, src_mask)

        i = 0
        while ys[0, -1] != dataset.end_symbol:
            out = self.decode(
                ys, dataset.subsequent_mask(ys.size(1)).type_as(src.data).to(dev), memory, src_mask
                )

            prob = self.generator(out[:, -1])

            _, next_word = torch.max(prob, dim=1)
            ys = torch.cat(
                        [ys, next_word.unsqueeze(0)],
                dim=1
            )
            if i == 50:
                break
            i += 1

        return ys


class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx=1, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())


def scheduler_func(step, dim_model, warmup=4000):
    if step == 0:
        step = 1
    return dim_model ** (-0.5) * min (step**(-.5),step*warmup**(-1.5))


class Trainer:
    def __init__(self, model, dataset, dev='cpu', criterion='cross_entropy'):
        self.dev = dev
        self.model = model
        self.dataset = dataset
        self.optim = torch.optim.Adam(model.parameters(), lr=1, betas=(0.9, 0.98), eps=10e-9)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optim, lr_lambda=lambda step: scheduler_func(step, model.dim_model, warmup=4000))
        self.loss_fn = self._get_loss(criterion)

        self.losses = []
        self.step = 0

        """
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        """

    def _get_loss(self, criterion):
        if criterion == 'cross_entropy':
            return nn.CrossEntropyLoss(ignore_index=self.dataset.pad_symbol, label_smoothing=0.1) 
        elif criterion == 'label_smoothing':
            return LabelSmoothing(
                size=len(self.dataset.trg_vocab),
                padding_idx=self.dataset.pad_symbol,
                smoothing=0.1)
        else:
            raise ValueError(f'{criterion} is not a valid criterion')

    def train_loop(self, steps, batch_size=1, accum_steps=1, save=True, notify=True):
        self.model.train()
        loader = DataLoader(
            self.dataset.train, shuffle=True, batch_size=batch_size, collate_fn=self.dataset.collate_fn)

        with tqdm(initial=self.step, total=steps) as tbar:
            while self.step <= steps:
                for i, batch in enumerate(loader):
                    src = batch['src']
                    tgt = batch['trg']
                    src_mask = batch['src_mask']
                    tgt_mask = batch['trg_mask']
                    trg_y = batch['trg_y']
                    ntokens = batch['ntokens']

                    y_hat = self.model(src, tgt, src_mask, tgt_mask)
                    loss = self.loss_fn(
                        y_hat.view(-1, self.model.tgt_vocab), trg_y.reshape(-1)) #/ ntokens
                    loss.backward()
                    self.losses.append(loss.item())
                    del loss

                    if i % accum_steps == 0:
                        for p in self.model.parameters():
                            p.grad = None

                        self.optim.step()
                        self.scheduler.step()

                    if self.step % 50_000 == 0:
                        if save:
                            self.save(f'./chkpnts/checkpnt_step-{self.step // 1000}k.pt')
                        if notify:
                            self.notify()
                    self.step+=1
                    tbar.update(1)
                    if self.step > steps:
                        break
        
    def save(self, path):
        torch.save({'model': self.model.state_dict(),
                    'optim': self.optim.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'losses': self.losses,
                    'step': self.step,
                    'timestamp': str(datetime.now())
                    },
            path)
    
    def load(self, path):
        chk = torch.load(path)
        self.model.load_state_dict(chk['model'])
        self.optim.load_state_dict(chk['optim'])
        self.scheduler.load_state_dict(chk['scheduler'])
        self.losses = chk['losses']
        self.step = chk['step']

    def notify(self):
        '''
        Send notification through:
        https://github.com/marcoperg/telegram-notifier
        '''

        import os
        import io
        import requests
        import dotenv
        dotenv.load_dotenv()

        test_dl = DataLoader(
            self.dataset.train, shuffle=True, batch_size=1, collate_fn=self.dataset.collate_fn)

        example =  next(iter(test_dl))
        src = example['src']
        src_mask = example['src_mask']

        self.model.eval()
        trg = self.model.translate(
            src, src_mask, self.dataset, dev=self.dev)
        src_str, trg_str = self.dataset.itos(src[0], field='src'), self.dataset.itos(trg[0])

        plt.figure()
        plt.plot(self.losses)
        plt.title('Train losses')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image_data = buf.getvalue()
        url = 'http://localhost:3000'
        files = {'photo': image_data}
        headers = {'token': os.environ['SECRET']}
        data = {'text': f'Step {self.step//1000}k\nEn: {src_str}\nDe: {trg_str}'}
        requests.post(url, files=files, data=data, headers=headers)