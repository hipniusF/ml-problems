from pathlib import Path
from urllib.request import urlopen
import linecache
from itertools import count
import pickle

import torch
from torchtext import data, datasets

class En2DeWMT14Dataset(torch.utils.data.Dataset):
    def __init__(self, folder_path, transform=None, download=False, train=True):
        self.path = Path(folder_path)
        self.train = train
        self.transform = transform
        self.train_en_url = 'https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.en'
        self.train_de_url = 'https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.de'

        self.test_en_url = 'https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2015.en'
        self.test_de_url = 'https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2015.de'

        self.length = 4_468_841 if train else 2_170
        if train:
            try:
                with (self.path / 'vocabs').open('rb') as f:
                    self.vocabs = pickle.load(f)  
            except FileNotFoundError:
                if download:
                    self._download()
                else:
                    raise FileNotFoundError('Set download to True to download the dataset')
        files = ('train.en', 'train.de') if self.train else ('test.en', 'test.de')
        line_path = self.path / files[0]
        label_path = self.path / files[1]
        if not line_path.exists() or not label_path.exists():
            if download:
                self._download()
            else:
                raise FileNotFoundError('Set download to True to download the dataset')     
    
    def _download(self):
        self.path.mkdir(parents=True, exist_ok=True)
        if self.train:
            files = (('en', 'train.en', self.train_en_url),
                     ('de', 'train.de', self.train_de_url))
        else:
            files = (('en', 'test.en', self.test_en_url),
                     ('de', 'test.de', self.test_de_url))

        self.vocabs = {'en': set(), 'de': set()}
        for lang, file, url in files:
            with urlopen(url) as webfile:
                localpath = self.path / file
                if localpath.exists():
                    localpath.unlink()
                with localpath.open("wb+") as localfile:
                    for i in tqdm(range(self.length)):
                        line = webfile.readline()
                        if self.train:
                            self.vocabs[lang].update(line.decode("utf-8").casefold().split(' '))
                        localfile.write(line)
                    assert(not line)
        vocab_path = self.path / 'vocabs'
        with vocab_path.open('wb') as f:
            pickle.dump(self.vocabs, f)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        files = ('train.en', 'train.de') if self.train else ('test.en', 'test.de')
        line_path = self.path / files[0]
        label_path = self.path / files[1]
        if not line_path.exists() or not label_path.exists():
            raise FileNotFoundError('Set download to True to download the dataset')
        
        line = linecache.getline(str(line_path.absolute()), idx)
        label = linecache.getline(str(label_path.absolute()), idx)

        if self.transform:
            line = self.transform(line)
            label = self.transform(label)
        return line, label


class Multi30KEn2DeDatasetTokenizer:
    def __init__(self, dev):
        self.dev = dev
        tokenize_en = data.get_tokenizer("spacy", language='en_core_web_sm')
        tokenize_de = data.get_tokenizer("spacy", language='de_core_news_sm')

        src = data.Field(tokenize_en)
        tgt = data.Field(tokenize_de)

        self.train, self.val, self.test = datasets.Multi30k.splits(
            ('.en', '.de'), fields=(src, tgt) , root='./downloads')

        self.src_field = self.train.fields['src'] 
        self.trg_field = self.train.fields['trg'] 

        src_list, trg_list = [], []
        for dt_pnt in self.train:
            src_list.append(dt_pnt.src)
            trg_list.append(dt_pnt.trg)

        specials = ['<pad>', '<s>', '</s>', "<blank>", "<unk>"]
        self.src_field.build_vocab(src_list, specials=specials)
        self.trg_field.build_vocab(trg_list, specials=specials)

        self.src_vocab = self.src_field.vocab
        self.trg_vocab = self.trg_field.vocab

        self.start_symbol = int(self.trg_field.numericalize([['<s>']]))
        self.pad_symbol = int(self.trg_field.numericalize([['<pad>']]))

    def itos(self, t, field='trg'):
        s = []
        for c in t:
            s.append(self.train.fields[field].vocab.itos[c])
        return ' '.join(s)

    def collate_fn(self, batch):
        src_list, trg_list = [], []
        for dt_pnt in batch:
            src_list.append(['<s>'] + dt_pnt.src + ['</s>'])
            trg_list.append(['<s>'] + dt_pnt.trg + ['</s>'])

        src_list = self.src_field.pad(src_list)
        trg_list = self.trg_field.pad(trg_list)
        
        src_list = self.src_field.numericalize(src_list).T.to(self.dev)
        trg_list = self.trg_field.numericalize(trg_list).T.to(self.dev)
        trg = trg_list[:, :-1]
        trg_y = trg_list[:,1:]

        pad = self.pad_symbol
        src_mask = (src_list != pad).unsqueeze(-2).unsqueeze(-3)
        trg_mask = (trg != pad).unsqueeze(-2).unsqueeze(-2)

        trg_mask = trg_mask & self.subsequent_mask(
            trg.size(-1)).type_as(trg_mask.data)
            
        return {'src': src_list,
                'trg': trg,
                'src_mask': src_mask.to(self.dev),
                'trg_mask': trg_mask.to(self.dev),
                'trg_y': trg_y,
                'ntokens': (trg_y != pad).data.sum()}

    def subsequent_mask(self, size):
        "Mask out subsequent positions."
        attn_shape = (1, size, size)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
        return subsequent_mask == 0