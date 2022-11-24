from pathlib import Path
from urllib.request import urlopen
import linecache
from itertools import count
import pickle

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
