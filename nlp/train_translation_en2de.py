import torch
from torch import nn
from torch.utils.data import DataLoader
from torchtext import data, datasets
import spacy
from matplotlib import pyplot as plt
import numpy as np
from tqdm.auto import tqdm

dev = 'cuda' if torch.cuda.is_available() else 'cpu'

from utils import Multi30KEn2DeDatasetTokenizer, get_trainer_model
from transformers.model import EncoderDecoder, Trainer


if __name__ == '__main__':
    dataset = Multi30KEn2DeDatasetTokenizer(dev=dev)
    trainer, model = get_trainer_model(dataset, EncoderDecoder, Trainer, dev=dev)
    try:
        trainer.load('./chkpnts/current.pt')
        print(f'Current checkpnt at step {trainer.step:,}')
    except FileNotFoundError:
        print('current.pt not found')
    try:
        trainer.train_loop(1_000_000, batch_size=400, save=True, notify=True)
    except KeyboardInterrupt:
        print('saving current.pt...')
        trainer.save('./chkpnts/current.pt')