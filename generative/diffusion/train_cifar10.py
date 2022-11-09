#!/home/marco/miniconda3/bin/python

from multiprocessing import cpu_count
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset
import torchvision
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Compose, CenterCrop, Normalize, RandomHorizontalFlip
from tqdm.notebook import tqdm
from matplotlib import pyplot as plt

dev = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_loader():
    train_set = CIFAR10('./dataset', train=True, transform=Compose([
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize(.5, .5),
        ]), download=True)

    test_set = CIFAR10('./dataset', train=False, transform=Compose([
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize(.5, .5),
        ]), download=True)

    dataset = ConcatDataset([train_set, test_set])

    BS = 64
    return DataLoader(dataset, batch_size=BS, num_workers=cpu_count(), shuffle=True)

from model import DenoisingDiffusion
import os

def get_model():
    model = DenoisingDiffusion(
        lr=2e-4,
        channels=[64, 128, 128, 128],
        diffusion_steps=1000,
        dropout_rate=0.1,
        dev=dev
    )
    try:
        chks = os.listdir('./chkpnts')
        ns = [int(chk.split('checkpnt_step-')[1].split('k.pt')[0])
             for chk in chks]
        if len(ns) == 0:
            raise Exception('No checkpoints in ./chkpnts')
        n = sorted(ns)[-1]
        print(f'checkpnt_epoch-{n}.pt')
        model.load(f'chkpnts/checkpnt_step-{n}k.pt')

    except Exception as e:
        print(e)
    return model

if __name__ == '__main__':
    loader = get_loader()
    model = get_model()
    model.train_loop(800_000, loader)
