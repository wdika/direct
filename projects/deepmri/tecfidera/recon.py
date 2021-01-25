#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
from os.path import join
from pathlib import Path

import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
import pickle5 as pickle


def readcfl(name):
    # get dims from .hdr
    h = open(name + ".hdr", "r")
    h.readline()  # skip
    l = h.readline()
    h.close()
    dims = [int(i) for i in l.split()]
    # remove singleton dimensions from the end
    n = np.prod(dims)
    dims_prod = np.cumprod(dims)
    dims = dims[:np.searchsorted(dims_prod, n) + 1]
    # load data and reshape into dims
    d = open(name + ".cfl", "r")
    a = np.fromfile(d, dtype=np.complex64, count=n);
    d.close()
    return a.reshape(dims, order='F')  # column-major


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='Name of the model to be used')
args = parser.parse_args()

dir = '/data1/projects/compressed_sensing/dimitris/plots/lesions/'
model = args.model + '/'
modeldir = dir + model
cfl_recon_dir = modeldir + 'cfl_recon/'
pickle_recon_dir = modeldir + 'pickle_recon/'
png_recon_dir = modeldir + 'png_recon/'

Path(modeldir).mkdir(parents=True, exist_ok=True)
Path(cfl_recon_dir).mkdir(parents=True, exist_ok=True)
Path(pickle_recon_dir).mkdir(parents=True, exist_ok=True)
Path(png_recon_dir).mkdir(parents=True, exist_ok=True)

os.system('./recon_' + args.model + '.sh')

files = list(Path(cfl_recon_dir).iterdir())
for fname in files:
    _fname = str(fname).split('/')[-1]
    if _fname.split('.')[-1] == 'cfl':
        data = readcfl('/'.join(str(fname).split('/')[:-1]) + '/' + '.'.join(_fname.split('.')[0:-1]))

        with open(join(pickle_recon_dir, '.'.join(_fname.split('.')[0:-1])), 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

files = list(Path(pickle_recon_dir).iterdir())
for fname in files:
    with open(fname, 'rb') as f:
        img = pickle.load(f)

    save_name = png_recon_dir + str(fname).split('/')[-1] + '.png'

    plt.imshow(np.rot90(np.abs(img)), cmap=cc.m_gray)
    plt.axis('off')
    plt.savefig(save_name)
