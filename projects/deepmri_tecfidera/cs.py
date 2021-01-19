import os
import sys

import numpy as np
import torch

from projects.deepmri_tecfidera.utils import mse

os.environ['TOOLBOX_PATH'] = "/home/dkarkalousos/bart-0.6.00/"
sys.path.append('/home/dkarkalousos/bart-0.6.00/python/')

import bart


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


def writecfl(name, array):
    h = open(name + ".hdr", "w")
    h.write('# Dimensions\n')
    for i in (array.shape):
        h.write("%d " % i)
    h.write('\n')
    h.close()
    d = open(name + ".cfl", "w")
    array.T.astype(np.complex64).tofile(d)  # tranpose for column-major order
    d.close()


class CompressedSensing(object):
    def __init__(self, l='1', r='0.01', i='20', d='5'):
        self.l = l
        self.r = r
        self.i = i
        self.d = d
        self.n_feature = np.nan
        self.t_max = 1

    def get_num_params(self):
        return np.nan

    def load_state_dict(self, *args):
        pass

    def inference(self, batch, t_max=1, get_loss=True, get_estimate=True, final=True):
        target = batch['target'].cpu().numpy()
        target = target[..., 0] + 1j * target[..., 1]
        if batch['sense'].dim() == 1:
            y = batch['y'].cpu().numpy()
            kspace = np.fft.fftshift(y[..., 0] + 1j * y[..., 1], axes=(-2, -1))
            sense = np.ones(kspace.shape)
        else:
            kspace = batch['y'].permute(0, 2, 3, 1, 4).unsqueeze(1).cpu().numpy()
            kspace = np.fft.fftshift(kspace[..., 0] + 1j * kspace[..., 1], axes=(-3, -2))
            sense = batch['sense'].permute(0, 2, 3, 1, 4).unsqueeze(1).cpu().numpy()
            sense = np.fft.fftshift(sense[..., 0] + 1j * sense[..., 1], axes=(-3, -2))
        recon = [bart.bart(1, 'pics -d0 -S -R W:7:0:0.005 -i 60', k, s) for k, s in zip(kspace, sense)]
        recon = np.concatenate(
            [rec * np.mean(np.absolute(target).real) / np.mean(np.absolute(rec).real) for rec in recon], 0)
        recon = np.fft.ifftshift(recon, axes=(-2, -1))
        recon = torch.stack((torch.tensor(recon.real), torch.tensor(recon.imag)), 3)

        if get_loss:
            loss = [mse(recon, batch['target'])]
            if get_estimate:
                yield loss, recon
            else:
                yield loss
        else:
            if final:
                recon = recon[0]
            yield recon


def _CompressedSensing(batch):
    target = batch['target'].cpu().numpy()
    target = target[..., 0] + 1j * target[..., 1]
    target = np.clip(np.absolute(target).real, 0, 1)

    kspace = batch['y'].permute(0, 2, 3, 1, 4).unsqueeze(1).cpu().numpy()
    kspace = np.fft.fftshift(kspace[..., 0] + 1j * kspace[..., 1], axes=(-3, -2))
    sense = batch['sense'].permute(0, 2, 3, 1, 4).unsqueeze(1).cpu().numpy()
    sense = np.fft.fftshift(sense[..., 0] + 1j * sense[..., 1], axes=(-3, -2))

    recon = [bart.bart(1, 'pics -d0 -S -R W:7:0:0.005 -i 60', k, s) for k, s in zip(kspace, sense)]
    recon = np.concatenate([rec * np.mean(np.absolute(target).real) / np.mean(np.absolute(rec).real) for rec in recon],
                           0)
    recon = np.fft.ifftshift(recon, axes=(-2, -1))
    recon = torch.stack((torch.tensor(recon.real), torch.tensor(recon.imag)), 3)

    yield recon
