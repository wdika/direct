# encoding: utf-8
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.fft import ifftn, fftn

from direct.data import transforms as T


def readcfl(cfl):
    h = open(cfl + ".hdr", "r")
    h.readline()  # skip
    line = h.readline()
    h.close()
    dims = [int(i) for i in line.split()]

    # remove singleton dimensions from the end
    n = int(np.prod(dims))
    dims_prod = np.cumprod(dims)
    dims = dims[:np.searchsorted(dims_prod, n) + 1]

    # load data and reshape into dims
    d = open(cfl + ".cfl", "r")
    a = np.fromfile(d, dtype=np.complex64, count=n)
    d.close()
    a = a.reshape(dims, order='F')  # column-major

    return a


def create_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def complex_tensor_to_real_np(x):
    return torch.abs(x).detach().cpu().numpy()


def complex_tensor_to_complex_np(x):
    return x.detach().cpu().numpy().astype(np.complex64)


def save_png_outputs(data, output_dir):
    create_dir(output_dir)

    for i in range(data.shape[0]):
        plt.imshow(data[i], cmap='gray')
        plt.savefig(output_dir + str(i) + '.png')
        plt.close()


def save_h5_outputs(data, key, filename):
    with h5py.File(filename + ".h5", "w") as f:
        f[key] = data


def slice_selection(data, start, end):
    return data[start:end]


def preprocessing_ifft(kspace):
    """

    Parameters
    ----------
    kspace : torch.Tensor

    Returns
    -------
    kspace tensor of the axial plane transformed with the correct/fixed preprocessing steps to estimate sense maps
    """
    return T.fftshift(ifftn(kspace, dim=(0, 1, 2), norm="ortho"), dim=0)


def preprocessing_csm_fft(kspace):
    """

    Parameters
    ----------
    kspace : torch.Tensor

    Returns
    -------
    image space tensor of the axial plane transformed with the correct/fixed preprocessing steps
    """
    return T.fftshift(fftn(preprocessing_ifft(kspace), dim=(1, 2), norm="ortho"), dim=(1, 2))


def extract_mask(kspace):
    """

    Parameters
    ----------
    kspace : torch.Tensor

    Returns
    -------
    extracts the mask from the subsampled kspace, after summing the slice and the coil dimensions
    """
    return torch.where(torch.sum(torch.sum(torch.abs(kspace), 0), -1) > 0, 1, 0)


def sense_reconstruction(imspace, csm, dim=-1):
    """

    Parameters
    ----------
    imspace : torch.Tensor
    csm : torch.Tensor
    dim : coil dimension

    Returns
    -------
    reconstructed complex-valued image using SENSE
    """
    return torch.sum(imspace * torch.conj(csm), dim=dim)


def csm_sense_coil_combination(csm, dim=-1):
    """

    Parameters
    ----------
    csm : torch.Tensor
    dim : coil dimension

    Returns
    -------
    coil combined image
    """
    return torch.sum(torch.conj(csm), dim=dim)
