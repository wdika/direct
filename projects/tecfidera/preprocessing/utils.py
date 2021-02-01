# coding=utf-8
__author__ = 'Dimitrios Karkalousos'

import os
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use('Agg')

from matplotlib import pyplot as plt

import numpy as np
import pickle5 as pickle
import torch
import torch.fft as fft
from torch.nn.functional import pad

os.environ['TOOLBOX_PATH'] = "/home/dkarkalousos/bart-0.6.00/"
sys.path.append('/home/dkarkalousos/bart-0.6.00/python/')
from bart import bart


def create_dir(path):
    """

    Parameters
    ----------
    path :

    Returns
    -------

    """
    Path(path).mkdir(parents=True, exist_ok=True)


def complex_tensor_to_real_np(x):
    """

    Parameters
    ----------
    x :

    Returns
    -------

    """
    return torch.abs(x).detach().cpu().numpy()


def complex_tensor_to_complex_np(x):
    """

    Parameters
    ----------
    x :

    Returns
    -------

    """
    return x.detach().cpu().numpy().astype(np.complex64)


def save_png_outputs(data, output_dir):
    """

    Parameters
    ----------
    data :
    output_dir :

    Returns
    -------

    """
    create_dir(output_dir)
    for i in range(data.shape[0]):
        plt.imshow(data[i], cmap='gray')
        plt.savefig(output_dir + str(i) + '.png')
        plt.close()


def save_h5_outputs(data, key, filename):
    """

    Parameters
    ----------
    data :
    key :
    filename :

    Returns
    -------

    """
    with h5py.File(filename + ".h5", "w") as f:
        f[key] = data


def save_pickle_outputs(data, filename):
    """

    Parameters
    ----------
    data :
    filename :

    Returns
    -------

    """
    for slice in range(data.shape[0]):
        with open(filename + '_' + str(slice), 'wb') as f:
            pickle.dump(data[slice], f, protocol=pickle.HIGHEST_PROTOCOL)


def save_pickle_mask(mask, filename):
    """

    Parameters
    ----------
    mask :
    filename :

    Returns
    -------

    """
    with open(filename, 'wb') as f:
        pickle.dump(mask, f, protocol=pickle.HIGHEST_PROTOCOL)


def slice_selection(data, start, end):
    """

    Parameters
    ----------
    data :
    start :
    end :

    Returns
    -------

    """
    return data[start:end]


def normalize(data):
    """

    Parameters
    ----------
    data :

    Returns
    -------

    """
    if data.shape[-1] == 2:
        data = data[..., 0] + 1j * data[..., 1]

    maximum = torch.max(torch.max(data.real), torch.max(
        data.imag)) if data.dtype == torch.complex64 or data.dtype == torch.complex128 else torch.max(data)
    return torch.where(data == 0, torch.tensor([0.0], dtype=data.dtype), (data / maximum))


def rss_reconstruction(imspace, dim=-1):
    """

    Parameters
    ----------
    imspace : torch.Tensor
    dim : coil dimension

    Returns
    -------
    reconstructed RSS image
    """
    return torch.sqrt(torch.sum(imspace ** 2, dim=dim))


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


def resize_sensitivity_map(sensitivity_map, shape):
    """

    Parameters
    ----------
    sensitivity_map :
    shape :

    Returns
    -------

    """
    [slices, readout, phase_enc, coils] = shape
    return fft.ifftn(pad(fft.fftshift(fft.fftn(sensitivity_map, dim=(0, 1, 2)), dim=(0, 1, 2)), (
        (coils - sensitivity_map.shape[3]) // 2, (coils - sensitivity_map.shape[3]) // 2,
        (phase_enc - sensitivity_map.shape[2]) // 2, (phase_enc - sensitivity_map.shape[2]) // 2,
        (readout - sensitivity_map.shape[1]) // 2, (readout - sensitivity_map.shape[1]) // 2,
        (slices - sensitivity_map.shape[0]) // 2, (slices - sensitivity_map.shape[0]) // 2)), dim=(0, 1, 2))


def estimate_csm(kspace, calibration_region_size):
    """

    Parameters
    ----------
    kspace :
    calibration_region_size :

    Returns
    -------

    """
    sensitivity_map = fft.ifftshift(torch.from_numpy(bart(1, f"caldir {calibration_region_size}", kspace)),
                                    dim=(0, 1, 2))
    if sensitivity_map.shape[-1] == 2:
        sensitivity_map = sensitivity_map[..., 0] + 1j * sensitivity_map[..., 1]
    return sensitivity_map


def load_kprops(list_file):
    """

    Parameters
    ----------
    list_file :

    Returns
    -------

    """
    kprops = {}
    with open(list_file, 'r') as fid:
        for prop in (r.split()[1:] for r in fid if r[0] == '.'):
            if len(prop) in [8, 9]:
                fieldname = '_'.join(prop[3:-2])
            else:
                fieldname = '_'.join(prop[3].split('-'))

            if prop[-2] == ':':
                try:
                    val = [int(prop[-1])]
                except ValueError:
                    val = [float(prop[-1])]
            else:
                val = list(map(int, prop[-2:]))
            kprops[fieldname] = val

        kprops['coil_channel_combination'] = list(map(int, prop[-1]))

    return kprops


def get_acquisition_meta_data(list_file):
    """

    Parameters
    ----------
    list_file :

    Returns
    -------

    """
    converters = {b'STD': 0, b'REJ': 1, b'PHX': 2, b'FRX': 3, b'NOI': 4, b'NAV': 5, b'DNA': 6}

    with open(list_file, 'r') as fid:
        llist = zip(*np.genfromtxt((r for r in fid if r[0] not in ('#', '.')), dtype=int,
                                   converters={0: lambda x: converters[x]}))

    converters = {v: k for k, v in converters.items()}
    kspace = {'typ': tuple(map(lambda x: converters[x], next(llist)))}

    headers = ['mix', 'dyn', 'card', 'echo', 'loca', 'chan', 'extr1', 'extr2', 'ky', 'kz', 'n.a.', 'aver', 'sign', 'rf',
               'grad', 'enc', 'rtop', 'rr', 'size', 'offset']

    for header, vals in zip(headers, llist):
        kspace[header] = vals

    kspace['kspace_properties'] = load_kprops(list_file)

    return kspace


def load_kspace(path):
    """
    Parameters
    ----------
    path : path to the list data

    Returns
    -------
    numpy array
    """
    with open(path + '.data', 'rb') as f:
        complexdata = np.fromfile(f, dtype='<f4')

    complexdata = complexdata[::2] + 1.0j * complexdata[1::2]

    kspace = get_acquisition_meta_data(path + '.list')
    kspace['complexdata'] = np.split(complexdata, np.array(kspace['offset'][1:]) // 8)

    if len(kspace['complexdata'][-1]) != len(kspace['complexdata'][-2]):
        kspace['complexdata'][-1] = kspace['complexdata'][-1][:len(kspace['complexdata'][-2])]

    return kspace


def get_kspace_from_listdata(fdir):
    """

    Parameters
    ----------
    fdir :
    noise :
    remove_oversampling :
    remove_freq_oversampling :
    take_signal_avg :

    Returns
    -------

    """
    kspace = load_kspace(fdir)

    channels = kspace['kspace_properties']['number_of_coil_channels'][0]
    mixes = kspace['kspace_properties']['number_of_mixes'][0]
    dynamic = kspace['kspace_properties']['number_of_dynamic_scans'][0]
    kxmin, kxmax = kspace['kspace_properties']['kx_range']
    kymin, kymax = np.amin(kspace['ky']), np.amax(kspace['ky'])
    signalavg = kspace['kspace_properties']['number_of_signal_averages'][0]

    kzflag = 'kz' if kspace['kspace_properties']['number_of_encoding_dimensions'][0] == 3 else 'loca'
    kzmin, kzmax = np.amin(kspace[kzflag]), np.amax(kspace[kzflag])
    sx = len(kspace['complexdata'][-1])
    tmp = np.array([elem for elem, ktyp in zip(kspace['complexdata'], kspace['typ']) if ktyp == b'STD'], dtype=object).T

    try:
        tmp = tmp.reshape((sx, channels, -1), order='A')
    except ValueError:
        print(f'ValueError: cannot reshape array of size {tmp.shape} into shape ({sx},{channels},newaxis). Skipping...')
        return None

    startidx = kspace['typ'].index(b'STD')

    yy = kspace['ky'][startidx::channels] - kymin
    zz = kspace[kzflag][startidx::channels] - kzmin
    tt = kspace['dyn'][startidx::channels]
    mm = kspace['mix'][startidx::channels]
    aa = kspace['aver'][startidx::channels]

    kshape = (kxmax - kxmin + 1, kymax - kymin + 1, kzmax - kzmin + 1, channels, mixes, dynamic, signalavg)
    kspace_filled = np.zeros(kshape, dtype=np.complex64)

    for n, (y, z, m, t, a) in enumerate(zip(yy, zz, mm, tt, aa)):
        kspace_filled[:, y, z, :, m, t, a] = tmp[:, :, n]

    return kspace_filled


def preprocess_volume(kspace, sense, slice_range, device='cuda'):
    """

    Parameters
    ----------
    kspace :
    sense :
    slice_range :
    device :

    Returns
    -------

    """
    raw_kspace = torch.from_numpy(kspace).squeeze().to(device)
    del kspace

    if raw_kspace.shape[-1] == 2:
        raw_kspace = raw_kspace[..., 0] + 1j * raw_kspace[..., 1]

    raw_sense = torch.from_numpy(sense).squeeze()

    if raw_sense.shape[-1] == 2:
        raw_sense = raw_sense[..., 0] + 1j * raw_sense[..., 1]

    transversal = [raw_kspace, raw_sense]
    sagittal = [raw_kspace.permute(1, 0, 2, 3), raw_sense.permute(1, 0, 2, 3)]
    coronal = [raw_kspace.permute(2, 0, 1, 3), raw_sense.permute(2, 0, 1, 3)]
    del raw_kspace, raw_sense

    imspaces = []
    kspaces = []
    sensitivities_maps = []
    masks = []
    for i, view in enumerate([transversal, sagittal, coronal]):
        raw_kspace, sense = view[0], view[1]

        mask = complex_tensor_to_real_np(extract_mask(raw_kspace))
        imspace = fft.ifftshift(normalize(fft.ifftn(raw_kspace.detach().cpu(), dim=(0, 1, 2), norm="ortho")).to(device),
                                dim=0 if i == 0 else 1).detach().cpu()
        del raw_kspace
        torch.cuda.empty_cache()

        sensitivity_map = estimate_csm(sense.numpy(), calibration_region_size=20)
        del sense

        if slice_range is not None:
            imspace = slice_selection(imspace, slice_range[0], slice_range[1])
            sensitivity_map = slice_selection(sensitivity_map, slice_range[0], slice_range[1])

        kspace = fft.fft2(imspace.to(device), dim=(1, 2), norm="ortho").detach().cpu()

        validate_sense = torch.sum(sensitivity_map)
        if torch.abs(validate_sense) == torch.tensor(0) or torch.isnan(validate_sense):
            sensitivity_map = normalize(estimate_csm(kspace.numpy(), calibration_region_size=20)).squeeze()
        else:
            sensitivity_map = resize_sensitivity_map(sensitivity_map.to(device), imspace.shape)
            sensitivity_map = fft.ifftshift(normalize(sensitivity_map.detach().cpu())).to(device)

            if sensitivity_map.shape[-1] < imspace.shape[-1]:
                expand_sensitivity_map = torch.sum(sensitivity_map, -1).unsqueeze(-1)
                for i in range(imspace.shape[-1] - sensitivity_map.shape[-1]):
                    sensitivity_map = torch.cat((sensitivity_map, expand_sensitivity_map), -1)
                del expand_sensitivity_map
        torch.cuda.empty_cache()

        imspaces.append(imspace.detach().cpu())
        kspaces.append(kspace.detach().cpu())
        sensitivities_maps.append(sensitivity_map.detach().cpu())
        masks.append(mask)

    return kspaces, masks, imspaces, sensitivities_maps
