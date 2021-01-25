# encoding: utf-8
__author__ = 'Dimitrios Karkalousos'

import numpy as np


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
        _kprops = (r.split()[1:] for r in fid if r[0] == '.')
        for prop in _kprops:
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


def get_kspace_from_listdata(fdir, noise=None, remove_oversampling=False, remove_freq_oversampling=False, take_signal_avg=False):
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
