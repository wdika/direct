# encoding: utf-8
__author__ = 'Dimitrios Karkalousos'

import argparse
import logging
import os
import sys
import time

import numpy as np

os.environ['TOOLBOX_PATH'] = "/home/dkarkalousos/bart-0.6.00/"
sys.path.append('/home/dkarkalousos/bart-0.6.00/python/')
from bart import bart

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def center_crop(data, shape):
    assert 0 < shape[0] <= data.shape[-3]
    assert 0 < shape[1] <= data.shape[-2]

    w_from = (data.shape[-3] - shape[0]) // 2
    h_from = (data.shape[-2] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    return data[:, w_from:w_to, h_from:h_to, :]


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
    tmp = np.array([elem for elem, ktyp in zip(kspace['complexdata'], kspace['typ']) if ktyp == b'STD']).T
    tmp = tmp.reshape((sx, channels, -1), order='A')

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

        if np.fmod(n, np.around(len(yy) / 100)) == 0:
            print('Progress: {}%'.format(np.around((n / len(yy)) * 100)), end='\r')

    return kspace_filled


def resize_sensitivity_map(sensmap, newSize):
    [xNew, yNew, zNew, _] = newSize
    sensEstResize = bart(1, 'fft -i 7', bart(1, 'resize -c 0 ' + str(xNew) + ' 1 ' + str(yNew) + ' 2 ' + str(zNew),
                                                                                            bart(1, 'fft 7', sensmap)))
    return sensEstResize * (np.max(np.abs(sensmap)) / (np.max(np.abs(sensEstResize))))


def main(args):
    start_time = time.perf_counter()
    logger.info("Converting data. This might take some time, please wait...")
    kspace = get_kspace_from_listdata(args.root)
    sense = get_kspace_from_listdata('/'.join(args.root.split('/')[:-1]) + '/raw_501')

    imspace = np.fft.ifftn(np.squeeze(kspace), axes=(0, 1, 2))
    if imspace.shape[-1] == 2:
        imspace = imspace[..., 0] + 1j * imspace[..., 1]

    if sense.shape[-1] == 2:
        sense = sense[..., 0] + 1j * sense[..., 1]
    sense = np.fft.ifftshift(bart(1, f"caldir 50", sense), axes=(0, 1, 2))
    imspace = imspace / np.abs(np.max(imspace))
    sense = sense / np.abs(np.max(sense))

    print('imspace', np.abs(np.min(imspace)), np.abs(np.max(imspace)))
    print('sense', np.abs(np.min(sense)), np.abs(np.max(sense)))

    # sense = np.transpose(np.fft.ifftn(pad(input=torch.from_numpy(np.fft.fftn(np.transpose(sense, (3, 0, 1, 2)),
    # axes=(-2, -1))), pad=((imspace.shape[-2]-sense.shape[-2])//2, (imspace.shape[-2]-sense.shape[-2])//2,
    # (imspace.shape[-3]-sense.shape[-3])//2, (imspace.shape[-3]-sense.shape[-3])//2), mode='constant',
    # value=0).numpy(), axes=(-2, -1)), (1, 2, 3, 0))

    # if imspace.shape[1] > sense.shape[1] and imspace.shape[2] > sense.shape[2]:
    # cropped_imspace = center_crop(imspace, (sense.shape[1], sense.shape[2]))

    target = np.sum(center_crop(imspace, (sense.shape[1], sense.shape[2]))[0] * sense[0].conj(), -1)
    rss_target = np.sqrt(np.sum(imspace ** 2, -1))
    sense = np.sqrt(np.sum(sense ** 2, -1))

    import matplotlib.pyplot as plt
    # for i in range(rss_target.shape[0]):
    plt.subplot(1, 3, 1)
    plt.imshow(np.abs(target), cmap='gray')
    plt.subplot(1, 3, 2)
    plt.imshow(np.abs(rss_target)[0], cmap='gray')
    plt.subplot(1, 3, 3)
    plt.imshow(np.abs(sense)[0], cmap='gray')
    plt.show()

    time_taken = time.perf_counter() - start_time
    logger.info(f"Done! Run Time = {time_taken:}s")


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=str, help='Root dir containing folders with raw files.')
    return parser


if __name__ == '__main__':
    args = create_arg_parser().parse_args(sys.argv[1:])
    main(args)
