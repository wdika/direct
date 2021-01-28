# encoding: utf-8
__author__ = 'Dimitrios Karkalousos'

import argparse
import glob
import logging
import os
import sys
import time
from multiprocessing import Process
from pathlib import Path

import numpy as np
import pickle5 as pickle
from tqdm import tqdm

os.environ['TOOLBOX_PATH'] = "/home/dkarkalousos/bart-0.6.00/"
sys.path.append('/home/dkarkalousos/bart-0.6.00/python/')
from bart import bart

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_pickle_outputs(data, filename):
    for slice in range(data.shape[0]):
        with open(filename + '_' + str(slice), 'wb') as f:
            pickle.dump(data[slice], f, protocol=pickle.HIGHEST_PROTOCOL)


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

    sensmap = np.expand_dims(sensmap[0], 0)

    sensEstResize = bart(1, 'fft 7', sensmap)
    sensEstResize = bart(1, 'resize -c 0 ' + str(xNew) + ' 1 ' + str(yNew) + ' 2 ' + str(zNew), sensEstResize)
    sensEstResize = bart(1, 'fft -i 7', sensEstResize)

    sensEstResize_np = np.fft.fftn(sensmap, axes=(0, 1, 2))
    sensEstResize_np = bart(1, 'resize -c 0 ' + str(xNew) + ' 1 ' + str(yNew) + ' 2 ' + str(zNew), sensEstResize_np)
    sensEstResize_np = np.fft.ifftn(sensEstResize_np, axes=(0, 1, 2))

    import matplotlib.pyplot as plt
    plt.subplot(1, 3, 1)
    plt.imshow(np.abs(np.sqrt(np.sum(sensmap**2, -1)))[0],          cmap='gray')
    plt.subplot(1, 3, 2)
    plt.imshow(np.abs(np.sqrt(np.sum(sensEstResize**2, -1)))[0],    cmap='gray')
    plt.subplot(1, 3, 3)
    plt.imshow(np.abs(np.sqrt(np.sum(sensEstResize_np**2, -1)))[0], cmap='gray')
    plt.show()

    return sensEstResize


def preprocessing(root, output):
    """
    Parses all subjects, acquisitions, and scans. Performs all the necessary preprocessing steps for the TECFIDERA data.

    Parameters
    ----------
    root :   root directory of containing cfl data
    output : output directory to save data
    """
    subjects = glob.glob(root + "/*/")
    for subject in tqdm(subjects):
        acquisitions = glob.glob(subject + "/*/")
        for acquisition in acquisitions:
            data = glob.glob(acquisition + "*.list")
            for filename in data:
                name = filename.split('.')[0].split('/')[-1].split('_')[1]
                if name != '501':  # exclude the sense ref scan
                    name = 'AXFLAIR' if name == '301' else 'AXT1_MPRAGE'

                    logger.info(
                        f"Processing subject: {subject.split('/')[-2]} | time-point: {acquisition.split('/')[-2]}"
                        f" | acquisition: {name}")

                    kspace = np.squeeze(get_kspace_from_listdata(filename.split('.')[0]))
                    mask = np.expand_dims(np.where(np.sum(np.sum(np.abs(kspace), 0), -1) > 0, 1, 0), 0)

                    #imspace = np.fft.ifftn(kspace, axes=(0, 1, 2))
                    #if imspace.shape[-1] == 2:
                        #imspace = imspace[..., 0] + 1j * imspace[..., 1]

                    sense = resize_sensitivity_map(np.fft.ifftshift(bart(1, f"caldir 20",
                            get_kspace_from_listdata('/'.join(filename.split('.')[0].split('/')[:-1]) + '/raw_501')),
                                                                                        axes=(0, 1, 2)), kspace.shape)

                    #imspace = imspace / np.abs(np.max(np.sum(imspace * sense.conj(), -1)))
                    #imspace = imspace / np.abs(np.max(imspace))
                    #sense = sense / np.abs(np.max(sense))

                    # Save data
                    output_dir = output + '/test/'
                    create_dir(output_dir)
                    Process(target=save_pickle_outputs, args=(np.stack((imspace, sense), 1), output_dir + \
                                      subject.split('/')[-2] + '_' + acquisition.split('/')[-2] + '_' + name)).start()

                    # Save mask
                    acceleration = np.round(mask.size / mask.sum())

                    if acceleration < 4:
                        acceleration = 4.0
                    elif 4 < acceleration < 6:
                        acceleration = 6.0
                    elif 6 < acceleration < 8:
                        acceleration = 8.0
                    elif 8 < acceleration < 10:
                        acceleration = 10.0

                    output_dir_mask = output + '/masks_' + subject.split('/')[-2] + '/' + acquisition.split('/')[
                        -2] + '/' + name + '/acc' + str(acceleration)
                    create_dir(output_dir_mask)
                    with open(output_dir_mask + '/mask0', 'wb') as f:
                        pickle.dump(mask, f)


def main(args):
    start_time = time.perf_counter()
    logger.info("Converting data. This might take some time, please wait...")
    preprocessing(args.root, args.output)
    time_taken = time.perf_counter() - start_time
    logger.info(f"Done! Run Time = {time_taken:}s")


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=str, help='Root dir containing folders with raw files.')
    parser.add_argument('output', type=str, help='Output dir to save data as pickle.')
    return parser


if __name__ == '__main__':
    args = create_arg_parser().parse_args(sys.argv[1:])
    main(args)
