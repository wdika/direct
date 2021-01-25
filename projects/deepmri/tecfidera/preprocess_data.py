# encoding: utf-8
__author__ = 'Dimitrios Karkalousos'

import argparse
import glob
import logging
import os
import pickle
import sys
import time
from multiprocessing import Process
from pathlib import Path

import numpy as np
from tqdm import tqdm

os.environ['TOOLBOX_PATH'] = "/home/dkarkalousos/bart-0.6.00/"
sys.path.append('/home/dkarkalousos/bart-0.6.00/python/')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


def create_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_pickle_outputs(data, filename):
    for slice in range(data.shape[0]):
        with open(filename + '_' + str(slice), 'wb') as f:
            pickle.dump(data[slice], f, protocol=pickle.HIGHEST_PROTOCOL)


def center_crop(data, shape):
    if not (0 < shape[0] <= data.shape[1]) or not (0 < shape[1] <= data.shape[2]):
        raise ValueError(f"Crop shape should be smaller than data. Requested {shape}, got {data.shape}.")

    width_lower = (data.shape[1] - shape[0]) // 2
    width_upper = width_lower + shape[0]
    height_lower = (data.shape[2] - shape[1]) // 2
    height_upper = height_lower + shape[1]

    return data[:, width_lower:width_upper, height_lower:height_upper, :]


def preprocessing(root, output, export_type, scale):
    """
    Parses all subjects, acquisitions, and scans. Performs all the necessary preprocessing steps for the TECFIDERA data.

    Parameters
    ----------
    root : root directory of containing cfl data
    output : output directory to save data
    skip_csm : toggle this option to skip storing the csms
    export_type : h5 or png
    device : cuda or cpu

    """
    subjects = glob.glob(root + "/*/")

    for subject in tqdm(subjects):
        acquisitions = glob.glob(subject + "/*/")

        for acquisition in acquisitions:
            kspaces = glob.glob(acquisition + "*kspace.cfl")

            for filename_kspace in kspaces:
                name = filename_kspace.split('.')[0].split('/')[-1].split('_')[0]

                if name != '501':  # exclude the sense ref scan
                    name = 'AXFLAIR' if name == '301' else 'AXT1_MPRAGE'

                    # fixed number of slices, selected after checking the pngs
                    start = 17 if name == 'AXFLAIR' else 22
                    end = 217 if name == 'AXFLAIR' else 222

                    logger.info(
                        f"Processing subject: {subject.split('/')[-2]} | time-point: {acquisition.split('/')[-2]}"
                        f" | acquisition: {name}")

                    input_kspace = readcfl(filename_kspace.split('.')[0])
                    mask = np.expand_dims(np.where(np.sum(np.sum(np.abs(input_kspace), 0), -1) > 0, 1, 0), 0)

                    acceleration = np.round(mask.size / mask.sum())
                    if acceleration < 4:
                        acceleration = 4.0
                    elif acceleration > 4 and acceleration < 6:
                        acceleration = 6.0
                    elif acceleration > 6 and acceleration < 8:
                        acceleration = 8.0
                    elif acceleration > 8 and acceleration < 10:
                        acceleration = 10.0

                    imspace = np.fft.ifft2(input_kspace, axes=(1, 2))
                    csm = readcfl(filename_kspace.split('_')[0] + '_csm')

                    # TODO (kp, dk): Find out if scaling and this normalization is necessary here or in the dataloader.
                    imspace = imspace * scale
                    csm = csm * scale

                    imspace = imspace / np.amax(np.abs(np.sum(imspace * csm.conj(), 2)).real.flatten())

                    data = np.stack((imspace, csm), 1)

                    # Normalize data
                    # TODO (dk, kp) : make sure about the csm normalization. Here it seems the csm is normalized.

                    if export_type == 'pickle':
                        name = subject.split('/')[-2] + '_' + acquisition.split('/')[-2] + '_' + name

                        # Save kspace
                        output_dir = output + '/test/'
                        create_dir(output_dir)
                        Process(target=save_pickle_outputs, args=(data, output_dir + name)).start()
                        # del imspace

                        # Save mask
                        output_dir_mask = output + '/masks_' + subject.split('/')[-2] + '/acc' + str(acceleration)
                        create_dir(output_dir_mask)
                        with open(output_dir_mask + '/mask0', 'wb') as f:
                            pickle.dump(mask, f)
                        del mask


def main(args):
    start_time = time.perf_counter()
    logger.info("Saving data. This might take some time, please wait...")
    preprocessing(args.root, args.output, args.export_type, args.scale)
    time_taken = time.perf_counter() - start_time
    logger.info(f"Done! Run Time = {time_taken:}s")


def create_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('root', type=str, help='Root dir containing folders with cfl files.')
    parser.add_argument('output', type=str, help='Output dir to save files.')
    parser.add_argument('--export-type', choices=['pickle', 'png'], default='pickle', help='Choose output format.')
    parser.add_argument('--scale', type=float, default=1.0, help='Set scaling of the data.')

    return parser


if __name__ == '__main__':
    args = create_arg_parser().parse_args(sys.argv[1:])
    main(args)
