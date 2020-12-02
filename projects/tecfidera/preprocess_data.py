# encoding: utf-8
__author__ = 'Dimitrios Karkalousos'

import torch
import argparse
import glob
import logging
import multiprocessing
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from projects.tecfidera.utils import readcfl
import direct.data.transforms as T

# import h5py

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_png_outputs(data):
    # if idx == 0:
    #     plane = 'axial'
    # elif idx == 1:
    #     plane = 'transversal'
    # elif idx == 2:
    #     plane = 'sagittal'
    # else:
    #     plane = 'tmp'

    plane = 'axial'

    # for i in tqdm(range(data[idx].shape[0])):
    #     plt.imshow(data[idx][i], cmap='gray')
    #     plt.savefig(args.output_path + '/' + plane + '/' + str(i) + '.png')
    #     plt.close()

    for i in tqdm(range(data.shape[0])):
        plt.imshow(data[i], cmap='gray')
        plt.savefig(args.output_dir + '/' + plane + '/' + str(i) + '.png')
        plt.close()


def preprocess_vol(kspace):
    logger.info("Preprocessing data. This might take some time, please wait...")
    start = time.perf_counter()

    kspace = T.to_tensor(kspace).refine_names('slice', 'height', 'width', 'coil', 'complex')

    logger.info("Processing the axial plane...")
    # axial_imspace = np.fft.ifftn(kspace, axes=(0, 1, 2))
    # axial_target = np.abs(np.sqrt(np.sum(axial_imspace ** 2, -1)))
    # del axial_imspace

    axial_imspace = torch.fft.ifftn(kspace.rename(None), dim=(0, 1, 2), norm="ortho")
    axial_imspace = axial_imspace.refine_names('slice', 'height', 'width', 'coil', 'complex')

    # logger.info("Processing the transversal plane...")
    # transversal_imspace = np.fft.ifftshift(np.fft.ifftn(np.transpose(kspace, (1, 0, 2, 3)), axes=(0, 1, 2)), axes=1)
    # transversal_target = np.abs(np.sqrt(np.sum(transversal_imspace ** 2, -1)))
    # del transversal_imspace
    #
    # logger.info("Processing the sagittal plane...")
    # sagittal_imspace = np.transpose(
    #     np.fft.ifftshift(np.fft.ifftn(np.transpose(kspace, (2, 1, 0, 3)), axes=(0, 1, 2)), axes=2), (0, 2, 1, 3))
    # sagittal_target = np.abs(np.sqrt(np.sum(sagittal_imspace ** 2, -1)))
    # del sagittal_imspace

    time_taken = time.perf_counter() - start
    logger.info(f"Done! Run Time = {time_taken:}s")

    # return axial_target, transversal_target, sagittal_target
    return axial_imspace


def main(num_workers, export_type):
    #with multiprocessing.Pool(num_workers) as pool:
    start_time = time.perf_counter()
    logger.info("Saving data. This might take some time, please wait...")

    if export_type == 'png':
        # pool.map(save_png_outputs, range(len(data)))
        save_png_outputs(data)
    # else:
    #     pool.map(save_h5_outputs, range(len(data)))

    time_taken = time.perf_counter() - start_time
    logger.info(f"Done! Run Time = {time_taken:}s")


def create_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('root', type=str, help='Root dir containing folders with cfl files.')
    parser.add_argument('output', type=str, help='Output dir to save files.')
    parser.add_argument('--export-type', choices=['h5', 'png'], default='png', help='Choose output format.')
    parser.add_argument('--num-workers', type=int, default=32, help='Number of workers for data loading')

    return parser


if __name__ == '__main__':
    args = create_arg_parser().parse_args(sys.argv[1:])

    dirs = glob.glob(args.root + "/*/")
    logger.info(f"Total scans: {len(dirs)}")

    for folder in dirs:
        logger.info(f"Processing scan: {folder.split('/')[-2]}")
        files = glob.glob(folder + "*.cfl")

        subjects = glob.glob(args.root + "/*/")
        logger.info(f"Total scans: {len(subjects)}")

        for subject in subjects:
            logger.info(f"Processing subject: {subject.split('/')[-2]}")
            scans = glob.glob(subject + "/*/")
            logger.info(f"Total scans: {len(scans)}")

            for scan in scans:
                logger.info(f"Processing scan: {scan.split('/')[-2]}")
                kspaces = glob.glob(scan + "*kspace.cfl")

                logger.info(f"Total volumes: {len(scan)}")

                for k in kspaces:
                    k = k.split('.')[0]
                    name = k.split('/')[-1].split('_')[0]
                    logger.info(f"Processing volume: {name}")

                    args.output_dir = args.output + '/' + subject.split('/')[-2] + '/' + scan.split('/')[
                        -2] + '/' + name + '/'

                    targets = preprocess_vol(readcfl(k))

                    if args.export_type == 'png':
                        args.output_dir = args.output_dir + '/png/targets/'
                        Path(args.output_dir + '/axial/').mkdir(parents=True, exist_ok=True)
                        # Path(args.output_dir + '/sagittal/').mkdir(parents=True, exist_ok=True)
                        # Path(args.output_dir + '/transversal/').mkdir(parents=True, exist_ok=True)

                        data = np.abs(T.root_sum_of_squares(targets).detach().cpu().numpy())

                    main(args.num_workers, args.export_type)
