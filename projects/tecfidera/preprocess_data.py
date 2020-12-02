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


def save_png_outputs(idx, output_dir):
    # for i in tqdm(range(data.shape[0])):
    #     plt.imshow(data[i], cmap='gray')
    #     plt.savefig(output_dir + str(i) + '.png')
    #     plt.close()

    plt.imshow(idx, cmap='gray')
    plt.savefig(output_dir + str(idx) + '.png')
    plt.close()

def preprocess_vol(kspace, output_dir, num_workers=32):
    logger.info("Preprocessing data. This might take some time, please wait...")
    start = time.perf_counter()

    kspace = T.to_tensor(kspace).refine_names('slice', 'height', 'width', 'coil', 'complex')

    logger.info("Processing the axial plane...")
    # axial_imspace = np.fft.ifftn(kspace, axes=(0, 1, 2))
    # axial_target = np.sqrt(np.sum(axial_imspace ** 2, -1))
    # del axial_imspace

    axial_imspace = torch.fft.ifftn(kspace.rename(None), dim=(0, 1, 2), norm="ortho")
    # axial_imspace = np.fft.ifftn(T.tensor_to_complex_numpy(kspace), axes=(0, 1, 2))
    axial_target = np.abs(T.root_sum_of_squares(axial_imspace.refine_names('slice', 'height', 'width', 'coil', 'complex')).detach().cpu().numpy())

    with multiprocessing.Pool(num_workers) as pool:
        # pool.map(save_png_outputs(axial_target, output_dir=output_dir + '/axial/'),  range(len(axial_target)))
        pool.map(save_png_outputs,  tqdm(range(len(axial_target))))

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
    return axial_target


def main(args):
    start_time = time.perf_counter()
    logger.info("Saving data. This might take some time, please wait...")

    subjects = glob.glob(args.root + "/*/")
    logger.info(f"Total subjects: {len(subjects)}")

    for subject in subjects:
        logger.info(f"Processing subject: {subject.split('/')[-2]}")
        acquisitions = glob.glob(subject + "/*/")
        logger.info(f"Total acquisitions: {len(acquisitions)}")

        for acquisition in acquisitions:
            logger.info(f"Processing scan: {acquisition.split('/')[-2]}")
            #scans = glob.glob(acquisition + "*kspace.cfl")
            scans = glob.glob(acquisition + "*.cfl")
            logger.info(f"Total scans: {len(scans)}")

            for scan in scans:
                kspace = scan.split('.')[0]
                name = kspace.split('/')[-1].split('_')[0]
                logger.info(f"Processing scan: {name}")

                if args.export_type == 'png':
                    output_dir = args.output + '/png/' + subject.split('/')[-2] + '/' + acquisition.split('/')[
                        -2] + '/' + name + '/targets/'
                    Path(output_dir + '/axial/').mkdir(parents=True, exist_ok=True)
                    # Path(args.output_dir + '/sagittal/').mkdir(parents=True, exist_ok=True)
                    # Path(args.output_dir + '/transversal/').mkdir(parents=True, exist_ok=True)
                    preprocess_vol(readcfl(kspace), output_dir)

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
    main(args)
