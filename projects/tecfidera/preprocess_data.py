# encoding: utf-8
__author__ = 'Dimitrios Karkalousos'

import argparse
import glob
import logging
import sys
import time
from multiprocessing import Process
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from torch.fft import ifftn
import torch
from tqdm import tqdm

from projects.tecfidera.utils import readcfl
from direct.data import transforms as T

# import h5py

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_png_outputs(data, output_dir):
    for i in range(data.shape[0]):
        plt.imshow(data[i], cmap='gray')
        plt.savefig(output_dir + str(i) + '.png')
        plt.close()


def preprocess_vol(input_kspace, input_csm, output_dir):
    kspace = torch.from_numpy(input_kspace)
    # csm = torch.from_numpy(input_csm)

    kspace = T.ifftshift(kspace, dim=(0, 1, 2))
    axial_imspace = ifftn(kspace, dim=(0, 1, 2), norm="ortho").detach().cpu().numpy()
    axial_imspace = T.fftshift(axial_imspace, dim=(0, 1, 2))
    # axial_csm = csm.refine_names('slice', 'height', 'width', 'coil')
    axial_target = np.abs(np.sum(axial_imspace * input_csm.conj(), -1))
    axial_csm = np.abs(np.sum(input_csm.conj(), -1))

    # transversal_imspace = np.fft.ifftshift(np.fft.ifftn(np.transpose(kspace, (1, 0, 2, 3)), axes=(0, 1, 2)), axes=1)
    # sagittal_imspace = np.transpose(
    #     np.fft.ifftshift(np.fft.ifftn(np.transpose(kspace, (2, 1, 0, 3)), axes=(0, 1, 2)), axes=2), (0, 2, 1, 3))

    Process(target=save_png_outputs, args=(axial_target, output_dir + '/axial/targets/')).start()
    Process(target=save_png_outputs, args=(axial_csm, output_dir + '/axial/csms/')).start()


def main(args):
    start_time = time.perf_counter()
    logger.info("Saving data. This might take some time, please wait...")

    subjects = glob.glob(args.root + "/*/")
    logger.info(f"Total subjects: {len(subjects)}")

    for subject in tqdm(subjects):
        logger.info(f"Processing subject: {subject.split('/')[-2]}")
        acquisitions = glob.glob(subject + "/*/")
        logger.info(f"Total acquisitions: {len(acquisitions)}")

        for acquisition in acquisitions:
            logger.info(f"Processing acquisition: {acquisition.split('/')[-2]}")
            kspaces = glob.glob(acquisition + "*kspace.cfl")
            # scans = glob.glob(acquisition + "*.cfl")
            logger.info(f"Total scans: {len(kspaces)}")

            for kspace in kspaces:
                kspace = kspace.split('.')[0]
                name = kspace.split('/')[-1].split('_')[0]
                csm = kspace.split('_')[0] + '_csm'

                logger.info(f"Processing scan: {name}")

                if args.export_type == 'png':
                    output_dir = args.output + '/png/' + subject.split('/')[-2] + '/' + acquisition.split('/')[
                        -2] + '/' + name
                    Path(output_dir + '/axial/targets/').mkdir(parents=True, exist_ok=True)
                    Path(output_dir + '/axial/csms/').mkdir(parents=True, exist_ok=True)
                    # Path(args.output_dir + '/sagittal/').mkdir(parents=True, exist_ok=True)
                    # Path(args.output_dir + '/transversal/').mkdir(parents=True, exist_ok=True)
                    preprocess_vol(readcfl(kspace), readcfl(csm), output_dir)

    time_taken = time.perf_counter() - start_time
    logger.info(f"Done! Run Time = {time_taken:}s")


def create_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('root', type=str, help='Root dir containing folders with cfl files.')
    parser.add_argument('output', type=str, help='Output dir to save files.')
    parser.add_argument('--export-type', choices=['h5', 'png'], default='png', help='Choose output format.')

    return parser


if __name__ == '__main__':
    args = create_arg_parser().parse_args(sys.argv[1:])
    main(args)
