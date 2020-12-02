# encoding: utf-8
__author__ = 'Dimitrios Karkalousos'

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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocess_png_vol(csm):
    logger.info("Preprocessing data. This might take some time, please wait...")
    start = time.perf_counter()

    logger.info("Processing the axial plane...")
    axial_target = np.abs(np.sqrt(np.sum(csm ** 2, -1)))

    for i in tqdm(range(axial_target.shape[0])):
        plt.imshow(axial_target[i], cmap='gray')
        plt.savefig(args.output_dir + '/axial/' + str(i) + '.png')
        plt.close()

    # logger.info("Processing the transversal plane...")
    # transversal_target = np.abs(np.sqrt(np.sum(np.transpose(csm, (1, 0, 2, 3))**2, -1)))
    #
    # logger.info("Processing the sagittal plane...")
    # sagittal_target = np.abs(np.sqrt(np.sum(np.transpose(csm, (2, 0, 1, 3))**2, -1)))

    time_taken = time.perf_counter() - start
    logger.info(f"Done! Run Time = {time_taken:}s")

    return axial_target  # , transversal_target, sagittal_target


def main(num_workers, export_type):
    with multiprocessing.Pool(num_workers) as pool:
        start_time = time.perf_counter()
        logger.info("Saving data. This might take some time, please wait...")

        if export_type == 'png':
            pool.map(preprocess_png_vol, range(len(data)))
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

    subjects = glob.glob(args.root + "/*/")
    logger.info(f"Total scans: {len(subjects)}")

    for subject in subjects:
        logger.info(f"Processing subject: {subject.split('/')[-2]}")
        scans = glob.glob(subject + "/*/")
        logger.info(f"Total scans: {len(scans)}")

        for scan in scans:
            logger.info(f"Processing scan: {scan.split('/')[-2]}")
            kspaces = glob.glob(scan + "*csm.cfl")

            logger.info(f"Total volumes: {len(scan)}")

            for k in kspaces:
                k = k.split('.')[0]
                name = k.split('/')[-1]
                logger.info(f"Processing volume: {k.split('/')[-1]}")

                args.output_dir = args.output + '/' + subject.split('/')[-2] + '/' + scan.split('/')[
                    -2] + '/' + name + '/'

                if args.export_type == 'png':
                    args.output_dir = args.output_dir + '/png/images/'
                    Path(args.output_dir + '/axial/').mkdir(parents=True, exist_ok=True)
                    # Path(args.output_dir + '/sagittal/').mkdir(parents=True, exist_ok=True)
                    # Path(args.output_dir + '/transversal/').mkdir(parents=True, exist_ok=True)

                # data = preprocess_vol(readcfl(k))
                data = readcfl(k)

                main(args.num_workers, args.export_type)
