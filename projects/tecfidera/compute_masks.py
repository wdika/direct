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

from projects.tecfidera.utils import readcfl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_outputs(idx):
    plt.imshow(data, cmap='gray')
    plt.savefig(args.output_path + '/mask_ ' + str(idx) + '.png')
    plt.close()


def preprocess_vol(kspace):
    logger.info("Preprocessing data. This might take some time, please wait...")
    start = time.perf_counter()

    mask = np.where(np.sum(np.sum(np.abs(kspace), 0), -1) > 0., 1, 0)

    time_taken = time.perf_counter() - start
    logger.info(f"Done! Run Time = {time_taken:}s")

    return mask


def main(num_workers):
    with multiprocessing.Pool(num_workers) as pool:
        start_time = time.perf_counter()
        logger.info("Saving data. This might take some time, please wait...")
        pool.map(save_outputs, range(len(data)))
        time_taken = time.perf_counter() - start_time
        logger.info(f"Done! Run Time = {time_taken:}s")


def create_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('root', type=str, help='Root dir containing folders with cfl files.')
    parser.add_argument('--num-workers', type=int, default=32, help='Number of workers for data loading')

    return parser


if __name__ == '__main__':
    args = create_arg_parser().parse_args(sys.argv[1:])

    dirs = glob.glob(args.root + "/*/")
    logger.info(f"Total scans: {len(dirs)}")

    for folder in dirs:
        logger.info(f"Processing scan: {folder.split('/')[-2]}")
        files = glob.glob(folder + "*.cfl")

        logger.info(f"Total volumes: {len(files)}")

        for f in files:
            f = f.split('.')[0]
            name = f.split('/')[-1]
            logger.info(f"Processing volume: {f.split('/')[-1]}")

            args.output_path = folder + '/png/' + name + '/'

            Path(args.output_path).mkdir(parents=True, exist_ok=True)

            data = preprocess_vol(readcfl(f))

            main(args.num_workers)