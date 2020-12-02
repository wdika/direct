# encoding: utf-8
__author__ = 'Dimitrios Karkalousos'

import argparse
import glob
import logging
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from projects.tecfidera.utils import readcfl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(args):
    start_time = time.perf_counter()
    logger.info("Saving data. This might take some time, please wait...")

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

                kspace = readcfl(k)
                mask = np.where(np.sum(np.sum(np.abs(kspace), 0), -1) > 0., 1, 0)

                output_dir = args.output + '/' + subject.split('/')[-2] + '/' + scan.split('/')[-2] + '/' + name + '/'

                if args.export_type == 'png':
                    Path(output_dir + '/png/').mkdir(parents=True, exist_ok=True)
                    plt.imshow(mask, cmap='gray')
                    plt.savefig(output_dir + '/png/mask.png')
                    plt.close()

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
