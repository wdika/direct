# encoding: utf-8
__author__ = 'Dimitrios Karkalousos'

import os
import argparse
import pathlib
import sys
import time
from tqdm import tqdm
import multiprocessing
import logging
import glob

import colorcet as cc
import h5py
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def readcfl(name):
    logger.info("Constructing a numpy array. This might take some time, please wait...")
    start_time = time.perf_counter()

    h = open(name + ".hdr", "r")
    h.readline()  # skip
    l = h.readline()
    h.close()
    dims = [int(i) for i in l.split()]

    # remove singleton dimensions from the end
    n = int(np.prod(dims))
    dims_prod = np.cumprod(dims)
    dims = dims[:np.searchsorted(dims_prod, n) + 1]

    # load data and reshape into dims
    d = open(name + ".cfl", "r")
    a = np.fromfile(d, dtype=np.complex64, count=n)
    d.close()
    a = a.reshape(dims, order='F')  # column-major

    time_taken = time.perf_counter() - start_time
    logger.info(f"Done! Run Time = {time_taken:}s")

    return a


def save_outputs(idx):
    if idx == 0:
        plane = 'axial'
    elif idx == 1:
        plane = 'transversal'
    elif idx == 2:
        plane = 'sagittal'

    for i in tqdm(range(data[idx].shape[0])):
        plt.imshow(data[idx][i], cmap='gray')
        plt.savefig(args.output_path + '/' + plane + '/' + str(i) + '.png')
        plt.close()
	

def preprocess_vol(csm):
    logger.info("Preprocessing data. This might take some time, please wait...")
    start = time.perf_counter()

    logger.info("Processing the axial plane...")
    axial_target = np.abs(np.sqrt(np.sum(csm**2, -1)))

    logger.info("Processing the transversal plane...")
    transversal_target = np.abs(np.sqrt(np.sum(np.transpose(csm, (1, 0, 2, 3))**2, -1)))

    logger.info("Processing the sagittal plane...")
    sagittal_target = np.abs(np.sqrt(np.sum(np.transpose(csm, (2, 0, 1, 3))**2, -1)))

    time_taken = time.perf_counter() - start
    logger.info(f"Done! Run Time = {time_taken:}s")

    return axial_target, transversal_target, sagittal_target


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
            
            Path(args.output_path + '/axial/').mkdir(parents=True, exist_ok=True)
            Path(args.output_path + '/sagittal/').mkdir(parents=True, exist_ok=True)
            Path(args.output_path + '/transversal/').mkdir(parents=True, exist_ok=True)

            data = preprocess_vol(readcfl(f))

            main(args.num_workers)

