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
import torch
from torch.fft import ifftn
from tqdm import tqdm

from direct.data import transforms as T
from projects.tecfidera.utils import readcfl

# import h5py

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_png_outputs(data, output_dir):
    for i in range(data.shape[0]):
        plt.imshow(data[i], cmap='gray')
        plt.savefig(output_dir + str(i) + '.png')
        plt.close()


def preprocess_vol(kspace, csm, output_dir):
    axial_imspace = T.fftshift(ifftn(kspace, dim=(0, 1, 2), norm="ortho"), dim=(0))

    axial_target = torch.abs(torch.sum(axial_imspace * torch.conj(csm), -1)).detach().cpu().numpy()
    axial_csm = torch.abs(torch.sum(torch.conj(csm), -1)).detach().cpu().numpy()

    Process(target=save_png_outputs, args=(axial_target, output_dir + '/axial/targets/')).start()
    Process(target=save_png_outputs, args=(axial_csm, output_dir + '/axial/csms/')).start()


def main(args):
    start_time = time.perf_counter()
    logger.info("Saving data. This might take some time, please wait...")

    subjects = glob.glob(args.root + "/*/")
    logger.info(f"Total subjects: {len(subjects)}")

    for subject in tqdm(subjects):
        acquisitions = glob.glob(subject + "/*/")

        for acquisition in acquisitions:
            kspaces = glob.glob(acquisition + "*kspace.cfl")
            # scans = glob.glob(acquisition + "*.cfl")

            for kspace in kspaces:
                kspace = kspace.split('.')[0]
                name = kspace.split('/')[-1].split('_')[0]
                csm = kspace.split('_')[0] + '_csm'

                logger.info(f"Processing subject: {subject.split('/')[-2]} | acquisition: {acquisition.split('/')[-2]}"
                            f"| scan: {name}")

                input_kspace = torch.from_numpy(readcfl(kspace)).to(args.device)
                input_csm = torch.from_numpy(readcfl(csm)).to(args.device)

                if args.export_type == 'png':
                    output_dir = args.output + '/png/' + subject.split('/')[-2] + '/' + acquisition.split('/')[
                        -2] + '/' + name
                    Path(output_dir + '/axial/targets/').mkdir(parents=True, exist_ok=True)
                    Path(output_dir + '/axial/csms/').mkdir(parents=True, exist_ok=True)

                    preprocess_vol(input_kspace, input_csm, output_dir)

    time_taken = time.perf_counter() - start_time
    logger.info(f"Done! Run Time = {time_taken:}s")


def create_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('root', type=str, help='Root dir containing folders with cfl files.')
    parser.add_argument('output', type=str, help='Output dir to save files.')
    parser.add_argument('--export-type', choices=['h5', 'png'], default='png', help='Choose output format.')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda', help='Enable GPU.')

    return parser


if __name__ == '__main__':
    args = create_arg_parser().parse_args(sys.argv[1:])

    main(args)
