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


def create_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def complex_tensor_to_real_np(x):
    return torch.abs(x).detach().cpu().numpy()


def save_png_outputs(data, output_dir):
    create_dir(output_dir)

    for i in range(data.shape[0]):
        plt.imshow(data[i], cmap='gray')
        plt.savefig(output_dir + str(i) + '.png')
        plt.close()


def preprocessing_ifft(kspace):
    """

    Parameters
    ----------
    kspace : torch.Tensor

    Returns
    -------
    image space tensor of the axial plane transformed with the correct/fixed preprocessing steps
    """
    return T.fftshift(ifftn(kspace, dim=(0, 1, 2), norm="ortho"), dim=0)


def extract_mask(kspace):
    """

    Parameters
    ----------
    kspace : torch.Tensor

    Returns
    -------
    extracts the mask from the subsampled kspace, after summing the slice and the coil dimensions
    """
    return torch.where(torch.sum(torch.sum(torch.abs(kspace), 0), -1) > 0, 1, 0)


def sense_reconstruction(imspace, csm, dim=-1):
    """

    Parameters
    ----------
    imspace : torch.Tensor
    csm : torch.Tensor
    dim : coil dimension

    Returns
    -------
    reconstructed complex-valued image using SENSE
    """
    return torch.sum(imspace * torch.conj(csm), dim=dim)


def csm_sense_coil_combination(csm, dim=-1):
    """

    Parameters
    ----------
    csm : torch.Tensor
    dim : coil dimension

    Returns
    -------
    coil combined image
    """
    return torch.sum(torch.conj(csm), dim=dim)


def preprocessing(root, output, export_type, device):
    """
    Parses all subjects, acquisitions, and scans. Performs all the necessary preprocessing steps for the TECFIDERA data.

    Parameters
    ----------
    root : root directory of containing cfl data
    output : output directory to save data
    export_type : h5 or png
    device : cuda or cpu

    """
    subjects = glob.glob(root + "/*/")

    for subject in tqdm(subjects):
        acquisitions = glob.glob(subject + "/*/")

        for acquisition in acquisitions:
            kspaces = glob.glob(acquisition + "*kspace.cfl")

            for kspace in kspaces:
                kspace = kspace.split('.')[0]
                name = kspace.split('/')[-1].split('_')[0]
                csm = kspace.split('_')[0] + '_csm'

                logger.info(f"Processing subject: {subject.split('/')[-2]} | acquisition: {acquisition.split('/')[-2]}"
                            f" | scan: {name}")

                input_kspace = torch.from_numpy(readcfl(kspace)).to(device)
                input_csm = torch.from_numpy(readcfl(csm)).to(device)

                imspace = preprocessing_ifft(input_kspace)
                mask = extract_mask(input_kspace)

                if export_type == 'png':
                    output_dir = output + '/png/' + subject.split('/')[-2] + '/' + acquisition.split('/')[
                        -2] + '/' + name
                    create_dir(output_dir)

                    # Save target (SENSE reconstructed) png images
                    Process(target=save_png_outputs, args=(
                        complex_tensor_to_real_np(sense_reconstruction(imspace, input_csm, dim=-1)),
                        output_dir + '/axial/targets/')).start()

                    # Save sense coil combined png images
                    Process(target=save_png_outputs, args=(
                        complex_tensor_to_real_np(csm_sense_coil_combination(input_csm, dim=-1)),
                        output_dir + '/axial/csms/')).start()

                    # Save mask
                    plt.imshow(torch.abs(mask).detach().cpu().numpy(), cmap='gray')
                    plt.savefig(output_dir + '/axial/mask.png')
                    plt.close()


def main(args):
    start_time = time.perf_counter()
    logger.info("Saving data. This might take some time, please wait...")
    preprocessing(args.root, args.output, args.export_type, args.device)
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
