# coding=utf-8
__author__ = 'Dimitrios Karkalousos'

import argparse
import logging
import multiprocessing
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch

from direct.data.transforms import fftshift
from projects.tecfidera.preprocessing.utils import complex_tensor_to_complex_np, normalize, rss_normalize
from projects.tecfidera.dataset import TECFIDERADataset

os.environ['TOOLBOX_PATH'] = "/home/dkarkalousos/bart-0.6.00/"
sys.path.append('/home/dkarkalousos/bart-0.6.00/python/')
from bart import bart

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_outputs(outputs, output_path):
    pics_recons = defaultdict(list)

    for filename, slice_data, pred in outputs:
        pics_recons[filename].append((slice_data, pred))

    pics = {filename: np.stack([pred for _, pred in sorted(slice_preds)]) for filename, slice_preds in
            pics_recons.items()}

    for filename in pics:
        output_filename = (output_path / filename).with_suffix(".h5")
        logger.info(f"Writing: {output_filename}...")

        pics_recon = (pics[filename].astype(np.complex64))

        with h5py.File(output_filename, "w") as f:
            f["pics"] = pics_recon


def pics_recon(data, device, reg=0.01):
    """
    Run Parallel Imaging Compressed Sensing algorithm using the BART toolkit.
    """
    data[()]['sensitivity_map'] = normalize(data[()]['sensitivity_map'])

    for i in range(20, len(data)):
        masked_kspace = data[i]['kspace']
        sensitivity_map = data[i]['sensitivity_map']

        kspace = complex_tensor_to_complex_np(torch.from_numpy(masked_kspace).permute(1, 2, 0).unsqueeze(-2))
        sense = complex_tensor_to_complex_np(fftshift(torch.from_numpy(sensitivity_map), dim=(1, 2)).permute(1, 2, 0).unsqueeze(-2))

        pred = bart(1, f'pics -g -i 200 -S -l1 -r {reg}', kspace, sense)
        pred = complex_tensor_to_complex_np(fftshift(torch.from_numpy(pred), dim=(1, 2)))[0]

        plot = True
        if plot:
            import matplotlib.pyplot as plt
            imspace = np.fft.ifft2(masked_kspace, axes=(1, 2))

            target = np.sum(sensitivity_map.conj() * imspace, 0)
            sense = np.sum(sensitivity_map.conj(), 0)

            rss_target = np.sqrt(np.sum(imspace ** 2, 0))

            plt.subplot(2, 4, 1)
            plt.imshow(np.abs(rss_target), cmap='gray')
            plt.title('rss_target')
            plt.colorbar()
            plt.subplot(2, 4, 2)
            plt.imshow(np.angle(rss_target), cmap='gray')
            plt.title('rss_target phase')
            plt.colorbar()
            plt.subplot(2, 4, 3)
            plt.imshow(np.abs(sense), cmap='gray')
            plt.title('sense')
            plt.colorbar()
            plt.subplot(2, 4, 4)
            plt.imshow(np.angle(sense), cmap='gray')
            plt.title('sense phase')
            plt.colorbar()
            plt.subplot(2, 4, 5)
            plt.imshow(np.abs(target), cmap='gray')
            plt.title('ifft(masked_kspace) * sense.conj()')
            plt.colorbar()
            plt.subplot(2, 4, 6)
            plt.imshow(np.angle(target), cmap='gray')
            plt.title('ifft(masked_kspace) * sense.conj() phase')
            plt.colorbar()
            plt.subplot(2, 4, 7)
            plt.imshow(np.abs(pred), cmap='gray')
            plt.title('pics')
            plt.colorbar()
            plt.subplot(2, 4, 8)
            plt.imshow(np.angle(pred), cmap='gray')
            plt.title('pics phase')
            plt.colorbar()
            plt.show()

    return pred


def main(args):
    start_time = time.perf_counter()
    data = TECFIDERADataset(root=args.data_root, sensitivity_maps=args.sensitivity_maps_root)

    pics_recon(data=data, device=args.device)
    time_taken = time.perf_counter() - start_time
    logging.info(f"Run Time = {time_taken:}s")
    save_outputs(outputs, args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=Path, default=None, help="Path to root")
    parser.add_argument("sensitivity_maps_root", type=Path, default=None, help="Path to sensitivity_maps root")
    parser.add_argument("output_path", type=Path, default=None, help="Path to save the sensitivity maps to", )
    parser.add_argument("--seed", default=42, type=int, help="Seed for random number generators.")
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda', help='Enable GPU.')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    main(args)
