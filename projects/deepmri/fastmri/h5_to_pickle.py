"""
Copyright (c) Facebook, Inc. and its affiliates.

This input code is licensed under the MIT license found in the
LICENSE file in the root directory of this input tree.
"""
import os
import sys

os.environ['TOOLBOX_PATH'] = "/home/dkarkalousos/bart-0.6.00/"
sys.path.append('/home/dkarkalousos/bart-0.6.00/python/')

import time
import argparse

import logging
import multiprocessing
import random

import numpy as np
import torch

import bart
import direct.data.transforms as T
from direct.data.datasets import FastMRIDataset

from collections import defaultdict
from pathlib import Path
import pickle
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_outputs(outputs, output_path):
    data = defaultdict(list)

    for filename, slice_data, pred in outputs:
        data[filename].append((slice_data, pred))

    data = {filename: np.stack([pred for _, pred in sorted(slice_preds)]) for filename, slice_preds in
            data.items()}

    for filename in data:
        output_filename = str(output_path / filename.split('.')[0])
        logger.info(f"Writing: {output_filename}...")

        volume = data[filename]
        volume = np.stack((volume[:, 0, :, :, :] / np.amax(np.abs(np.sum(volume[:, 0, :, :, :] * \
                                                                         volume[:, 1, :, :, :].conj(),
                                                                         1)).real.flatten()), volume[:, 1, :, :, :]), 1)

        for slice in range(volume.shape[0]):
            with open(output_filename + '_' + str(slice), 'wb') as f:
                pickle.dump(volume[slice], f, protocol=pickle.HIGHEST_PROTOCOL)


class DataTransform:
    def __init__(self):
        pass

    def __call__(self, sample):
        return T.to_tensor(sample["kspace"]), sample["filename"], sample["slice_no"]


def create_data_loader(data_root, filenames_filter):
    return FastMRIDataset(root=data_root, sensitivity_maps=None, transform=DataTransform(),
                          filenames_filter=filenames_filter, pass_max=False)


def compute_sensitivity_maps(kspace):
    imspace = T.complex_center_crop(T.ifft2(kspace.refine_names("coil", "height", "width", "complex")), (300, 300))
    sens_maps = np.transpose(bart.bart(1, f"ecalib -d0 -m1 -S", T.tensor_to_complex_numpy(T.fft2(imspace).rename(None
                                                                                                                 ).permute(
        1, 2, 0, 3).unsqueeze(0)))[0], (2, 0, 1))

    plot = False
    if plot:
        target = imspace.refine_names("coil", "height", "width", "complex")
        sensitivities_map = T.to_tensor(sens_maps).rename(None).refine_names("coil", "height", "width", "complex")
        sense_recon = np.abs(
            T.tensor_to_complex_numpy(T.complex_multiplication(T.conjugate(sensitivities_map), target).sum("coil")))

        import matplotlib.pyplot as plt
        plt.subplot(1, 3, 1)
        plt.imshow(np.abs(T.root_sum_of_squares(target).cpu().numpy()), cmap='gray')
        plt.title('rss(ifft(kspace))')
        plt.subplot(1, 3, 2)
        plt.imshow(np.abs(T.tensor_to_complex_numpy(torch.sum(T.conjugate(sensitivities_map), 0))), cmap='gray')
        plt.title('sensitivities map')
        plt.subplot(1, 3, 3)
        plt.imshow(sense_recon, cmap='gray')
        plt.title('SENSE')
        plt.show()

    return np.stack((T.tensor_to_complex_numpy(imspace), sens_maps), 0)


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def run_model(idx):
    kspace, filename, slice_no = data[idx]
    return filename, slice_no, compute_sensitivity_maps(kspace)


def main(num_workers):
    with multiprocessing.Pool(num_workers) as pool:
        outputs = list(tqdm(pool.imap(run_model, range(len(data))), total=len(data)))
        save_outputs(outputs, args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=Path, default=None, help="Path to root")
    parser.add_argument("output_path", type=Path, default=None, help="Path to save the sensitivity maps to", )
    parser.add_argument("--seed", default=42, type=int, help="Seed for random number generators.")
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--filter', type=str, default=None, help='Filter filenames with this string')
    parser.add_argument('--num-groups', type=int, default=1, help='Number of data samples to store in m'
                                                                  'memory before writing')

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    all_h5s = list(args.data_root.glob("*.h5"))
    logger.info(f"Found {len(all_h5s)} h5s.")

    if args.filter:
        all_h5s = [_ for _ in all_h5s if args.filter in _.name]
        logger.info(f"Filter {args.filter} reduced to {len(all_h5s)} h5s.")

    output_available = [_.name for _ in list(args.output_path.glob("*.h5"))]
    all_h5s_ = []
    excluded = 0
    for fn in all_h5s:
        if fn.name not in output_available:
            all_h5s_.append(fn)
            continue
        excluded += 1

    logger.info(f"Excluded {excluded} already done.")

    # Chunk per volume
    h5s_per_group = list(chunks(all_h5s_, args.num_groups))

    for idx, group in enumerate(h5s_per_group):
        logger.info(f'{time.time()}: Group {idx} / {len(h5s_per_group)}')
        data = create_data_loader(args.data_root, filenames_filter=group)
        main(args.num_workers)
