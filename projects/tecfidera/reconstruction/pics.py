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
from projects.tecfidera.preprocessing.utils import complex_tensor_to_complex_np, normalize
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


class DataTransform:
    def __init__(self):
        pass

    def __call__(self, sample):
        # masked_kspace = T.tensor_to_complex_numpy(
        #     T.to_tensor(sample["kspace"]).rename(None).permute(1, 2, 0, 3).unsqueeze(0))
        #
        # sensitivity_map = T.tensor_to_complex_numpy(
        #     T.to_tensor(sample["sensitivity_map"]).rename(None).permute(1, 2, 0, 3).unsqueeze(0))

        return sample["kspace"], sample["sensitivity_map"], sample["filename"], sample["slice_no"]


def compute_pics_recon(masked_kspace, sensitivity_map, reg=0.01):
    """
    Run Parallel Imaging Compressed Sensing algorithm using the BART toolkit.
    """
    kspace = complex_tensor_to_complex_np(torch.from_numpy(fftshift(masked_kspace, dim=(1, 2))).permute(1, 2, 0).unsqueeze(0))
    sense = complex_tensor_to_complex_np(torch.from_numpy(fftshift(sensitivity_map, dim=(1, 2))).permute(1, 2, 0).unsqueeze(0))

    pred = bart(1, f'pics -g -i 200 -S -l1 -r {reg}', kspace, sense)
    pred = normalize(complex_tensor_to_complex_np(torch.from_numpy(fftshift(pred, dim=(1, 2)))))[0]

    plot = True
    if plot:
        import matplotlib.pyplot as plt
        target = normalize(np.sum(sensitivity_map.conj() * np.fft.ifftn(masked_kspace, axes=(1, 2)), 0))
        sense = normalize(np.sum(sensitivity_map.conj(), 0))

        plt.subplot(1, 6, 1)
        plt.imshow(np.abs(target), cmap='gray')
        plt.colorbar()
        plt.subplot(1, 6, 2)
        plt.imshow(np.angle(target), cmap='gray')
        plt.colorbar()
        plt.subplot(1, 6, 3)
        plt.imshow(np.abs(sense), cmap='gray')
        plt.colorbar()
        plt.subplot(1, 6, 4)
        plt.imshow(np.angle(sense), cmap='gray')
        plt.colorbar()
        plt.subplot(1, 6, 5)
        plt.imshow(np.abs(pred), cmap='gray')
        plt.colorbar()
        plt.subplot(1, 6, 6)
        plt.imshow(np.angle(pred), cmap='gray')
        plt.colorbar()
        plt.show()

    return torch.from_numpy(pred[0])


def run_model(idx):
    masked_kspace, sensitivity_map, filename, slice_no = data[idx]
    return filename, slice_no, compute_pics_recon(masked_kspace, sensitivity_map)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def main(num_workers):
    with multiprocessing.Pool(num_workers) as pool:
        start_time = time.perf_counter()
        outputs = pool.map(run_model, range(len(data)))
        time_taken = time.perf_counter() - start_time
        logging.info(f"Run Time = {time_taken:}s")
        save_outputs(outputs, args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=Path, default=None, help="Path to root")
    parser.add_argument("sensitivity_maps_root", type=Path, default=None, help="Path to sensitivity_maps root")
    parser.add_argument("output_path", type=Path, default=None, help="Path to save the sensitivity maps to", )
    parser.add_argument("--json_path", type=Path, default=None, help="Path to saved json file with reg values", )
    parser.add_argument("--seed", default=42, type=int, help="Seed for random number generators.")
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--filter', type=str, default=None, help='Filter filenames with this string')
    parser.add_argument('--num-groups', type=int, default=1,
                        help='Number of data samples to store in memory before writing')
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
        data = TECFIDERADataset(root=args.data_root, transform=DataTransform(),
                                sensitivity_maps=args.sensitivity_maps_root)
        main(args.num_workers)
