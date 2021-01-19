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

from direct.data.transforms import ifftshift
from projects.tecfidera.dataset import TECFIDERADataset
from projects.tecfidera.preprocessing.utils import *

os.environ['TOOLBOX_PATH'] = "/home/dkarkalousos/bart-0.6.00/"
sys.path.append('/home/dkarkalousos/bart-0.6.00/python/')
from bart import bart

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataTransform:
    def __init__(self):
        pass

    def __call__(self, sample):
        masked_kspace = complex_tensor_to_complex_np(torch.from_numpy(sample["kspace"]).permute(1, 2, 0).unsqueeze(0))

        # TODO : Find out why normalization works here but not when saving to h5 using the preprocess_data script.
        # masked_kspace = complex_tensor_to_complex_np(fftn(torch.from_numpy(normalize(complex_tensor_to_complex_np(
        #     ifftn(torch.from_numpy(sample["kspace"]).permute(1, 2, 0).unsqueeze(0), dim=(1, 2))))), dim=(1, 2)))

        # sensitivity_map = complex_tensor_to_complex_np(
        #     ifftshift(torch.from_numpy(sample["sensitivity_map"]).permute(1, 2, 0).unsqueeze(0), dim=(1, 2)))

        sensitivity_map = complex_tensor_to_complex_np(
            torch.from_numpy(sample["sensitivity_map"]).permute(1, 2, 0).unsqueeze(0))

        return masked_kspace, sensitivity_map, sample["filename"], sample["slice_no"]


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


def pics_recon(idx):
    kspace, sensitivity_map, filename, slice_no = data[idx]
    print(kspace.shape)

    # TODO (dk) : pics per slice appears to not working properly
    pred = normalize(complex_tensor_to_complex_np(ifftshift(torch.from_numpy(
        # bart(1, f'pics -g -i 200 -S -l1 -r 0.01', kspace, sensitivity_map)[0]),
        bart(1, f'pics -i 20 -S -l1 -r 0.01', kspace, sensitivity_map)[0]),
        dim=(0, 1))))

    import matplotlib.pyplot as plt
    imspace = np.fft.ifft2(kspace, axes=(1, 2))
    rss_target = np.sqrt(np.sum(imspace ** 2, -1))[0]

    sensitivity_map = np.fft.ifftshift(sensitivity_map, axes=(1, 2))
    target = np.sum(sensitivity_map.conj() * imspace, -1)[0]
    sense = np.sqrt(np.sum(sensitivity_map ** 2, -1))[0]

    print('imspace', np.max(np.abs(imspace)), np.min(np.abs(imspace)))
    print('sensitivity_map', np.max(np.abs(sensitivity_map)), np.min(np.abs(sensitivity_map)))
    print('target', np.max(np.abs(target)), np.min(np.abs(target)))
    print('rss_target', np.max(np.abs(rss_target)), np.min(np.abs(rss_target)))
    print('sense', np.max(np.abs(sense)), np.min(np.abs(sense)))
    print('pred', np.max(np.abs(pred)), np.min(np.abs(pred)))

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

    return filename, slice_no, pred


def main(num_workers):
    with multiprocessing.Pool(num_workers) as pool:
        start_time = time.perf_counter()
        outputs = pool.map(pics_recon, range(len(data)))
        time_taken = time.perf_counter() - start_time
        logging.info(f"Run Time = {time_taken:}s")
        save_outputs(outputs, args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=Path, default=None, help="Path to root.")
    parser.add_argument("sensitivity_maps_root", type=Path, default=None, help="Path to sensitivity_maps root.")
    parser.add_argument("output_path", type=Path, default=None, help="Path to save the pics recons.", )
    parser.add_argument("--seed", default=42, type=int, help="Seed for random number generators.")
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading.')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda', help='Enable GPU.')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data = TECFIDERADataset(root=args.data_root, transform=DataTransform(), sensitivity_maps=args.sensitivity_maps_root)
    main(args.num_workers)
