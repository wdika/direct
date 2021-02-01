# coding=utf-8
__author__ = 'Dimitrios Karkalousos'

import argparse
import logging
import multiprocessing
import random
import time
from collections import defaultdict

from projects.tecfidera.dataset import TECFIDERADataset
from projects.tecfidera.preprocessing.utils import *

os.environ['TOOLBOX_PATH'] = "/home/dkarkalousos/bart-0.6.00/"
sys.path.append('/home/dkarkalousos/bart-0.6.00/python/')
from bart import bart

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_outputs(outputs, output_path):
    """

    Parameters
    ----------
    outputs :
    output_path :

    Returns
    -------

    """
    create_dir(output_path)

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
    def __init__(self, device):
        self.device = device

    def __call__(self, sample):
        masked_kspace = complex_tensor_to_complex_np(torch.from_numpy(sample["kspace"]).permute(1, 2, 0).unsqueeze(0))
        sensitivity_map = complex_tensor_to_complex_np(
            fft.ifftshift(torch.from_numpy(sample["sensitivity_map"]).permute(1, 2, 0).unsqueeze(0), dim=(1, 2)))

        return masked_kspace, sensitivity_map, sample["filename"], sample["slice_no"], self.device


def pics_recon(idx):
    """

    Parameters
    ----------
    idx :

    Returns
    -------

    """
    kspace, sensitivity_map, filename, slice_no, device = data[idx]

    pred = fft.fftshift(torch.from_numpy(bart(1, f'pics -g -d0 -S -R W:7:0:0.005 -i 60', kspace,
                                              sensitivity_map) if device == 'cuda' else bart(1,
                                                                                             f'pics -d0 -S -R W:7:0:0.005 -i 60',
                                                                                             kspace,
                                                                                             sensitivity_map)).to(
        device), dim=(-2, -1))[0]
    pred = pred / torch.max(torch.abs(pred))

    if device == 'cuda':
        pred = pred.detach().cpu().numpy()

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

    data = TECFIDERADataset(root=args.data_root, transform=DataTransform(device=args.device),
                            sensitivity_maps=args.sensitivity_maps_root)
    main(args.num_workers)
