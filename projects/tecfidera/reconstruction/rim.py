# coding=utf-8
# Copyright (c) DIRECT Contributors
import functools
import logging
import os
import pathlib
import sys

import numpy as np
import torch

import direct.launch
from direct.data.mri_transforms import Compose
from direct.environment import Args
from direct.inference import setup_inference_save_to_h5, build_inference_transforms
from direct.utils import set_all_seeds

logger = logging.getLogger(__name__)


class CreateSamplingMask:
    """
    Create sampling mask from a dictionary.
    """

    def __init__(self, masks_dict):
        self.masks_dict = masks_dict

    def __call__(self, sample, **kwargs):
        sample["sampling_mask"] = self.masks_dict[sample["filename"]][np.newaxis, ..., np.newaxis]
        return sample


def inference_cfg_validation(cfg):
    # Some checks for the inference
    if cfg.inference.dataset.transforms.crop:
        logger.warning(
            f"cfg.inference.dataset.transforms.crop has to be None in inference. "
            f"Got {cfg.inference.dataset.transforms.crop}."
        )
        cfg.inference.dataset.transforms.crop = None

    if cfg.inference.dataset.transforms.image_center_crop:
        logger.warning(
            f"cfg.inference.dataset.transforms.image_center_crop has to be None in inference. "
            f"Got {cfg.inference.dataset.transforms.image_center_crop}."
        )
        cfg.inference.dataset.transforms.image_center_crop = None

    if cfg.inference.dataset.transforms.masking:
        logger.warning(
            f"cfg.inference.dataset.transforms.masking does not make sense in inference, but is set. "
            f"Got {cfg.inference.dataset.transforms.masking}."
        )
        cfg.inference.dataset.transforms.masking = None

    return cfg


def _get_transforms(masks_dict, env):
    dataset_cfg = env.cfg.inference.dataset
    transforms = build_inference_transforms(env, None, dataset_cfg)
    transforms = Compose([CreateSamplingMask(masks_dict), transforms])
    return dataset_cfg, transforms


if __name__ == "__main__":
    # This sets MKL threads to 1.
    # DataLoader can otherwise bring a lot of difficulties when computing CPU FFTs in the transforms.
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"

    # Remove warnings from named tensors being experimental
    os.environ["PYTHONWARNINGS"] = "ignore"

    epilog = f"""
        Examples:
        Run on single machine:
            $ {sys.argv[0]} data_root output_directory --checkpoint <checkpoint_num> --name <name> [--other-flags]
        Run on multiple machines:
            (machine0)$ {sys.argv[0]} data_root output_directory --checkpoint <checkpoint_num> --name <name> --masks <path to masks> --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
            (machine1)$ {sys.argv[0]} data_root output_directory --checkpoint <checkpoint_num> --name <name> --masks <path to masks> --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]
        """

    parser = Args(epilog=epilog)
    parser.add_argument(
        "data_root", type=pathlib.Path, help="Path to the inference data directory."
    )
    parser.add_argument(
        "output_directory", type=pathlib.Path, help="Path to the output directory."
    )
    parser.add_argument(
        "experiment_directory",
        type=pathlib.Path,
        help="Path to the directory with checkpoints and config.",
    )
    parser.add_argument(
        "--checkpoint",
        type=int,
        required=True,
        help="Number of an existing checkpoint.",
    )
    parser.add_argument(
        "--masks",
        type=pathlib.Path,
        required=True,
        help="Path to masks as generated by compute_masks.py.",
    )
    parser.add_argument(
        "--sensitivity_maps",
        type=pathlib.Path,
        required=True,
        help="Path to sensitivity_maps.",
    )
    parser.add_argument(
        "--filenames-filter",
        type=pathlib.Path,
        help="Path to list of filenames to parse.",
    )
    parser.add_argument("--name", help="Run name.", required=True, type=str)
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Config file for inference. "
             "Only use it to overwrite the standard loading of the config in the project directory.",
        required=False,
        type=pathlib.Path,
    )

    args = parser.parse_args()

    set_all_seeds(args.seed)

    # Process all masks
    all_maps = args.masks.glob("*.npy")
    logger.info("Loading masks...")
    masks_dict = {
        filename.name.replace(".npy", ".h5"): np.load(filename) for filename in all_maps
    }
    logger.info(f"Loaded {len(masks_dict)} masks.")

    setup_inference_save_to_h5 = functools.partial(
        setup_inference_save_to_h5, functools.partial(_get_transforms, masks_dict)
    )

    direct.launch.launch(
        setup_inference_save_to_h5,
        args.num_machines,
        args.num_gpus,
        args.machine_rank,
        args.dist_url,
        args.name,
        args.data_root,
        args.experiment_directory,
        args.output_directory,
        args.filenames_filter,
        args.sensitivity_maps,
        args.checkpoint,
        args.device,
        args.num_workers,
        args.machine_rank,
        args.mixed_precision,
        args.debug,
    )
