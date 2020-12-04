# coding=utf-8
__author__ = 'Dimitrios Karkalousos'

import numpy as np
import pathlib

from typing import Callable, Dict, Optional, Any, List

from direct.data.h5_data import H5SliceData
from direct.types import PathOrString

import logging
from direct.utils import str_to_class, remove_keys
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class TECFIDERADataset(H5SliceData):
    def __init__(
        self,
        root: pathlib.Path,
        transform: Optional[Callable] = None,
        regex_filter: Optional[str] = None,
        filenames_filter: Optional[List[PathOrString]] = None,
        pass_mask: bool = False,
        pass_h5s: Optional[Dict] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            root=root,
            filenames_filter=filenames_filter,
            regex_filter=regex_filter,
            metadata=None,
            extra_keys=None,
            text_description=kwargs.get("text_description", None),
            pass_h5s=pass_h5s,
            pass_dictionaries=kwargs.get("pass_dictionaries", None),
        )
        print('Hey TECFIDERA!!!!!!!!!!!!!!!!')
        # Sampling rate in the slice-encode direction
        self.transform = transform
        self.pass_mask: bool = pass_mask

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = super().__getitem__(idx)

        if self.transform:
            sample = self.transform(sample)

        return sample


def build_dataset(
    name,
    root: pathlib.Path,
    filenames_filter: Optional[List[PathOrString]] = None,
    sensitivity_maps: Optional[pathlib.Path] = None,
    transforms: Optional[Any] = None,
    text_description: Optional[str] = None,
    kspace_context: Optional[int] = 0,
    **kwargs,
) -> Dataset:
    """

    Parameters
    ----------
    dataset_name : str
        Name of dataset class (without `Dataset`) in direct.data.datasets.
    root : pathlib.Path
        Root path to the data for the dataset class.
    filenames_filter : List
        List of filenames to include in the dataset, should be the same as the ones that can be derived from a glob
        on the root. If set, will skip searching for files in the root.
    sensitivity_maps : pathlib.Path
        Path to sensitivity maps.
    transforms : object
        Transformation object
    text_description : str
        Description of dataset, can be used for logging.
    kspace_context : int
        If set, output will be of shape -kspace_context:kspace_context.

    Returns
    -------
    Dataset
    """

    # TODO: Maybe only **kwargs are fine.
    logger.info(f"Building dataset for: {name}.")
    dataset_class: Callable = str_to_class("projects.tecfidera.data.dataset", name + "Dataset")
    logger.debug(f"Dataset class: {dataset_class}.")
    dataset = dataset_class(
        root=root,
        filenames_filter=filenames_filter,
        transform=transforms,
        sensitivity_maps=sensitivity_maps,
        text_description=text_description,
        kspace_context=kspace_context,
        **kwargs,
    )

    logger.debug(f"Dataset:\n{dataset}")

    return dataset


def build_dataset_from_input(
    transforms,
    dataset_config,
    initial_images,
    initial_kspaces,
    filenames_filter,
    data_root,
    pass_dictionaries,
):
    pass_h5s = None
    if initial_images is not None and initial_kspaces is not None:
        raise ValueError(
            f"initial_images and initial_kspaces are mutually exclusive. "
            f"Got {initial_images} and {initial_kspaces}."
        )

    if initial_images:
        pass_h5s = {"initial_image": (dataset_config.input_image_key, initial_images)}

    if initial_kspaces:
        pass_h5s = {
            "initial_kspace": (dataset_config.input_kspace_key, initial_kspaces)
        }

    dataset = build_dataset(
        root=data_root,
        filenames_filter=filenames_filter,
        transforms=transforms,
        pass_h5s=pass_h5s,
        pass_dictionaries=pass_dictionaries,
        **remove_keys(dataset_config, ["transforms", "lists"]),
    )
    return dataset


