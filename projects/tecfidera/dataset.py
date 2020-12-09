# coding=utf-8
__author__ = 'Dimitrios Karkalousos'

import numpy as np
import pathlib

from typing import Callable, Dict, Optional, Any, List

from direct.data.h5_data import H5SliceData
from direct.types import PathOrString

import logging

logger = logging.getLogger(__name__)


class TECFIDERADataset(H5SliceData):
    def __init__(
        self,
        root: pathlib.Path,
        transform: Optional[Callable] = None,
        regex_filter: Optional[str] = None,
        filenames_filter: Optional[List[PathOrString]] = None,
        sensitivity_maps: Optional[pathlib.Path] = None,
        pass_mask: bool = False,
        pass_h5s: Optional[Dict] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            root=root,
            filenames_filter=filenames_filter,
            sensitivity_maps=sensitivity_maps,
            regex_filter=regex_filter,
            metadata=None,
            extra_keys=None,
            text_description=kwargs.get("text_description", None),
            pass_h5s=pass_h5s,
            pass_dictionaries=kwargs.get("pass_dictionaries", None),
        )
        # Sampling rate in the slice-encode direction
        self.transform = transform
        self.pass_mask: bool = pass_mask
        self.sensitivity_maps = sensitivity_maps

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = super().__getitem__(idx)

        sample["kspace"] = np.ascontiguousarray(sample["kspace"].transpose(2, 0, 1))
        imspace = np.fft.ifftn(sample["kspace"], axes=(1, 2))
        imspace = np.clip(np.where(imspace == 0, np.array([0.0], dtype=imspace.dtype), (imspace / np.max(imspace))), 0.0, 1.0)
        sample["kspace"] = np.fft.fftn(imspace, axes=(1, 2))

        if self.sensitivity_maps is not None:
            sample["sensitivity_map"] = np.ascontiguousarray(sample["sensitivity_map"].transpose(2, 0, 1))
            sample["sensitivity_map"] = np.clip(np.where(sample["sensitivity_map"] == 0, np.array([0.0], dtype=sample["sensitivity_map"].dtype),
                                                         (sample["sensitivity_map"] / np.max(sample["sensitivity_map"]))), 0.0, 1.0)

        if self.transform:
            sample = self.transform(sample)

        print('imspace', np.max(np.abs(imspace)), np.min(np.abs(imspace)))
        print('sensitivity_map', np.max(np.abs(sample["sensitivity_map"])), np.min(np.abs(sample["sensitivity_map"])))

        return sample
