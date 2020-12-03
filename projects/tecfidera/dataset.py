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

