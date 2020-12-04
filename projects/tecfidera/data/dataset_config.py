# coding=utf-8
# Copyright (c) DIRECT Contributors
from dataclasses import dataclass, field
from typing import Tuple, Optional, List

from direct.config.defaults import BaseConfig
from direct.common.subsample_config import MaskingConfig

from omegaconf import MISSING

from direct.data.datasets_config import DatasetConfig

@dataclass
class TECFIDERAConfig(DatasetConfig):
    pass_mask: bool = False
