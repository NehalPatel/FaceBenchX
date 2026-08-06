"""Public evaluation dataset adapters.

Milestone M3 provides adapters for all eight v1 public datasets.
Recognition models remain out of scope until later milestones.
"""

from __future__ import annotations

from facebench.datasets.agedb import AgeDBDataset
from facebench.datasets.ar_face import ARFaceDataset
from facebench.datasets.base import BaseDataset
from facebench.datasets.category_registry import (
    DATASET_PREP_DOCS,
    CategoryRegistry,
    get_default_registry,
    get_prep_doc,
)
from facebench.datasets.cfp_fp import CFPFPDataset
from facebench.datasets.chokepoint import ChokePointDataset
from facebench.datasets.cplfw import CPLFWDataset
from facebench.datasets.factory import (
    DatasetFactory,
    DatasetFactoryError,
    create_dataset,
)
from facebench.datasets.integrity import IntegrityValidator
from facebench.datasets.lfw import LFWDataset
from facebench.datasets.tinyface import TinyFaceDataset
from facebench.datasets.types import (
    DatasetIndex,
    IdentityPair,
    Sample,
    ValidationResult,
)
from facebench.datasets.ytf import YTFDataset

__all__ = [
    "DATASET_PREP_DOCS",
    "AgeDBDataset",
    "ARFaceDataset",
    "BaseDataset",
    "CFPFPDataset",
    "CategoryRegistry",
    "ChokePointDataset",
    "CPLFWDataset",
    "DatasetFactory",
    "DatasetFactoryError",
    "DatasetIndex",
    "IdentityPair",
    "IntegrityValidator",
    "LFWDataset",
    "Sample",
    "TinyFaceDataset",
    "ValidationResult",
    "YTFDataset",
    "create_dataset",
    "get_default_registry",
    "get_prep_doc",
]
