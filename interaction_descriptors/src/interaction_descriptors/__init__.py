"""Frozen interaction-oriented descriptors; no source fitting at inference."""

from .composition import CompositionInputError, audit_and_align, common_component_schema
from .extension import DescriptorResult, boundary_extension
from .frozen import FrozenBank, FrozenModel
from .observed_support import inspect_source_observations, audit_with_source_observations

__version__ = "0.1.0"
__all__ = ["CompositionInputError", "audit_and_align", "common_component_schema",
           "DescriptorResult", "boundary_extension", "FrozenBank", "FrozenModel",
           "inspect_source_observations", "audit_with_source_observations"]
