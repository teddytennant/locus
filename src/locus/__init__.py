import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from locus.heads import head_unembedding, rms_norm, select_heads, select_heads_matrix
from locus.somp import evr, property_dictionary, somp
from locus.steer import (
    apply_steering,
    difference_of_means,
    project_onto_atoms,
    strength_scale,
)

__all__ = [
    "apply_steering",
    "difference_of_means",
    "evr",
    "head_unembedding",
    "project_onto_atoms",
    "property_dictionary",
    "rms_norm",
    "select_heads",
    "select_heads_matrix",
    "somp",
    "strength_scale",
]
