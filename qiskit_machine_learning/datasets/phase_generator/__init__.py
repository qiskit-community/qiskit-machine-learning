"""
Datasets Package Initialization

This module imports and exposes the main dataset generation functionality.
"""

from qiskit_machine_learning.datasets.phase_generator.phase_of_matter import (
    generate_heisenberg_dataset,
    generate_haldane_dataset,
    generate_annni_dataset,
    generate_cluster_dataset
)

__all__ = [
    'generate_heisenberg_dataset',
    'generate_haldane_dataset',
    'generate_annni_dataset',
    'generate_cluster_dataset'
]