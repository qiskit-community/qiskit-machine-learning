# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2019, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
===============================================================
Qiskit Machine Learning module (:mod:`qiskit_machine_learning`)
===============================================================

.. currentmodule:: qiskit_machine_learning

Qiskit Machine Learning is an ML framework that comes with essential tools like quantum kernels and
quantum neural networks. These tools are the building blocks of the framework and can be used for
building and training classification, regression, and other models.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

    QiskitMachineLearningError

Submodules
==========

.. autosummary::
   :toctree:

   algorithms
   circuit.library
   connectors
   datasets
   gradients
   kernels
   neural_networks
   optimizers
   state_fidelities
   utils

"""

from .version import __version__
from .exceptions import QiskitMachineLearningError, AlgorithmError

__all__ = ["__version__", "QiskitMachineLearningError", "AlgorithmError"]
