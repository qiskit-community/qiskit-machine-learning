# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" QGAN Package """

from .discriminative_network import DiscriminativeNetwork
from .generative_network import GenerativeNetwork
from .numpy_discriminator import NumPyDiscriminator
from .pytorch_discriminator import PyTorchDiscriminator
from .quantum_generator import QuantumGenerator
from .qgan import QGAN

__all__ = [
    "DiscriminativeNetwork",
    "GenerativeNetwork",
    "NumPyDiscriminator",
    "PyTorchDiscriminator",
    "QuantumGenerator",
    "QGAN",
]
