# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


from .generative_network import GenerativeNetwork
from .quantum_generator import QuantumGenerator
from .discriminative_network import DiscriminativeNetwork

__all__ = [
    'DiscriminativeNetwork',
    'GenerativeNetwork',
    'QuantumGenerator'
]

try:
    from .classical_discriminator import ClassicalDiscriminator
    __all__ += ['ClassicalDiscriminator']
except Exception:
    pass

