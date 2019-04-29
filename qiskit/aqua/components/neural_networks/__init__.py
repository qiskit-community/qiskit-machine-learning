# -*- coding: utf-8 -*-

# Copyright 2018 IBM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

from qiskit.aqua.components.neural_networks.discriminative_networks.discriminative_network import DiscriminativeNetwork
from qiskit.aqua.components.neural_networks.generative_networks.generative_network import GenerativeNetwork
from qiskit.aqua.components.neural_networks.discriminative_networks.classical_discriminator import ClassicalDiscriminator
from qiskit.aqua.components.neural_networks.generative_networks.quantum_generator import QuantumGenerator


__all__ = ['DiscriminativeNetwork',
           'GenerativeNetwork',
           'ClassicalDiscriminator',
           'QuantumGenerator']
