# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
from typing import Dict

from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes

from qiskit_machine_learning import QiskitMachineLearningError


def retrieve_arguments_if_none(ansatz: QuantumCircuit, feature_map: QuantumCircuit,
                               num_qubits: int):
    num_qubits_dic = {}
    if ansatz:
        num_qubits_dic["ansatz"] = ansatz.num_qubits
    if num_qubits:
        num_qubits_dic["num_qubits"] = num_qubits
    if feature_map:
        num_qubits_dic["feature_map"] = feature_map.num_qubits

    num_qubits = _validate_and_get_num_qubits(num_qubits_dic)

    return ansatz if ansatz else RealAmplitudes(
        num_qubits), feature_map if feature_map else ZZFeatureMap(num_qubits), num_qubits


def _validate_and_get_num_qubits(num_qubits_dic: Dict[str, int]):
    if not num_qubits_dic:
        raise QiskitMachineLearningError('Need at least one of num_qubits, feature_map, or ansatz!')
    num_qubits_origin_name = next(iter(num_qubits_dic))
    num_qubits = num_qubits_dic[num_qubits_origin_name]
    for curr_num_qubits_origin_name in num_qubits_dic.keys():
        if num_qubits != num_qubits_dic[curr_num_qubits_origin_name]:
            raise QiskitMachineLearningError(
                f'Incompatible {num_qubits_origin_name} and {curr_num_qubits_origin_name}!')
    return num_qubits
