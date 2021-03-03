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

"""Quantum Support Vector Classifier"""

from typing import Union
from sklearn.svm import SVC

from qiskit import QuantumCircuit
from qiskit.providers import Backend, BaseBackend
from qiskit.utils import QuantumInstance
from qiskit_machine_learning.kernels.quantum_kernel import QuantumKernel


class QSVC(SVC):
    r"""Quantum Support Vector Classifier.

        **Example**
        .. code-block::
            qsvc = QSVC(feature_map=map, quantum_instance=backend)
            qsvc.fit(sample_train,label_train)
            qsvc.predict(sample_test)
    """

    def __init__(self,
                 feature_map: QuantumCircuit,
                 quantum_instance:
                 Union[QuantumInstance, BaseBackend, Backend],
                 *args, **kwargs) -> None:

        self._qkernel = QuantumKernel(feature_map=feature_map,
                                      quantum_instance=quantum_instance)

        super().__init__(kernel=self._qkernel.evaluate, *args, **kwargs)
