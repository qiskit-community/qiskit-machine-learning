# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2020, 2025
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The raw feature vector circuit."""

import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.circuit import (
    QuantumCircuit,
    ParameterVector,
    Instruction,
    ParameterExpression,
)


def raw_feature_vector(feature_dimension: int) -> QuantumCircuit:
    """The raw feature vector circuit.

    This circuit acts as parameterized initialization for statevectors with ``feature_dimension``
    dimensions, thus with ``log2(feature_dimension)`` qubits. The circuit contains a
    placeholder instruction that can only be synthesized/defined when all parameters are bound.

    In ML, this circuit can be used to load the training data into qubit amplitudes. It does not
    apply an kernel transformation (therefore, it is a "raw" feature vector).

    Since initialization is implemented via a ``QuantumCircuit.initialize()`` call, this circuit
    can't be used with gradient based optimizers, one can see a warning that gradients can't be
    computed.

    Examples:

    .. code-block::

        from qiskit_machine_learning.circuit.library import RawFeatureVector
        circuit = RawFeatureVector(4)
        print(circuit.num_qubits)
        # prints: 2

        print(circuit.draw(output='text'))
        # prints:
        #      ┌───────────────────────────────────────────────┐
        # q_0: ┤0                                              ├
        #      │  Parameterizedinitialize(x[0],x[1],x[2],x[3]) │
        # q_1: ┤1                                              ├
        #      └───────────────────────────────────────────────┘

        print(circuit.ordered_parameters)
        # prints: [Parameter(p[0]), Parameter(p[1]), Parameter(p[2]), Parameter(p[3])]

        import numpy as np
        state = np.array([1, 0, 0, 1]) / np.sqrt(2)
        bound = circuit.assign_parameters(state)
        print(bound)
        # prints:
        #      ┌───────────────────────────────────────────────┐
        # q_0: ┤0                                              ├
        #      │  Parameterizedinitialize(0.70711,0,0,0.70711) │
        # q_1: ┤1                                              ├
        #      └───────────────────────────────────────────────┘

    Args:
        feature_dimension: The feature dimension from which the number of
                           qubits is inferred as ``n_qubits = log2(feature_dim)``

    Raises:
        ValueError: If ``feature_dimension`` is not a power of 2.

    Returns:
        The raw feature
    """
    num_qubits = np.log2(feature_dimension)
    if int(num_qubits) != num_qubits:
        raise ValueError("feature_dimension must be a power of 2!")

    ordered_parameters = ParameterVector("x", feature_dimension)
    placeholder = ParameterizedInitialize(ordered_parameters[:])
    qc = QuantumCircuit(num_qubits)
    qc.append(placeholder, qc.qubits)
    return qc


class ParameterizedInitialize(Instruction):
    """A normalized parameterized initialize instruction."""

    def __init__(self, amplitudes):
        num_qubits = np.log2(len(amplitudes))
        if int(num_qubits) != num_qubits:
            raise ValueError("feature_dimension must be a power of 2!")

        super().__init__("ParameterizedInitialize", int(num_qubits), 0, amplitudes)

    def _define(self):
        # cast ParameterExpressions that are fully bound to numbers
        cleaned_params = []
        for param in self.params:
            if not isinstance(param, ParameterExpression) or len(param.parameters) == 0:
                cleaned_params.append(complex(param))
            else:
                raise QiskitError("Cannot define a ParameterizedInitialize with unbound parameters")

        # normalize
        norm = np.linalg.norm(cleaned_params)
        normalized = cleaned_params if np.isclose(norm, 1) else cleaned_params / norm

        circuit = QuantumCircuit(self.num_qubits)
        circuit.initialize(normalized, range(self.num_qubits))
        self.definition = circuit
