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

"""The raw feature vector circuit."""

from typing import Optional
import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.circuit import QuantumRegister, QuantumCircuit, ParameterVector, Instruction
from qiskit.circuit.library import BlueprintCircuit


class RawFeatureVector(BlueprintCircuit):
    """The raw feature vector circuit.

    This circuit acts as parameterized initialization for statevectors with ``feature_dimension``
    dimensions, thus with ``log2(feature_dimension)`` qubits. The circuit contains a
    placeholder instruction that can only be synthesized/defined when all parameters are bound.

    In ML, this circuit can be used to load the training data into qubit amplitudes. It does not
    apply an kernel transformation. (Therefore, it is a "raw" feature vector.)

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
        #      │  PARAMETERIZEDINITIALIZE(x[0],x[1],x[2],x[3]) │
        # q_1: ┤1                                              ├
        #      └───────────────────────────────────────────────┘

        print(circuit.ordered_parameters)
        # prints: [Parameter(p[0]), Parameter(p[1]), Parameter(p[2]), Parameter(p[3])]

        import numpy as np
        state = np.array([1, 0, 0, 1]) / np.sqrt(2)
        bound = circuit.assign_parameters(state)
        print(bound.draw())
        # prints:
        #      ┌───────────────────────────────────────────────┐
        # q_0: ┤0                                              ├
        #      │  PARAMETERIZEDINITIALIZE(0.70711,0,0,0.70711) │
        # q_1: ┤1                                              ├
        #      └───────────────────────────────────────────────┘

    """

    def __init__(self, feature_dimension: Optional[int]) -> None:
        """
        Args:
            feature_dimension: The feature dimension and number of qubits.

        """
        super().__init__()

        self._num_qubits = None
        self._ordered_parameters = ParameterVector("x")

        if feature_dimension:
            self.feature_dimension = feature_dimension

    def _build(self):
        super()._build()

        placeholder = ParameterizedInitialize(self._ordered_parameters[:])
        self.append(placeholder, self.qubits)

    def _unsorted_parameters(self):
        if self.data is None:
            self._build()
        return super()._unsorted_parameters()

    def _check_configuration(self, raise_on_failure=True):
        if isinstance(self._ordered_parameters, ParameterVector):
            self._ordered_parameters.resize(self.feature_dimension)
        elif len(self._ordered_parameters) != self.feature_dimension:
            if raise_on_failure:
                raise ValueError("Mismatching number of parameters and feature dimension.")
            return False
        return True

    @property
    def num_qubits(self) -> int:
        """Returns the number of qubits in this circuit.

        Returns:
            The number of qubits.
        """
        return self._num_qubits if self._num_qubits is not None else 0

    @num_qubits.setter
    def num_qubits(self, num_qubits: int) -> None:
        """Set the number of qubits for the n-local circuit.

        Args:
            The new number of qubits.
        """
        if self._num_qubits != num_qubits:
            # invalidate the circuit
            self._invalidate()
            self._num_qubits = num_qubits
            self.add_register(QuantumRegister(self.num_qubits, "q"))

    @property
    def feature_dimension(self) -> int:
        """Return the feature dimension.

        Returns:
            The feature dimension, which is ``2 ** num_qubits``.
        """
        return 2 ** self.num_qubits

    @feature_dimension.setter
    def feature_dimension(self, feature_dimension: int) -> None:
        """Set the feature dimension.

        Args:
            feature_dimension: The new feature dimension. Must be a power of 2.

        Raises:
            ValueError: If ``feature_dimension`` is not a power of 2.
        """
        num_qubits = np.log2(feature_dimension)
        if int(num_qubits) != num_qubits:
            raise ValueError("feature_dimension must be a power of 2!")

        if self._num_qubits is None or num_qubits != self._num_qubits:
            self._invalidate()
            self.num_qubits = int(num_qubits)

    def _invalidate(self):
        super()._invalidate()
        self._num_qubits = None


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
            if len(param.parameters) == 0:
                cleaned_params.append(complex(param))
            else:
                raise QiskitError("Cannot define a ParameterizedInitialize with unbound parameters")

        # normalize
        normalized = np.array(cleaned_params) / np.linalg.norm(cleaned_params)

        circuit = QuantumCircuit(self.num_qubits)
        circuit.initialize(normalized, range(self.num_qubits))
        self.definition = circuit
