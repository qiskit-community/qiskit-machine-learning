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

from typing import Set, List, Optional
import numpy as np
from qiskit.circuit import QuantumRegister, ParameterVector, ParameterExpression, Gate
from qiskit.circuit.library import BlueprintCircuit


class RawFeatureVector(BlueprintCircuit):
    """The raw feature vector circuit.

    This circuit acts as parameterized initialization for statevectors with ``feature_dimension``
    dimensions, thus with ``log2(feature_dimension)`` qubits. As long as there are free parameters,
    this circuit holds a placeholder instruction and can not be decomposed.
    Once all parameters are bound, the placeholder is replaced by a state initialization and can
    be unrolled.

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
        #      ┌──────┐
        # q_0: ┤0     ├
        #      │  Raw │
        # q_1: ┤1     ├
        #      └──────┘

        print(circuit.ordered_parameters)
        # prints: [Parameter(p[0]), Parameter(p[1]), Parameter(p[2]), Parameter(p[3])]

        import numpy as np
        state = np.array([1, 0, 0, 1]) / np.sqrt(2)
        bound = circuit.assign_parameters(state)
        print(bound.draw())
        # prints:
        #      ┌──────────────────────────────────┐
        # q_0: ┤0                                 ├
        #      │  initialize(0.70711,0,0,0.70711) │
        # q_1: ┤1                                 ├
        #      └──────────────────────────────────┘

        """

    def __init__(self, feature_dimension: Optional[int]) -> None:
        """
        Args:
            feature_dimension: The feature dimension and number of qubits.

        """
        super().__init__()

        self._num_qubits = None  # type: int
        self._parameters = None  # type: List[ParameterExpression]

        if feature_dimension:
            self.feature_dimension = feature_dimension

    def _build(self):
        super()._build()

        # if the parameters are fully specified, use the initialize instruction
        if len(self.parameters) == 0:
            self.initialize(self._parameters, self.qubits)  # pylint: disable=no-member

        # otherwise get a gate that acts as placeholder
        else:
            placeholder = Gate('Raw', self.num_qubits, self._parameters[:], label='Raw')
            self.append(placeholder, self.qubits)

    def _check_configuration(self, raise_on_failure=True):
        pass

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
            self._parameters = list(ParameterVector('p', length=self.feature_dimension))
            self.add_register(QuantumRegister(self.num_qubits, 'q'))

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
            raise ValueError('feature_dimension must be a power of 2!')

        if self._num_qubits is None or num_qubits != self._num_qubits:
            self._invalidate()
            self.num_qubits = int(num_qubits)

    def _invalidate(self):
        super()._invalidate()
        self._parameters = None
        self._num_qubits = None

    @property
    def parameters(self) -> Set[ParameterExpression]:
        """Return the free parameters in the RawFeatureVector.

        Returns:
            A set of the free parameters.
        """
        return set(self.ordered_parameters)

    @property
    def ordered_parameters(self) -> List[ParameterExpression]:
        """Return the free parameters in the RawFeatureVector.

        Returns:
            A list of the free parameters.
        """
        return list(param for param in self._parameters if isinstance(param, ParameterExpression))

    def bind_parameters(self, values):  # pylint: disable=arguments-differ
        """Bind parameters."""
        if not isinstance(values, dict):
            values = dict(zip(self.ordered_parameters, values))
        return super().bind_parameters(values)

    def assign_parameters(self, parameters, inplace=False):  # pylint: disable=arguments-differ
        """Call the initialize instruction."""
        if not isinstance(parameters, dict):
            parameters = dict(zip(self.ordered_parameters, parameters))

        if inplace:
            dest = self
        else:
            dest = RawFeatureVector(2 ** self.num_qubits)
            dest._build()
            dest._parameters = self._parameters.copy()

        # update the parameter list
        for i, param in enumerate(dest._parameters):
            if param in parameters.keys():
                dest._parameters[i] = parameters[param]

        # if fully bound call the initialize instruction
        if len(dest.parameters) == 0:
            dest._data = []  # wipe the current data
            parameters = dest._parameters / np.linalg.norm(dest._parameters)
            dest.initialize(parameters, dest.qubits)  # pylint: disable=no-member

        # else update the placeholder
        else:
            dest.data[0][0].params = dest._parameters

        return None if inplace else dest
