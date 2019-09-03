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

""" Generative Quantum and Classical Neural Networks. """

from abc import abstractmethod

from qiskit.aqua import Pluggable


class GenerativeNetwork(Pluggable):
    """Base class for generative Quantum and Classical Neural Networks.

        This method should initialize the module and its configuration, and
        use an exception if a component of the module is
        available.
    """
    @abstractmethod
    def __init__(self):
        super().__init__()
        self._num_parameters = 0
        self._num_qubits = 0
        self._bounds = list()
        pass

    @classmethod
    def init_params(cls, params):
        """ init params """
        generative_params = params.get(Pluggable.SECTION_KEY_GENERATIVE_NETWORK)
        args = {k: v for k, v in generative_params.items() if k != 'name'}

        return cls(**args)

    @classmethod
    @abstractmethod
    def get_section_key_name(cls):
        """ get section key name """
        pass

    @abstractmethod
    def set_seed(self, seed):
        """
        Set seed.
        Args:
            seed (int): seed
        Raises:
            NotImplementedError: not implemented
        """
        raise NotImplementedError()

    @abstractmethod
    def get_output(self, quantum_instance, qc_state_in, params, shots):
        """ Apply quantum/classical neural network to given input and get the respective output
         Args:
            quantum_instance (QuantumInstance): Quantum Instance, used to run the generator circuit.
            qc_state_in (QuantumCircuit): corresponding to the input state
            params (numpy.ndarray): parameters which should be used to run the generator,
                    if None use self._params
            shots (int): if not None use a number of shots that is different from the number
                    set in quantum_instance

        Returns:
            Neural network output
        Raises:
            NotImplementedError: not implemented
        """
        raise NotImplementedError()

    @abstractmethod
    def loss(self):
        """Loss function used for optimization
        """
        raise NotImplementedError()

    @abstractmethod
    def train(self, quantum_instance=None, shots=None):
        """
        Perform one training step w.r.t to the generator's parameters
        Args:
            quantum_instance (QuantumInstance): Quantum Instance, used to run
                                the generator circuit.
                                Depreciated for classical network
            shots (int): Number of shots for hardware or qasm execution.
                                Depreciated for classical network

        Returns:
            dict: generator loss and updated parameters.
        Raises:
            NotImplementedError: not implemented
        """
        raise NotImplementedError()
