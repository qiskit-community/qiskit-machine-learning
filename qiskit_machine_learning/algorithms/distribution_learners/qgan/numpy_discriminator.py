# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Discriminator

The neural network is based on a neural network introduced in:
https://towardsdatascience.com/lets-code-a-neural-network-in-plain-numpy-ae7e74410795
"""

from typing import Dict, Any
import os
import numpy as np
from qiskit.utils import algorithm_globals
from qiskit.algorithms.optimizers import ADAM
from .discriminative_network import DiscriminativeNetwork


# pylint: disable=invalid-name


class DiscriminatorNet:
    """
    Discriminator

    The neural network is based on a neural network introduced in:
    https://towardsdatascience.com/lets-code-a-neural-network-in-plain-numpy-ae7e74410795
    """

    def __init__(self, n_features=1, n_out=1):
        """
        Initialize the discriminator network.

        Args:
            n_features (int): Dimension of input data samples.
            n_out (int): output dimension
        """
        self.architecture = [
            {"input_dim": n_features, "output_dim": 50, "activation": "leaky_relu"},
            {"input_dim": 50, "output_dim": 20, "activation": "leaky_relu"},
            {"input_dim": 20, "output_dim": n_out, "activation": "sigmoid"},
        ]

        self.parameters = []
        self.memory = {}

        for _, layer in enumerate(self.architecture):
            activ_function_curr = layer["activation"]
            layer_input_size = layer["input_dim"]
            layer_output_size = layer["output_dim"]
            params_layer = algorithm_globals.random.random(layer_output_size * layer_input_size)
            if activ_function_curr == "leaky_relu":
                params_layer = (params_layer * 2 - np.ones(np.shape(params_layer))) * 0.7
            elif activ_function_curr == "sigmoid":
                params_layer = (params_layer * 2 - np.ones(np.shape(params_layer))) * 0.2
            else:
                params_layer = params_layer * 2 - np.ones(np.shape(params_layer))
            self.parameters = np.append(self.parameters, params_layer)
            self.parameters.flatten()

    def forward(self, x):
        """
        Forward propagation.

        Args:
            x (numpy.ndarray): , Discriminator input, i.e. data sample.

        Returns:
            list: Discriminator output, i.e. data label.
        """

        def sigmoid(z):
            sig = 1 / (1 + np.exp(-z))
            return sig

        def leaky_relu(z, slope=0.2):
            return np.maximum(np.zeros(np.shape(z)), z) + slope * np.minimum(
                np.zeros(np.shape(z)), z
            )

        def single_layer_forward_propagation(x_old, w_new, activation="leaky_relu"):
            z_curr = np.dot(w_new, x_old)

            if activation == "leaky_relu":
                activation_func = leaky_relu
            elif activation == "sigmoid":
                activation_func = sigmoid
            else:
                raise Exception("Non-supported activation function")

            return activation_func(z_curr), z_curr

        x_new = x
        pointer = 0
        for idx, layer in enumerate(self.architecture):
            layer_idx = idx + 1
            activ_function_curr = layer["activation"]
            layer_input_size = layer["input_dim"]
            layer_output_size = layer["output_dim"]
            if idx == 0:
                x_old = np.reshape(x_new, (layer_input_size, len(x_new)))
            else:
                x_old = x_new
            pointer_next = pointer + (layer_output_size * layer_input_size)
            w_curr = self.parameters[pointer:pointer_next]
            w_curr = np.reshape(w_curr, (layer_output_size, layer_input_size))
            pointer = pointer_next
            x_new, z_curr = single_layer_forward_propagation(x_old, w_curr, activ_function_curr)

            self.memory["a" + str(idx)] = x_old
            self.memory["z" + str(layer_idx)] = z_curr

        return x_new

    def backward(self, x, y, weights=None):
        """
        Backward propagation.

        Args:
           x (numpy.ndarray): sample label (equivalent to discriminator output)
           y (numpy.ndarray): array, target label
           weights (numpy.ndarray): customized scaling for each sample (optional)

        Returns:
            tuple(numpy.ndarray, numpy.ndarray): parameter gradients
        """

        def sigmoid_backward(da, z):
            sig = 1 / (1 + np.exp(-z))
            return da * sig * (1 - sig)

        def leaky_relu_backward(da, z, slope=0.2):
            dz = np.array(da, copy=True)
            for i, line in enumerate(z):
                for j, element in enumerate(line):
                    if element < 0:
                        dz[i, j] = dz[i, j] * slope
            return dz

        def single_layer_backward_propagation(
            da_curr, w_curr, z_curr, a_prev, activation="leaky_relu"
        ):
            # m = a_prev.shape[1]
            if activation == "leaky_relu":
                backward_activation_func = leaky_relu_backward
            elif activation == "sigmoid":
                backward_activation_func = sigmoid_backward
            else:
                raise Exception("Non-supported activation function")

            dz_curr = backward_activation_func(da_curr, z_curr)
            dw_curr = np.dot(dz_curr, a_prev.T)
            da_prev = np.dot(w_curr.T, dz_curr)

            return da_prev, dw_curr

        grads_values = np.array([])
        m = y.shape[1]
        y = y.reshape(np.shape(x))
        if weights is not None:
            da_prev = -np.multiply(
                weights,
                np.divide(y, np.maximum(np.ones(np.shape(x)) * 1e-4, x))
                - np.divide(1 - y, np.maximum(np.ones(np.shape(x)) * 1e-4, 1 - x)),
            )
        else:
            da_prev = (
                -(
                    np.divide(y, np.maximum(np.ones(np.shape(x)) * 1e-4, x))
                    - np.divide(1 - y, np.maximum(np.ones(np.shape(x)) * 1e-4, 1 - x))
                )
                / m
            )

        pointer = 0

        for layer_idx_prev, layer in reversed(list(enumerate(self.architecture))):
            layer_idx_curr = layer_idx_prev + 1
            activ_function_curr = layer["activation"]

            da_curr = da_prev

            a_prev = self.memory["a" + str(layer_idx_prev)]
            z_curr = self.memory["z" + str(layer_idx_curr)]

            layer_input_size = layer["input_dim"]
            layer_output_size = layer["output_dim"]
            pointer_prev = pointer - (layer_output_size * layer_input_size)
            if pointer == 0:
                w_curr = self.parameters[pointer_prev:]
            else:
                w_curr = self.parameters[pointer_prev:pointer]
            w_curr = np.reshape(w_curr, (layer_output_size, layer_input_size))
            pointer = pointer_prev

            da_prev, dw_curr = single_layer_backward_propagation(
                da_curr, np.array(w_curr), z_curr, a_prev, activ_function_curr
            )

            grads_values = np.append([dw_curr], grads_values)

        return grads_values


class NumPyDiscriminator(DiscriminativeNetwork):
    """
    Discriminator based on NumPy
    """

    def __init__(self, n_features: int = 1, n_out: int = 1) -> None:
        """
        Args:
            n_features: Dimension of input data vector.
            n_out: Dimension of the discriminator's output vector.
        """
        super().__init__()
        self._n_features = n_features
        self._n_out = n_out
        self._discriminator = DiscriminatorNet(self._n_features, self._n_out)
        self._optimizer = ADAM(
            maxiter=1,
            tol=1e-6,
            lr=1e-3,
            beta_1=0.7,
            beta_2=0.99,
            noise_factor=1e-4,
            eps=1e-6,
            amsgrad=True,
        )

        self._ret = {}  # type: Dict[str, Any]

    def set_seed(self, seed):
        """
        Set seed.
        Args:
            seed (int): seed
        """
        algorithm_globals.random_seed = seed

    def save_model(self, snapshot_dir):
        """
        Save discriminator model

        Args:
            snapshot_dir (str): directory path for saving the model
        """
        # save self._discriminator.params_values
        np.save(
            os.path.join(snapshot_dir, "np_discriminator_architecture.csv"),
            self._discriminator.architecture,
        )
        np.save(
            os.path.join(snapshot_dir, "np_discriminator_memory.csv"),
            self._discriminator.memory,
        )
        np.save(
            os.path.join(snapshot_dir, "np_discriminator_params.csv"),
            self._discriminator.parameters,
        )
        self._optimizer.save_params(snapshot_dir)

    def load_model(self, load_dir):
        """
        Load discriminator model

        Args:
            load_dir (str): file with stored pytorch discriminator model to be loaded
        """
        self._discriminator.architecture = np.load(
            os.path.join(load_dir, "np_discriminator_architecture.csv")
        )
        self._discriminator.memory = np.load(os.path.join(load_dir, "np_discriminator_memory.csv"))
        self._discriminator.parameters = np.load(
            os.path.join(load_dir, "np_discriminator_params.csv")
        )
        self._optimizer.load_params(load_dir)

    @property
    def discriminator_net(self):
        """
        Get discriminator

        Returns:
            DiscriminatorNet: discriminator object
        """
        return self._discriminator

    @discriminator_net.setter
    def discriminator_net(self, net):
        self._discriminator = net

    def get_label(self, x, detach=False):  # pylint: disable=arguments-differ,unused-argument
        """
        Get data sample labels, i.e. true or fake.

        Args:
            x (numpy.ndarray): Discriminator input, i.e. data sample.
            detach (bool): depreciated for numpy network

        Returns:
            numpy.ndarray: Discriminator output, i.e. data label
        """

        return self._discriminator.forward(x)

    def loss(self, x, y, weights=None):
        """
        Loss function

        Args:
            x (numpy.ndarray): sample label (equivalent to discriminator output)
            y (numpy.ndarray): target label
            weights(numpy.ndarray): customized scaling for each sample (optional)

        Returns:
            float: loss function
        """
        if weights is not None:
            # Use weights as scaling factors for the samples and compute the sum
            return (-1) * np.dot(
                np.multiply(y, np.log(np.maximum(np.ones(np.shape(x)) * 1e-4, x)))
                + np.multiply(
                    np.ones(np.shape(y)) - y,
                    np.log(np.maximum(np.ones(np.shape(x)) * 1e-4, np.ones(np.shape(x)) - x)),
                ),
                weights,
            )
        else:
            # Compute the mean
            return (-1) * np.mean(
                np.multiply(y, np.log(np.maximum(np.ones(np.shape(x)) * 1e-4, x)))
                + np.multiply(
                    np.ones(np.shape(y)) - y,
                    np.log(np.maximum(np.ones(np.shape(x)) * 1e-4, np.ones(np.shape(x)) - x)),
                )
            )

    def _get_objective_function(self, data, weights):
        """
        Get the objective function

        Args:
            data (tuple): training and generated data
            weights (numpy.ndarray): weights corresponding to training resp. generated data

        Returns:
            objective_function: objective function for the optimization
        """
        real_batch = data[0]
        real_prob = weights[0]
        generated_batch = data[1]
        generated_prob = weights[1]

        def objective_function(params):
            self._discriminator.parameters = params
            # Train on Real Data
            prediction_real = self.get_label(real_batch)
            loss_real = self.loss(prediction_real, np.ones(np.shape(prediction_real)), real_prob)
            prediction_fake = self.get_label(generated_batch)
            loss_fake = self.loss(
                prediction_fake, np.zeros(np.shape(prediction_fake)), generated_prob
            )
            return 0.5 * (loss_real[0] + loss_fake[0])

        return objective_function

    def _get_gradient_function(self, data, weights):
        """
        Get the gradient function

        Args:
            data (tuple): training and generated data
            weights (numpy.ndarray): weights corresponding to training resp. generated data

        Returns:
            gradient_function: Gradient function for the optimization
        """
        real_batch = data[0]
        real_prob = weights[0]
        generated_batch = data[1]
        generated_prob = weights[1]

        def gradient_function(params):
            self._discriminator.parameters = params
            prediction_real = self.get_label(real_batch)
            grad_real = self._discriminator.backward(
                prediction_real, np.ones(np.shape(prediction_real)), real_prob
            )
            prediction_generated = self.get_label(generated_batch)
            grad_generated = self._discriminator.backward(
                prediction_generated,
                np.zeros(np.shape(prediction_generated)),
                generated_prob,
            )
            return np.add(grad_real, grad_generated)

        return gradient_function

    def train(
        self, data, weights, penalty=False, quantum_instance=None, shots=None
    ) -> Dict[str, Any]:
        """
        Perform one training step w.r.t to the discriminator's parameters

        Args:
            data (tuple(numpy.ndarray, numpy.ndarray)):
                real_batch: array, Training data batch.
                generated_batch: array, Generated data batch.
            weights (tuple):real problem, generated problem
            penalty (bool): Depreciated for classical networks.
            quantum_instance (QuantumInstance): Depreciated for classical networks.
            shots (int): Number of shots for hardware or qasm execution.
                Ignored for classical networks.

        Returns:
            dict: with Discriminator loss and updated parameters.
        """

        # Train on Generated Data
        # Force single optimization iteration
        self._optimizer._maxiter = 1
        self._optimizer._t = 0
        objective = self._get_objective_function(data, weights)
        gradient = self._get_gradient_function(data, weights)
        self._discriminator.parameters, loss, _ = self._optimizer.optimize(
            num_vars=len(self._discriminator.parameters),
            objective_function=objective,
            initial_point=np.array(self._discriminator.parameters),
            gradient_function=gradient,
        )

        self._ret["loss"] = loss
        self._ret["params"] = self._discriminator.parameters

        return self._ret
