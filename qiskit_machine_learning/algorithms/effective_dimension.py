from math import pi
from scipy.special import logsumexp
import numpy as np
import time

# how should this import be handled??
from ..neural_networks import OpflowQNN, NeuralNetwork
from typing import Optional, Union, List

class EffectiveDimension:

    """This class computes the global effective dimension for Qiskit NeuralNetworks.
    """

    def __init__(self,
            qnn: NeuralNetwork,
            num_thetas: Optional[int] = 1,
            num_inputs: Optional[int] = 1,
            thetas: Optional[Union[List, np.array]] = None,
            inputs: Optional[Union[List, np.array]] = None
            ) -> None:
        """
        Args:
            qnn: A Qiskit NeuralNetwork, with a specific number of weights (qnn_num_weights) and input size (qnn_input_size)
            num_thetas: Number of parameters, if user chooses to randomly sample from a uniform distribution.
            num_inputs:  Number of inputs, if user chooses to randomly sample from a normal distribution.
            thetas: An array of neural network weights, of shape (num_thetas, qnn_num_weights).
            inputs: An array of inputs to the neural network, of shape (num_inputs, qnn_input_size).
        """
        np.random.seed(0)
        # Store inputs
        self.model = qnn
        self.num_thetas = num_thetas
        self.num_inputs = num_inputs

        # Define Fisher Matrix size (d)
        self.d = qnn.num_weights

        # Check for user-defined inputs and thetas
        if thetas is not None:
            self.params = thetas
            self.num_thetas = len(self.params)
        else:
            # if thetas are not provided, sample randomly from uniform distribution
            self.params = np.random.uniform(0, 1, size=(self.num_thetas, self.d))

        if inputs is not None:
            self.x = inputs
            self.num_inputs = len(self.x)
        else:
            # if inputs are not provided, sample randomly from normal distribution
            self.x = np.random.normal(0, 1, size=(self.num_inputs, self.model.num_inputs))


    def get_fisher(self,
            gradients: Optional[Union[List, np.array]], # dp_thetas
            model_output: Optional[Union[List, np.array]] # p_thetas
            ) -> np.array:
        """
        Computes the empirical Fisher Information Matrix, of shape (num_inputs * num_thetas, d, d),
        by calculating the average jacobian for every set of gradients and model output given:

        1/K * (sum_k(dp_thetas_k/p_thetas_k)), where K = num_labels

        Args:
            gradients: A numpy array with dp_thetas
            model_output: A numpy array with p_thetas
        Returns:
            fishers: A numpy matrix of shape (num_inputs * num_thetas, d, d) with the Fisher information
        """

        # add dimension for broadcasting reasons
        model_output = np.expand_dims(model_output, axis=2)
        # get dp_thetas/p_thetas for every label
        # multiply by sqrt(p_thetas) so that the outer product cross term is correct
        gradvectors = np.sqrt(model_output) * gradients / model_output
        # compute sum of matrices obtained from outer product of gradvectors
        fishers = np.einsum('ijk,lji->ikl', gradvectors, gradvectors.T)

        return fishers

    def get_fhat(self) -> [np.array, np.array]:
        """
        Computes the normalized Fisher Information Matrix.

        Returns:
             f_hat: The normalized FIM, of size (num_inputs, d, d)
             fisher_trace: The trace of the FIM (before normalizing)
        """
        grads, output = self.do_montecarlo()
        fishers = self.get_fisher(gradients=grads, model_output=output)
        # compute the trace with all fishers
        fisher_trace = np.trace(np.average(fishers, axis=0))
        # average the fishers over the num_inputs to get the empirical fishers
        fisher = np.average(np.reshape(fishers, (self.num_thetas, self.num_inputs, self.d, self.d)), axis=1)
        # calculate f_hats for all the empirical fishers
        f_hat = self.d * fisher / fisher_trace
        return f_hat, fisher_trace

    def do_montecarlo(self) -> [np.array, np.array]:
        """
        Computes the qnn output and gradient Monte Carlo sampling for a set of inputs and parameters (thetas).

        Returns:
             grads: QNN gradient vector, of shape (num_inputs * num_thetas, outputsize, d)
             outputs: QNN goutput vector, of shape (num_inputs * num_thetas, outputsize)
        """
        grads = np.zeros((self.num_inputs * self.num_thetas, self.model.output_shape[0], self.d))
        outputs = np.zeros((self.num_inputs * self.num_thetas, self.model.output_shape[0]))

        for (i, p) in enumerate(self.params):
            back_pass = np.array(self.model.backward(input_data=self.x, weights=p)[1])
            fwd_pass = np.array(self.model.forward(input_data=self.x, weights=p))  # get model output

            grads[self.num_inputs * i:self.num_inputs * (i + 1)] = back_pass
            outputs[self.num_inputs * i:self.num_inputs * (i + 1)] = fwd_pass

        # post-process in the case of OpflowQNN output to match CircuitQNN output format
        if isinstance(self.model, OpflowQNN):
            grads = np.concatenate([grads/ 2, -1 * grads / 2], 1)
            outputs = np.concatenate([(outputs + 1) / 2, (1 - outputs) / 2], 1)

        return grads, outputs

    def eff_dim(self,
                n: List):
        """
        Compute the effective dimension.
        Args:
            n: list of ranges for number of data
        Returns:
             effective_dim: list of effective dimensions for each data range in n
             time: [DEBUG] time estimation for eff dim algorithm
        """
        t0 = time.time()
        f_hat, trace = self.get_fhat()
        effective_dim = []
        for ns in n:
            Fhat = f_hat * ns / (2 * pi * np.log(ns))
            one_plus_F = np.eye(self.d) + Fhat
            det = np.linalg.slogdet(one_plus_F)[1]  # log det because of overflow
            r = det / 2  # divide by 2 because of sqrt
            effective_dim.append(2 * (logsumexp(r) - np.log(self.num_thetas)) / np.log(ns / (2 * pi * np.log(ns))))
        t1 = time.time()
        return effective_dim, t1-t0

class LocalEffectiveDimension(EffectiveDimension):
    """
    Computes the LOCAL effective dimension for a parametrized model.
    """
    def __init__(self,
            qnn: NeuralNetwork,
            num_inputs: Optional[int] = 1,
            thetas: Optional[Union[List, np.array]] = None,
            inputs: Optional[Union[List, np.array]] = None
            ) -> None:
        """
        Args:
            qnn: A Qiskit NeuralNetwork, with a specific number of weights (qnn_num_weights) and input size (qnn_input_size)
            num_inputs:  Number of inputs, if user chooses to randomly sample from a normal distribution.
            thetas: An array of neural network weights, of shape (1, qnn_num_weights).
            inputs: An array of inputs to the neural network, of shape (num_inputs, qnn_input_size).

        Raises:
            ValueError: If len(thetas) > 1
        """
        np.random.seed(0)
        self.model = qnn
        self.d = qnn.num_weights
        self.num_thetas = 1
        self.num_inputs = num_inputs

        # check that parameters are provided
        thetas = np.array(thetas)
        inputs = np.array(inputs)

        if thetas is not None:
            if len(thetas.shape) > 1:
                if thetas.shape[0] > 1:
                    raise ValueError("The local effective dimension algorithm uses only 1 set of parameters.")
                else:
                    self.params = thetas
            else:
                self.params = np.reshape(thetas, (1,-1))
            self.num_thetas = len(self.params)
        else:
            self.params = np.random.uniform(0, 1, size=(self.num_thetas, self.d))

        if inputs is not None:
            self.x = inputs
            self.num_inputs = len(self.x)

        elif num_inputs is not None:
            self.x = np.random.normal(0, 1, size=(self.num_inputs, self.model.num_inputs))

