from math import pi
from scipy.special import logsumexp
import numpy as np
import time

# how should this import be handled??
from ..neural_networks import OpflowQNN, NeuralNetwork
from typing import Optional, Union, List

class EffectiveDimension:

    """This class computes the effective dimension for Qiskit NeuralNetworks.
    """

    def __init__(
            self,
            qnn: NeuralNetwork,
            num_thetas: Optional[int] = 1,
            num_inputs: Optional[int] = 1,
            thetas: Optional[Union[List, np.array]] = None,
            inputs: Optional[Union[List, np.array]] = None
            ) -> None:
        """
        Args:
            qnn: A Qiskit NeuralNetwork
            num_thetas:
            num_inputs:
            thetas:
            inputs:
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


    def get_fisher(
            self,
            gradients: Optional[Union[List, np.array]], # dp_theta
            model_output: Optional[Union[List, np.array]] # p_theta
            )-> None:
        """
        Computes the empirical Fisher Information Matrix, of shape (num_inputs*num_thetas, d, d),
        by calculating the average jacobian for every set of gradients and model output given.

        1/K(sum_k(sum_i dp_theta_i/sum_i p_theta_i)) for i in index for label k
        :param gradients: ndarray, dp_theta
        :param model_output: ndarray, p_theta
        :return: ndarray, average jacobian for every set of gradients and model output given
        """

        model_output = np.expand_dims(model_output, axis=2)
        gradients = np.sqrt(model_output) * gradients / model_output # shape: (num_inputs*num_thetas, outputsize, d)
        fishers = np.einsum('ijk,lji->ikl', gradients, gradients.T)

        return fishers

    def get_fhat(self):
        """
        :return: ndarray, f_hat values of size (num_inputs, d, d)
        """
        grads, output = self.do_montecarlo()
        fishers = self.get_fisher(gradients=grads, model_output=output)
        fisher_trace = np.trace(np.average(fishers, axis=0))  # compute the trace with all fishers
        # average the fishers over the num_inputs to get the empirical fishers
        fisher = np.average(np.reshape(fishers, (self.num_thetas, self.num_inputs, self.d, self.d)), axis=1)
        f_hat = self.d * fisher / fisher_trace  # calculate f_hats for all the empirical fishers
        return f_hat, fisher_trace

    def do_montecarlo(self):

        grads = np.zeros((self.num_inputs * self.num_thetas, self.model.output_shape[0], self.d))
        output = np.zeros((self.num_inputs * self.num_thetas, self.model.output_shape[0]))

        for (i, p) in enumerate(self.params):
            back_pass = np.array(self.model.backward(input_data=self.x, weights=p)[1])
            fwd_pass = np.array(self.model.forward(input_data=self.x, weights=p))  # get model output

            grads[self.num_inputs * i:self.num_inputs * (i + 1)] = back_pass
            output[self.num_inputs * i:self.num_inputs * (i + 1)] = fwd_pass

        # post-process in the case of OpflowQNN output to match CircuitQNN output format
        if isinstance(self.model, OpflowQNN):
            grads = np.concatenate([grads/ 2, -1 * grads / 2], 1)
            output = np.concatenate([(output + 1) / 2, (1 - output) / 2], 1)

        return grads, output

    def eff_dim(self, n):
        """
        Compute the effective dimension.
        :param f_hat: ndarray
        :param n: list, used to represent number of data samples available as per the effective dimension calc
        :return: list, effective dimension for each n
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
    def __init__(self, qnn, num_inputs=1, thetas=None, inputs=None):
        """
        Computes the effective dimension for a parameterised model.
        :param model: class instance
        :param num_thetas: int, number of parameter sets to include
        :param num_inputs: int, number of input samples to include
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
                    print("NOOO, ERROR!!!")
                else:
                    self.params = thetas
            else:
                self.params = np.reshape(thetas, (1,-1))
            self.num_thetas = len(self.params)
        else:
            self.params = np.random.uniform(0, 1, size=(self.num_thetas, self.d))


        print(self.params.shape)

        if inputs is not None:
            self.x = inputs
            self.num_inputs = len(self.x)

        elif num_inputs is not None:
            self.x = np.random.normal(0, 1, size=(self.num_inputs, self.model.num_inputs))

