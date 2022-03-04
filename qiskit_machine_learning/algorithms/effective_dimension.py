from math import pi
from scipy.special import logsumexp
import numpy as np
import time

# how should this import be handled??
from ..neural_networks import OpflowQNN

class EffectiveDimension:

    """This class computes the effective dimension for Qiskit QuantumNeuralNetworks.
    """

    def __init__(self, qnn, num_thetas=1, num_inputs=1, thetas=None, inputs=None):

        self.model = qnn
        self.d = qnn.num_weights
        self.num_thetas = num_thetas
        self.num_inputs = num_inputs

        # check that parameters are provided


        if thetas is not None:
            self.params = thetas
            self.num_thetas = len(self.params)

        elif num_thetas is not None:
            self.params = np.random.uniform(0, 1, size=(self.num_thetas, self.d))

        if inputs is not None:
            self.x = inputs
            self.num_inputs = len(self.x)

        elif num_inputs is not None:
            self.x = np.random.normal(0, 1, size=(self.num_inputs, self.model.num_inputs))


    def get_fisher(self, gradients, model_output):
        """
        Computes the jacobian as we defined it and then returns the average jacobian:
        1/K(sum_k(sum_i dp_theta_i/sum_i p_theta_i)) for i in index for label k
        :param gradients: ndarray, dp_theta
        :param model_output: ndarray, p_theta
        :return: ndarray, average jacobian for every set of gradients and model output given
        """
        gradvectors = []
        outputsize = model_output.shape[1]

        for k in range(len(gradients)):
            jacobian = []
            m_output = model_output[k]  # p_theta size: (1, outputsize)
            new_gradients = np.transpose(gradients, (0, 2, 1))
            jacobians_ = new_gradients[k, :, :]  # dp_theta size: (d, 2**num_qubits)
            for idx, y in enumerate(m_output):
                denominator = m_output[idx]  # get correct model output sum(p_theta) for indices
                for j in range(self.d):
                    row = jacobians_[j, :]
                    # for each row of a particular dp_theta, do sum(dp_theta)/sum(p_theta) for indices
                    # multiply by sqrt(sum(p_theta)) so that the outer product cross term is correct
                    jacobian.append(np.sqrt(denominator) * (row[idx] / denominator))
            # append gradient vectors for every output for all data points
            gradvectors.append(np.reshape(jacobian, (outputsize, self.d)))
        # full gradient vector
        gradients = np.reshape(gradvectors, (len(gradients), outputsize, self.d))

        fishers = np.zeros((len(gradients), self.d, self.d))
        for i in range(len(gradients)):
            grads = gradients[i]  # size = (outputsize, d)
            temp_sum = np.zeros((outputsize, self.d, self.d))
            for j in range(outputsize):
                temp_sum[j] += np.array(np.outer(grads[j], np.transpose(grads[j])))
            fishers[i] += np.sum(temp_sum, axis=0)  # sum the two matrices to get fisher estimate
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

