import numpy as np
from qiskit_machine_learning import QiskitMachineLearningError
from qiskit_machine_learning.neural_networks import TwoLayerQNN, CircuitQNN, SamplingNeuralNetwork
from .loss import L2Loss

class NNRegressor():
    """ Quantum neural network regressor
    """

    def __init__(self, qnn, loss, optimizer, warm_start=False, callback=None):  # TODO: callback
        """
        Args:
        """

        self._qnn = qnn
        self._loss = loss
        self._optimizer = optimizer

        self._value_objective = True
        if isinstance(self._qnn, SamplingNeuralNetwork):
            if self._qnn.return_samples:
                pass  # TODO
            else:
                self._value_objective = False

        self._warm_start = warm_start
        self._fit_result = None


    def fit(self, X, y):

        if self._value_objective:

            def objective(w):
                val = 0.0
                for x, y_target in zip(X, y):
                    # TODO: enable batching / proper handling of batches
                    val += self._loss(self._qnn.forward(x, w), np.array([y_target]))
                return val

            def objective_grad(w):
                val = 0.0
                for x, y_target in zip(X, y):
                    # TODO: allow setting which gradients to evaluate (input/weights)
                    _, weights_grad = self._qnn.backward(x, w)
                    # TODO: can we store the forward result and reuse it?
                    val += self._loss.gradient(self._qnn.forward(x, w)[0], y_target) * weights_grad
                return val

        else:

            def objective(w):
                val = 0.0
                for x, y_target in zip(X, y):
                    probs = self._qnn.forward(x, w)
                    for y_predict, p in probs.items():
                        val += p * self._loss(y_predict, y_target)
                return val

            def objective_grad(w):
                grad = np.zeros(self._qnn.num_weights)
                for x, y_target in zip(X, y):
                    _, weight_prob_grad = self._qnn.backward(x, w)
                    for i in range(self._qnn.num_weights):
                        for y_predict, p_grad in weight_prob_grad[i].items():
                            grad[i] += p_grad * self._loss(y_predict, y_target)
                return grad

        if self._warm_start and not self._fit_result is None:
            initial_point = self._fit_result[0]
        else:
            initial_point = np.random.rand(self._qnn.num_weights)

        self._fit_result = self._optimizer.optimize(self._qnn.num_weights, objective, objective_grad,
                                                    initial_point=initial_point)
        return self

    def predict(self, X):

        if self._fit_result is None:
            raise QiskitMachineLearningError('Model needs to be fit to some training data first!')

        # TODO: proper handling of batching
        result = np.zeros(len(X))
        for i, x in enumerate(X):
            # TODO: handle sampling case too
            result[i] = self._qnn.forward(x, self._fit_result[0])
        return result

    def score(self, X, y):
        if self._fit_result is None:
            raise QiskitMachineLearningError('Model needs to be fit to some training data first!')
        return np.sum(self.predict(X) - y) / len(y)


## do we need save & load model?