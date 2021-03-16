import abc
import numpy as np
from math import log2

class Loss:
    """
    Abstract base class for Loss.
    """
    def __call__(self, predict, target):
        self.predict = predict
        self.target = target

    @abc.abstractmethod
    def evaluate(self, predict, target):
        raise NotImplementedError

    @abc.abstractmethod
    def gradient(self, predict, target):
        raise NotImplementedError

class L2Loss(Loss):

    def __call__(self, predict, target):
        super().__call__(predict, target)
        self.predict = predict
        self.target = target
        return self.evaluate(self.predict, self.target)

    def evaluate(self, predict, target):
        predict = np.array(predict)
        target = np.array(target)

        if len(predict.shape) <= 1:
            return np.linalg.norm(predict - target)**2
        elif len(predict.shape) > 1:
            return np.linalg.norm(predict - target, axis=len(predict.shape)-1)**2
        else:
            raise QiskitMachineLearningError('TODO')  # TODO

    def gradient(self, predict, target):
        predict = np.array(predict)
        target = Snp.array(target)
        return 2*(predict - target)

class L1Loss(Loss): # L1 loss is not differentiable at 0

    def __call__(self, predict, target):
        super().__call__(predict, target)
        self.predict = predict
        self.target = target
        return self.evaluate(self.predict, self.target)

    def evaluate(self, predict, target):
        predict = np.array(predict)
        target = np.array(target)

        return np.sum(np.abs(predict - target))


#################################################
#################################################
### TBD
#################################################
#################################################

class L2Loss_Probability(Loss):

    def __init__(self, predict, target): #predict and target are both probabilities
        super().__init__(predict, target)
        self.joint_keys = set(predict.keys())
        self.joint_keys.update(target.keys())

    def evaluate(self):
        val = 0.0
        for k in self.joint_keys:
            val += (self.predict.get(k, 0) - self.target.get(k,0))**2
        return val

    def gradient(self):
        val = {}
        for k in self.joint_keys:
            val[k] = 2*(self.predict.get(k, 0) - self.target.get(k,0))
        return val

class CrossEntropyLoss(Loss):

    def __init__(self, predict, target): #predict and target are both probabilities
        super().__init__(predict, target)
        self.predict=np.array(predict)
        self.target=np.array(target)

    def evaluate(self):
        return -sum([predict[i]*log2(target[i]) for i in range(len(predict))])

    #gradient depends on how to handling softmax

class KLDivergence(Loss):

    def __init__(self, predict, target): #predict and target are both probabilities
        super().__init__(predict, target)
        self.predict=np.array(predict)
        self.target=np.array(target)

    def evaluate(self):
        return sum(predict[i] * log2(predict[i]/target[i]) for i in range(len(predict)))