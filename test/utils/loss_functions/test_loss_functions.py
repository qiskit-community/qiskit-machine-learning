# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2021, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests of loss functions."""

import unittest
from test import QiskitMachineLearningTestCase

import numpy as np
from ddt import ddt, data

import qiskit_machine_learning.optionals as _optionals
from qiskit_machine_learning.utils.loss_functions import CrossEntropyLoss, L1Loss, L2Loss


@ddt
class TestLossFunctions(QiskitMachineLearningTestCase):
    """Tests of loss functions."""

    @data(
        # predict, target, expected_loss
        (np.array([0.5, 0.5]), np.array([1.0, 0.0]), 1.0),
        (np.array([1.0, 0.0]), np.array([1.0, 0.0]), 0.0),
    )
    def test_cross_entropy_loss(self, config):
        """
        Tests that CrossEntropyLoss returns the correct value, and no `nan` when one of the
        probabilities is zero.
        """
        predict, target, expected_loss = config
        loss_fn = CrossEntropyLoss()
        loss = loss_fn.evaluate(predict, target)
        assert np.allclose(loss, expected_loss, atol=1e-5)

    @data(
        # input shape, loss shape
        ((5,), (5,), "absolute_error"),
        ((5, 2), (5,), "absolute_error"),
        ((5,), (5,), "squared_error"),
        ((5, 2), (5,), "squared_error"),
    )
    @unittest.skipIf(not _optionals.HAS_TORCH, "PyTorch not available.")
    def test_l1_l2_loss(self, config):
        """Tests L1 and L2 loss functions on different input types."""
        import torch
        from torch.nn import L1Loss as TL1Loss
        from torch.nn import MSELoss as TL2Loss

        input_shape, loss_shape, loss_function = config
        qpredict = np.random.rand(*input_shape) if input_shape else np.random.rand()
        qtarget = np.random.rand(*input_shape) if input_shape else np.random.rand()
        tpredict = torch.tensor(qpredict, requires_grad=True)  # pylint:disable=not-callable
        ttarget = torch.tensor(qtarget, requires_grad=True)  # pylint:disable=not-callable

        # quantum loss
        if loss_function == "absolute_error":
            q_loss_fun = L1Loss()
            # PyTorch loss
            t_loss_fun = TL1Loss(reduction="none")
        elif loss_function == "squared_error":
            q_loss_fun = L2Loss()
            t_loss_fun = TL2Loss(reduction="none")
        else:
            raise ValueError(f"Unsupported loss function: {loss_function}")

        # q values
        # Note: the numpy as array method was not here before. That loss call returns
        # a numpy array but the pylint 3.0.0 that was just released ends up with
        # a recursion error when accessing shape from it afterward that this circumvents.
        qloss = np.asarray(q_loss_fun(qpredict, qtarget))
        np.testing.assert_equal(qloss.shape, loss_shape)
        qloss_sum = np.sum(q_loss_fun(qpredict, qtarget))
        qgrad = q_loss_fun.gradient(qpredict, qtarget)

        # torch values
        tloss_sum = t_loss_fun(tpredict, ttarget).sum()
        tloss_sum.backward()
        tgrad = tpredict.grad.detach().numpy()

        # comparison
        np.testing.assert_almost_equal(qloss_sum, tloss_sum.detach().numpy())
        np.testing.assert_almost_equal(qgrad, tgrad)


if __name__ == "__main__":
    unittest.main()
