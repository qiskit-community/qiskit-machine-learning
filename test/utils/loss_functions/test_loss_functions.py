# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests of loss functions."""

from test import QiskitMachineLearningTestCase

import numpy as np
import torch
from ddt import ddt, data
from torch.nn import L1Loss as TL1Loss
from torch.nn import MSELoss as TL2Loss

from qiskit_machine_learning.utils.loss_functions import L1Loss, L2Loss, CrossEntropyLoss


@ddt
class TestLossFunctions(QiskitMachineLearningTestCase):
    """Tests of loss functions."""

    @data(
        # input shape, loss shape
        (None, (), "l1"),
        ((5,), (5,), "l1"),
        ((5, 2), (5,), "l1"),
        (None, (), "l2"),
        ((5,), (5,), "l2"),
        ((5, 2), (5,), "l2"),
    )
    def test_l1_l2_loss(self, config):
        """Tests L1 and L2 loss functions on different input types."""
        input_shape, loss_shape, loss_function = config
        qpredict = np.random.rand(*input_shape) if input_shape else np.random.rand()
        qtarget = np.random.rand(*input_shape) if input_shape else np.random.rand()
        tpredict = torch.tensor(qpredict, requires_grad=True)  # pylint:disable=not-callable
        ttarget = torch.tensor(qtarget, requires_grad=True)  # pylint:disable=not-callable

        # quantum loss
        if loss_function == "l1":
            q_loss_fun = L1Loss()
            # pytorch loss
            t_loss_fun = TL1Loss(reduction="none")
        elif loss_function == "l2":
            q_loss_fun = L2Loss()
            t_loss_fun = TL2Loss(reduction="none")
        else:
            raise ValueError(f"Unsupported loss function: {loss_function}")

        # q values
        qloss = q_loss_fun(qpredict, qtarget)
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

    # def test_cross_entropy_loss(self):
    #     input_shape, loss_shape = (5, 2), (5, 1)
    #     qpredict = np.random.rand(*input_shape) if input_shape else np.random.rand()
    #     qtarget = np.random.rand(*input_shape) if input_shape else np.random.rand()
    #     tpredict = torch.tensor(qpredict, requires_grad=True)
    #     ttarget = torch.tensor(qtarget, requires_grad=True)
    #
    #     # quantum loss
    #     loss = CrossEntropyLoss()
    #     qloss = loss(qpredict, qtarget)
    #
    #     np.testing.assert_equal(qloss.shape, loss_shape)
    #     qloss_sum = np.sum(loss(qpredict, qtarget))
    #     qgrad = loss.gradient(qpredict, qtarget)
    #
    #     # pytorch loss
    #     loss = TCrossEntropyLoss(reduction="none")
    #     tloss_sum = loss(tpredict, ttarget).sum()
    #     tloss_sum.backward()
    #     tgrad = tpredict.grad.detach().numpy()
    #
    #     np.testing.assert_almost_equal(qloss_sum, tloss_sum.detach().numpy())
    #     np.testing.assert_almost_equal(qgrad, tgrad)
