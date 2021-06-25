# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests of loss functions."""
from test import QiskitMachineLearningTestCase, requires_extra_library

from qiskit.exceptions import MissingOptionalLibraryError

import numpy as np
from ddt import ddt, data

from qiskit_machine_learning.utils.loss_functions import L1Loss, L2Loss


@ddt
class TestLossFunctions(QiskitMachineLearningTestCase):
    """Tests of loss functions."""

    @data(
        # input shape, loss shape
        ((5,), (5,), "absolute_error"),
        ((5, 2), (5,), "absolute_error"),
        ((5,), (5,), "squared_error"),
        ((5, 2), (5,), "squared_error"),
    )
    @requires_extra_library
    def test_l1_l2_loss(self, config):
        """Tests L1 and L2 loss functions on different input types."""
        try:
            import torch
            from torch.nn import L1Loss as TL1Loss
            from torch.nn import MSELoss as TL2Loss
        except ImportError as ex:
            raise MissingOptionalLibraryError(
                libname="Pytorch",
                name="TestLossFunctions",
                pip_install="pip install 'qiskit-machine-learning[torch]'",
            ) from ex

        input_shape, loss_shape, loss_function = config
        qpredict = np.random.rand(*input_shape) if input_shape else np.random.rand()
        qtarget = np.random.rand(*input_shape) if input_shape else np.random.rand()
        tpredict = torch.tensor(qpredict, requires_grad=True)  # pylint:disable=not-callable
        ttarget = torch.tensor(qtarget, requires_grad=True)  # pylint:disable=not-callable

        # quantum loss
        if loss_function == "absolute_error":
            q_loss_fun = L1Loss()
            # pytorch loss
            t_loss_fun = TL1Loss(reduction="none")
        elif loss_function == "squared_error":
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
