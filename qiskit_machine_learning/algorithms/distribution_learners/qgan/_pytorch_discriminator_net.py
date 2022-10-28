# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2022.
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
"""

import qiskit_machine_learning.optionals as _optionals
from ....deprecation import deprecate_function

if _optionals.HAS_TORCH:
    import torch

# torch 1.6.0 fixed a mypy error about not applying contravariance rules
# to inputs by defining forward as a value, rather than a function.  See also
# https://github.com/python/mypy/issues/8795
# The fix introduced an error on Module class about '_forward_unimplemented'
# not being implemented.
# The pylint disable=abstract-method fixes it.


@_optionals.HAS_TORCH.require_in_instance
class DiscriminatorNet(torch.nn.Module):  # pylint: disable=abstract-method
    """
    Discriminator
    """

    @deprecate_function(
        "0.5.0",
        additional_msg="with no direct replacement for it. "
        "Instead, please refer to the new QGAN tutorial",
        stack_level=3,
    )
    def __init__(self, n_features: int = 1, n_out: int = 1) -> None:
        """
        Initialize the discriminator network.

        Args:
            n_features: Dimension of input data samples.
            n_out: n out
        """

        super().__init__()
        # pylint: disable=import-error
        from torch import nn

        self.n_features = n_features

        self.hidden0 = nn.Sequential(
            nn.Linear(self.n_features, 512),
            nn.LeakyReLU(0.2),
        )

        self.hidden1 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
        )
        self.out = nn.Sequential(nn.Linear(256, n_out), nn.Sigmoid())

    def forward(self, x):  # pylint: disable=arguments-differ
        """

        Args:
            x (torch.Tensor): Discriminator input, i.e. data sample.

        Returns:
            torch.Tensor: Discriminator output, i.e. data label.
        """
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.out(x)

        return x
