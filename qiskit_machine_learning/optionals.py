# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Additional optional constants.
"""

from qiskit.utils import LazyImportTester


HAS_TORCH = LazyImportTester(
    {
        "torch": ("cat", "einsum", "is_tensor", "nn", "optim", "sparse_coo_tensor", "Tensor"),
        "torch.autograd": ("Function",),
        "torch.autograd.variable": ("Variable",),
        "torch.nn": (
            "L1Loss",
            "Linear",
            "Module",
            "MSELoss",
            "Parameter",
        ),
        "torch.nn.functional": (),
        "torch.optim": (
            "Adam",
            "SGD",
        ),
        "torch.utils.data": ("Dataset",),
    },
    name="PyTorch",
    install="pip install 'qiskit-machine-learning[torch]'",
)

HAS_SPARSE = LazyImportTester(
    {
        "sparse": ("SparseArray", "COO", "DOK"),
    },
    name="sparse",
    install="pip install 'qiskit-machine-learning[sparse]'",
)
