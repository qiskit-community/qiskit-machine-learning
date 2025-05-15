# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Sbplx (Subplex) optimizer."""

from .nloptimizer import NLoptOptimizer, NLoptOptimizerType


class SBPLX(NLoptOptimizer):
    """
    Subplex optimizer.

    "Subplex (a variant of Nelder-Mead that uses Nelder-Mead on a sequence of subspaces)
    is claimed to be much more efficient and robust than the original Nelder-Mead,
    while retaining the latter's facility with discontinuous objectives,
    and in my experience these claims seem to be true in many cases.
    (However, I'm not aware of any proof that Subplex is globally convergent,
    and perhaps it may fail for some objectives like Nelder-Mead; YMMV.)"
    Description by Steven G. Johnson, author of NLopt library.

    NLopt local optimizer, derivative-free.
    For further detail, please refer to
    https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/#sbplx-based-on-subplex
    """

    def get_nlopt_optimizer(self) -> NLoptOptimizerType:
        """Return NLopt optimizer type."""
        return NLoptOptimizerType.LN_SBPLX
