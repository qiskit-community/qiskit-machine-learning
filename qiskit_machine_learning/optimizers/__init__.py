# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2018, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

r"""
Optimizers (:mod:`qiskit_machine_learning.optimizers`)
======================================================

Contains a variety of classical optimizers designed for
Qiskit Algorithm's quantum variational algorithms.
Logically, these optimizers can be divided into two categories:

`Local Optimizers`_
  Given an optimization problem, a **local optimizer** is a function
  that attempts to find an optimal value within the neighboring set of a candidate solution.

`Global Optimizers`_
  Given an optimization problem, a **global optimizer** is a function
  that attempts to find an optimal value among all possible solutions.

.. currentmodule:: qiskit_machine_learning.optimizers

Optimizer base classes
----------------------

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   OptimizerResult
   Optimizer
   Minimizer

Steppable optimization
----------------------

.. autosummary::
   :toctree: ../stubs/

   optimizer_utils

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   SteppableOptimizer
   AskData
   TellData
   OptimizerState


Local optimizers
----------------

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   ADAM
   AQGD
   CG
   COBYLA
   L_BFGS_B
   GSLS
   GradientDescent
   GradientDescentState
   NELDER_MEAD
   NFT
   P_BFGS
   POWELL
   SLSQP
   SPSA
   QNSPSA
   TNC
   SciPyOptimizer
   UMDA

The optimizers from
`scikit-quant <https://scikit-quant.readthedocs.io/en/latest/>`_ are not included in the
Qiskit Machine Learning library.
To continue using them, please import them from Qiskit Algorithms. Be aware that and a
deprecation of the methods `snobfit`, `imfil` and `bobyqa` the was considered:
https://github.com/qiskit-community/qiskit-algorithms/issues/84.


Global optimizers
-----------------
The global optimizers here all use `NLOpt <https://nlopt.readthedocs.io/en/latest/>`_ for their
core function and can only be used if the optional dependent ``NLOpt`` package is installed.
To install the ``NLOpt`` dependent package you can use ``pip install nlopt``.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   CRS
   DIRECT_L
   DIRECT_L_RAND
   ESCH
   ISRES

"""

from .adam_amsgrad import ADAM
from .aqgd import AQGD
from .cg import CG
from .cobyla import COBYLA
from .gsls import GSLS
from .gradient_descent import GradientDescent, GradientDescentState
from .l_bfgs_b import L_BFGS_B
from .nelder_mead import NELDER_MEAD
from .nft import NFT
from .nlopts.crs import CRS
from .nlopts.direct_l import DIRECT_L
from .nlopts.direct_l_rand import DIRECT_L_RAND
from .nlopts.esch import ESCH
from .nlopts.isres import ISRES
from .steppable_optimizer import SteppableOptimizer, AskData, TellData, OptimizerState
from .optimizer import Minimizer, Optimizer, OptimizerResult, OptimizerSupportLevel
from .p_bfgs import P_BFGS
from .powell import POWELL
from .qnspsa import QNSPSA
from .scipy_optimizer import SciPyOptimizer
from .slsqp import SLSQP
from .spsa import SPSA
from .tnc import TNC
from .umda import UMDA

__all__ = [
    "Optimizer",
    "OptimizerSupportLevel",
    "SteppableOptimizer",
    "AskData",
    "TellData",
    "OptimizerState",
    "OptimizerResult",
    "Minimizer",
    "ADAM",
    "AQGD",
    "CG",
    "COBYLA",
    "GSLS",
    "GradientDescent",
    "GradientDescentState",
    "L_BFGS_B",
    "NELDER_MEAD",
    "NFT",
    "P_BFGS",
    "POWELL",
    "SciPyOptimizer",
    "SLSQP",
    "SPSA",
    "QNSPSA",
    "TNC",
    "CRS",
    "DIRECT_L",
    "DIRECT_L_RAND",
    "ESCH",
    "ISRES",
    "UMDA",
]
