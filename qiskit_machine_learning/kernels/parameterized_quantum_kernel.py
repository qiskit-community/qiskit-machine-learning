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

"""Quantum Kernel Algorithm"""

from typing import Optional, Union, List, Dict, Callable

import time
import numpy as np
import warnings
from functools import partial

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZZFeatureMap
from qiskit.providers import Backend, BaseBackend
from qiskit.utils import QuantumInstance
from qiskit.algorithms.variational_algorithm import VariationalResult
from qiskit.algorithms.optimizers import SPSA
from ..exceptions import QiskitMachineLearningError
from qiskit_machine_learning.kernels import QuantumKernel

class ParamQuantumKernel(QuantumKernel):
    r"""Quantum Kernel.

    The general task of machine learning is to find and study patterns in data. For many
    algorithms, the datapoints are better understood in a higher dimensional feature space,
    through the use of a kernel function:

    .. math::

        K(x, y) = \langle f(x), f(y)\rangle.

    Here K is the kernel function, x, y are n dimensional inputs. f is a map from n-dimension
    to m-dimension space. :math:`\langle x, y \rangle` denotes the dot product.
    Usually m is much larger than n.

    The quantum kernel algorithm calculates a kernel matrix, given datapoints x and y and feature
    map f, all of n dimension. This kernel matrix can then be used in classical machine learning
    algorithms such as support vector classification, spectral clustering or ridge regression.
    """

    def __init__(self,
                feature_map: Optional[QuantumCircuit] = None,
                input_params: Optional[Union[ParameterVector, List]] = None,
                free_params: Optional[Union[ParameterVector, List]] = None,
                enforce_psd: bool = True,
                batch_size: int = 900,
                quantum_instance: Optional[Union[QuantumInstance, BaseBackend, Backend]] = None,
    ) -> None:
        r"""
        Args:
            feature_map: Parameterized circuit to be used as the feature map. If None is given,
                the `ZZFeatureMap` is used with two qubits.
            input_params: the parameters of feature_map which will be bound to the values of our
                input data. In other words, these are the inputs to our feature map. 
            free_params: the parameters of feature_map which will be optimized via an outer-training
                loop in order to tune the performance of our resulting feature map. In other words,
                these are the learnable parameters of our feature map.
            enforce_psd: Project to closest positive semidefinite matrix if x = y.
                Only enforced when not using the state vector simulator. Default True.
            batch_size: Number of circuits to batch together for computation. Default 1000.
            quantum_instance: Quantum Instance or Backend
        """

        # free_params and input_params must together account for all of the parameters 
        # in feature_map
        if set(feature_map.parameters) != set(input_params).union(set(free_params)):
            raise ValueError("The union of input_params and free_params must equal " 
                             "the set of feature_map's parameters")

        # free_params and input_params must be disjoint
        assert len(set(input_params).intersection(set(free_params))) == 0
 
        self._feature_map = feature_map if feature_map else ZZFeatureMap(2)
        self._unbound_feature_map = self._feature_map
        if (free_params is None) or (len(free_params) == 0):
            return super.__init__(feature_map,
                                  enforce_psd,
                                  batch_size,
                                  quantum_instance)
        self._free_params = free_params
        self._input_params = input_params if input_params is not None \
            else [p for p in feature_map.parameters if p not in self._free_params]

        self._free_param_bindings = {param: param for param in self._free_params}

        super().__init__(feature_map,
                       enforce_psd,
                       batch_size,
                       quantum_instance)

    @property
    def feature_map(self) -> QuantumCircuit:
        """Returns feature map"""
        return self._unbound_feature_map.assign_parameters(self._free_param_bindings)

    @property
    def unbound_feature_map(self) -> QuantumCircuit:
        """Returns feature map"""
        return self._unbound_feature_map

    @property
    def quantum_instance(self) -> QuantumInstance:
        """Returns quantum instance"""
        return self._quantum_instance

    @property
    def free_param_bindings(self) -> Dict:
        """Returns free_params parameter bindings"""
        return self._free_param_bindings

    @free_param_bindings.setter
    def free_param_bindings(param_values):
        if isinstance(param_values, list):
            # make sure the param values are the right length o.w. the
            # ordering is arbitrary
            self._free_param_bindings = {param: param_values[i] 
                                        for i,param in enumerate(self._free_params)}
        elif isinstance(param_values, dict):
            self._free_param_bindings.update(param_values)

    @quantum_instance.setter
    def quantum_instance(
        self, quantum_instance: Union[Backend, BaseBackend, QuantumInstance]
    ) -> None:
        """Sets quantum instance"""
        if isinstance(quantum_instance, (BaseBackend, Backend)):
            self._quantum_instance = QuantumInstance(quantum_instance)
        else:
            self._quantum_instance = quantum_instance

    def bind_free_params(self, free_param_values):
        """Binds free param values to the feature map"""
        if isinstance(free_param_values, dict):
            self._free_param_bindings.update(free_param_values)
        elif isinstance(free_param_values,(List, np.ndarray)):
            assert len(free_param_values) == len(self._free_params)
            param_binds = {param: free_param_values[i] \
                           for i, param in enumerate(self._free_params)}
            self._free_param_bindings.update(param_binds)

        self._feature_map = self._unbound_feature_map.bind_parameters(self._free_param_bindings)
     
    def _alignment(self,
                   params, 
                   x_vec=None, 
                   y_vec=None):
        r"""
        docstring
        """

        assert x_vec is not None
        assert y_vec is not None
        
        # check that params are the right dimension
        self.bind_free_params(params)
        K = self.evaluate(x_vec)
        y = np.array(y_vec)
        
        # The -1 is here because qiskit 
        # optimizers minimize by default
        return -1 * y.T @ K @ y

    def _weighted_alignment(self, 
                           params, 
                           x_vec=None, 
                           y_vec=None):
        r"""
        docstring
        """

        assert x_vec is not None
        assert y_vec is not None
        from qiskit_machine_learning.algorithms import QSVC    
        
        # check that params are the right dimension
        self.bind_free_params(params)
        
        qsvc = QSVC(quantum_kernel = self)
        qsvc.fit(x_vec, y_vec)
        score = qsvc.score(x_vec, y_vec)

        # The dual coefficients are equal
        # to the Lagrange multipliers, termwise
        # multiplied by the corresponding datapoint's 
        # label. Only nonzero coefficients are returned
        a = qsvc.dual_coef_[0]
        
        # Get the indices of our data which correspond to
        # support vectors
        sv = qsvc.support_
        
        #Filter out entries for non-support vectors
        K = self.evaluate(x_vec)[sv,:][:,sv]

        # The -1 is here because qiskit 
        # optimizers minimize by default
        return np.sum(np.abs(a)) - 1/2 * (a.T @ K @ a) + np.sum(a)

    def _model_complexity(self, params, x_vec=None, y_vec=None):
        r"""
        docstring
        """

        assert x_vec is not None
        assert y_vec is not None
        
        # check that params are the right dimension
        self.bind_free_params(params)
        K = self.evaluate(x_vec)
        y = np.array(y_vec)
        
        # The -1 is here because qiskit 
        # optimizers minimize by default
        return -1 * y.T @ np.linalg.pinv(K) @ y

    def train_kernel(self,
                     objective_function = 'alignment',
                     optimizer = SPSA(),
                     x_vec=None,
                     y_vec=None,
                     bounds = None,
                     initial_point = None,
                     gradient_fn = None):

        # objective_function(params) --> scalar

        if isinstance(objective_function, str):
            assert x_vec is not None and y_vec is not None, "x_vec and y_vec must be supplied " \
                   "if a pre-made objective function is chosen."
            if objective_function == 'weighted_alignment':
                obj_fun = partial(self._weighted_alignment, x_vec=x_vec, y_vec=y_vec)
            elif objective_function == 'alignment':
                obj_fun = partial(self._alignment, x_vec=x_vec, y_vec=y_vec)
            elif objective_function == 'model_complexity':
                obj_fun = partial(self._model_complexity, x_vec=x_vec, y_vec=y_vec)
            else:
                raise ValueError('{} is not a valid choice of objecive function'\
                                 ''.format(objective_function))
        elif isinstance(objective_function, Callable):
            obj_fun = objective_function

        nparams = len(self._free_params)
        if initial_point is None:
            initial_point = np.random.rand(nparams)

        start = time.time()

        #logger.info('Starting optimizer.\nbounds=%s\ninitial point=%s', bounds, initial_point)
        opt_params, opt_val, num_optimizer_evals = optimizer.optimize(nparams,
                                                                      obj_fun,
                                                                      variable_bounds=bounds,
                                                                      initial_point=initial_point,
                                                                      gradient_function=gradient_fn)
        eval_time = time.time() - start

        result = VariationalResult()
        result.optimizer_evals = num_optimizer_evals
        result.optimizer_time = eval_time
        result.optimal_value = opt_val
        result.optimal_point = opt_params
        result.optimal_parameters = dict(zip(self._free_params, opt_params))

        self.bind_free_params(opt_params)

        return result
    
    def construct_circuit(
        self,
        x: ParameterVector,
        y: ParameterVector = None,
        measurement: bool = True,
        is_statevector_sim: bool = False,
    ) -> QuantumCircuit:
        r"""
        Construct inner product circuit for given datapoints and feature map.

        If using `statevector_simulator`, only construct circuit for :math:`\Psi(x)|0\rangle`,
        otherwise construct :math:`Psi^dagger(y) x Psi(x)|0>`
        If y is None and not using `statevector_simulator`, self inner product is calculated.

        Args:
            x: first data point parameter vector
            y: second data point parameter vector, ignored if using statevector simulator
            measurement: include measurement if not using statevector simulator
            is_statevector_sim: use state vector simulator

        Returns:
            QuantumCircuit

        Raises:
            ValueError:
                - x and/or y have incompatible dimension with feature map
        """

        # Enure no free parameters are bound to objects (such as other Parameter) objects
        if np.array(list(self._free_param_bindings.values())).dtype == np.dtype('O'):
            raise ValueError("Feature Map contains unbound free parameters.")

        # Ensure no free parameters are bound to NaN values
        if np.any(np.isnan(list(self._free_param_bindings.values()))):
            raise ValueError("Feature Map contains unbound free parameters.")

        return super().construct_circuit(x,y,measurement,is_statevector_sim)
