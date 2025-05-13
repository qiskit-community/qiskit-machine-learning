"""
VQE Implementation Module

This module provides an implementation of the Variational Quantum Eigensolver (VQE)
algorithm compatible with Qiskit.
"""

import numpy as np
from typing import Optional, Union, List, Callable, Dict, Any
from scipy.optimize import minimize
from qiskit.circuit import QuantumCircuit, Parameter  
from qiskit.primitives import BaseEstimatorV2
from qiskit.compiler import transpile

class VQE:
    """
    Variational Quantum Eigensolver implementation compatible with Qiskit.

    This class implements the VQE algorithm to find the lowest eigenvalue
    of a given Hamiltonian using parameterized quantum circuits.
    """

    def __init__(
        self,
        estimator: BaseEstimatorV2,
        ansatz: QuantumCircuit,
        optimizer: Union[Callable, str] = "SLSQP",
        *,  # Forces remaining arguments to be keyword arguments
        gradient: Optional[Callable] = None,
        initial_point: Optional[np.ndarray] = None,
        callback: Optional[Callable] = None,
        max_iterations: int = 50  # Default max iterations reduced for speed
    ):
        """
        Initialize the VQE algorithm.

        Args:
            estimator: Qiskit estimator primitive for expectation value calculations
            ansatz: Parameterized quantum circuit for the variational form
            optimizer: Classical optimizer method or callable
            gradient: Optional method to compute gradients
            initial_point: Initial parameters for the ansatz circuit
            callback: Optional callback function called after each optimization iteration
            max_iterations: Maximum number of iterations for the optimizer
        """
        self._estimator = estimator
        self._ansatz = ansatz
        self._max_iterations = max_iterations

        # Handle optimizer
        if isinstance(optimizer, str):
            self._optimizer = optimizer
            self._optimizer_callable = None
        else:
            self._optimizer = "custom"
            self._optimizer_callable = optimizer

        self._gradient = gradient
        self._initial_point = initial_point
        self._callback = callback

        # Results storage
        self._result = None
        self._optimal_parameters = None
        self._optimal_value = None
        self._energies = []
        self._iterations = 0

        # Transpile the circuit once for efficiency
        self._transpiled_ansatz = None

    def _prepare_circuit(self):
        """Transpile the ansatz circuit for more efficient execution."""
        if self._transpiled_ansatz is None:
            self._transpiled_ansatz = transpile(
                self._ansatz,
                basis_gates=['rx', 'ry', 'rz', 'cx'],
                optimization_level=1
            )
        return self._transpiled_ansatz

    def _cost_function(self, parameters: np.ndarray, hamiltonian) -> float:
        """
        Compute expectation value of the Hamiltonian for given parameters.

        Args:
            parameters: Ansatz circuit parameters
            hamiltonian: The Hamiltonian operator

        Returns:
            float: Expectation value (energy)
        """
        job = self._estimator.run(
            self._transpiled_ansatz,
            hamiltonian,
            parameter_values=[parameters]
        )
        energy = job.result().values[0]
        self._energies.append(float(energy))
        return float(energy)

    def _internal_callback(self, xk):
        """
        Internal callback to track optimization progress.

        Args:
            xk: Current parameter values
        """
        self._iterations += 1
        current_energy = self._energies[-1] if self._energies else float('nan')

        # Call user-provided callback if available
        if self._callback is not None:
            iteration_data = {
                'iteration': self._iterations,
                'energy': current_energy,
                'parameters': xk,
                'nfev': self._iterations
            }
            self._callback(iteration_data)

    def compute_minimum_eigenvalue(self, operator) -> Dict[str, Any]:
        """
        Execute the VQE algorithm to find the minimum eigenvalue.

        Args:
            operator: Hamiltonian as a SparsePauliOp object

        Returns:
            dict: Results dictionary containing optimal parameters, eigenvalue,
                  optimization success status, and convergence information
        """
        # Reset tracking variables
        self._energies = []
        self._iterations = 0

        # Prepare ansatz circuit
        self._prepare_circuit()

        # Determine initial point if not provided
        if self._initial_point is None:
            self._initial_point = np.random.uniform(
                -np.pi, np.pi, self._ansatz.num_parameters
            )

        # Create cost function with fixed operator
        def cost_fn(params):
            return self._cost_function(params, operator)

        # Set up optimizer options
        optimizer_options = {'maxiter': self._max_iterations}
        if self._optimizer == "SLSQP":
            optimizer_options.update({'ftol': 1e-4})  # Reduced tolerance for faster convergence

        # Execute the optimization
        if self._optimizer_callable is not None:
            # Use custom optimizer if provided
            result = self._optimizer_callable(
                cost_fn,
                self._initial_point,
                callback=self._internal_callback
            )
        else:
            # Use scipy's minimize with specified method
            result = minimize(
                cost_fn,
                self._initial_point,
                method=self._optimizer,
                callback=self._internal_callback,
                options=optimizer_options
            )

        # Store and prepare results
        self._result = result

        # Extract optimal parameters and values
        if hasattr(result, 'x'):
            self._optimal_parameters = result.x
            self._optimal_value = result.fun
        else:
            # For custom optimizers that may return different result format
            self._optimal_parameters = result.get('x', None)
            self._optimal_value = result.get('fun', self._energies[-1] if self._energies else None)

        # Create results dictionary
        vqe_results = {
            'optimal_point': self._optimal_parameters,
            'optimal_value': self._optimal_value,
            'optimal_parameters': dict(zip(
                [str(p) for p in self._ansatz.parameters],
                self._optimal_parameters
            )),
            'cost_function_evals': len(self._energies),
            'optimizer_time': None,  # Would need timing information
            'optimizer_result': result,
            'eigenvalue': self._optimal_value,
            'eigenstate': None,  # Would need state preparation
            'energy_evaluation': self._energies
        }

        return vqe_results

    @property
    def optimal_parameters(self):
        """Return the optimal parameters found during optimization."""
        return self._optimal_parameters

    @property
    def optimal_value(self):
        """Return the optimal value (minimum eigenvalue) found."""
        return self._optimal_value

    @property
    def energies(self):
        """Return the energy values computed during optimization."""
        return self._energies