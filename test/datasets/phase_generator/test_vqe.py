"""
Test suite for the VQE (Variational Quantum Eigensolver) implementation.

This module contains comprehensive test cases for the VQE class,
covering initialization, optimization, and result handling.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from scipy.optimize import OptimizeResult

# Import the VQE class from qiskit_machine_learning.datasets.phase_generator
from qiskit_machine_learning.datasets.phase_generator.vqe import VQE

class TestVQE(unittest.TestCase):
    """Test cases for the VQE implementation."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock objects
        self.mock_estimator = Mock()
        self.mock_ansatz = Mock()
        self.mock_ansatz.num_parameters = 2
        self.mock_ansatz.parameters = ['param1', 'param2']
        
        # Mock transpiled ansatz
        self.mock_transpiled_ansatz = Mock()
        
        # Configure mocks
        self.mock_estimator.run.return_value = Mock()
        self.mock_estimator.run.return_value.result.return_value = Mock()
        self.mock_estimator.run.return_value.result.return_value.values = [0.5]

    def test_init_default_values(self):
        """Test VQE initialization with default values."""
        vqe = VQE(self.mock_estimator, self.mock_ansatz)
        
        self.assertEqual(vqe._estimator, self.mock_estimator)
        self.assertEqual(vqe._ansatz, self.mock_ansatz)
        self.assertEqual(vqe._optimizer, "SLSQP")
        self.assertEqual(vqe._max_iterations, 50)
        self.assertIsNone(vqe._gradient)
        self.assertIsNone(vqe._initial_point)
        self.assertIsNone(vqe._callback)
        self.assertIsNone(vqe._result)
        self.assertIsNone(vqe._optimal_parameters)
        self.assertIsNone(vqe._optimal_value)
        self.assertEqual(vqe._energies, [])
        self.assertEqual(vqe._iterations, 0)

    def test_init_custom_values(self):
        """Test VQE initialization with custom values."""
        mock_gradient = Mock()
        mock_callback = Mock()
        initial_point = np.array([0.1, 0.2])
        custom_optimizer = Mock()
        
        vqe = VQE(
            self.mock_estimator,
            self.mock_ansatz,
            optimizer=custom_optimizer,
            gradient=mock_gradient,
            initial_point=initial_point,
            callback=mock_callback,
            max_iterations=100
        )
        
        self.assertEqual(vqe._optimizer, "custom")
        self.assertEqual(vqe._optimizer_callable, custom_optimizer)
        self.assertEqual(vqe._gradient, mock_gradient)
        self.assertTrue(np.array_equal(vqe._initial_point, initial_point))
        self.assertEqual(vqe._callback, mock_callback)
        self.assertEqual(vqe._max_iterations, 100)

    def test_init_string_optimizer(self):
        """Test VQE initialization with string optimizer."""
        vqe = VQE(self.mock_estimator, self.mock_ansatz, optimizer="COBYLA")
        
        self.assertEqual(vqe._optimizer, "COBYLA")
        self.assertIsNone(vqe._optimizer_callable)

    @patch('qiskit_machine_learning.datasets.phase_generator.vqe.transpile')
    def test_prepare_circuit(self, mock_transpile):
        """Test the circuit preparation method."""
        mock_transpile.return_value = self.mock_transpiled_ansatz
        
        vqe = VQE(self.mock_estimator, self.mock_ansatz)
        result = vqe._prepare_circuit()
        
        # Check if transpile was called correctly
        mock_transpile.assert_called_once_with(
            self.mock_ansatz,
            basis_gates=['rx', 'ry', 'rz', 'cx'],
            optimization_level=1
        )
        
        # Check if result is the transpiled circuit
        self.assertEqual(result, self.mock_transpiled_ansatz)
        
        # Check if caching works
        vqe._prepare_circuit()
        # Should still be called only once
        mock_transpile.assert_called_once()

    def test_cost_function(self):
        """Test the cost function calculation."""
        vqe = VQE(self.mock_estimator, self.mock_ansatz)
        vqe._transpiled_ansatz = self.mock_transpiled_ansatz
        
        mock_hamiltonian = Mock()
        parameters = np.array([0.1, 0.2])
        
        energy = vqe._cost_function(parameters, mock_hamiltonian)
        
        # Check if estimator was called correctly
        self.mock_estimator.run.assert_called_once_with(
            self.mock_transpiled_ansatz,
            mock_hamiltonian,
            parameter_values=[parameters]
        )
        
        # Check if energy is stored and returned correctly
        self.assertEqual(energy, 0.5)
        self.assertEqual(vqe._energies, [0.5])

    def test_internal_callback(self):
        """Test the internal callback function."""
        mock_callback = Mock()
        vqe = VQE(self.mock_estimator, self.mock_ansatz, callback=mock_callback)
        vqe._energies = [0.5]
        
        parameters = np.array([0.1, 0.2])
        vqe._internal_callback(parameters)
        
        # Check if iterations are incremented
        self.assertEqual(vqe._iterations, 1)
        
        # Check if callback is called with correct data
        mock_callback.assert_called_once()
        callback_args = mock_callback.call_args[0][0]
        self.assertEqual(callback_args['iteration'], 1)
        self.assertEqual(callback_args['energy'], 0.5)
        self.assertTrue(np.array_equal(callback_args['parameters'], parameters))
        self.assertEqual(callback_args['nfev'], 1)

    @patch('numpy.random.uniform')
    @patch('qiskit_machine_learning.datasets.phase_generator.vqe.minimize')
    def test_compute_minimum_eigenvalue_default_optimizer(self, mock_minimize, mock_uniform):
        """Test compute_minimum_eigenvalue with default optimizer."""
        # Mock the minimize function
        mock_result = OptimizeResult(
            x=np.array([0.1, 0.2]),
            fun=0.5,
            success=True,
            nfev=10
        )
        mock_minimize.return_value = mock_result
        
        # Mock the random initial point
        mock_uniform.return_value = np.array([0.3, 0.4])
        
        vqe = VQE(self.mock_estimator, self.mock_ansatz)
        vqe._transpiled_ansatz = self.mock_transpiled_ansatz
        
        mock_operator = Mock()
        result = vqe.compute_minimum_eigenvalue(mock_operator)
        
        # Check if minimize was called correctly
        mock_minimize.assert_called_once()
        # Extract the first positional argument (cost function)
        cost_fn = mock_minimize.call_args[0][0]
        # Extract the second positional argument (initial point)
        initial_point = mock_minimize.call_args[0][1]
        # Extract the keyword arguments
        kwargs = mock_minimize.call_args[1]
        
        self.assertTrue(np.array_equal(initial_point, np.array([0.3, 0.4])))
        self.assertEqual(kwargs['method'], "SLSQP")
        self.assertEqual(kwargs['options']['maxiter'], 50)
        self.assertEqual(kwargs['options']['ftol'], 1e-4)
        
        # Check if cost function works
        self.mock_estimator.run.return_value.result.return_value.values = [0.7]
        energy = cost_fn(np.array([0.5, 0.6]))
        self.assertEqual(energy, 0.7)
        
        # Check result structure
        self.assertEqual(result['optimal_value'], 0.5)
        self.assertTrue(np.array_equal(result['optimal_point'], np.array([0.1, 0.2])))
        self.assertEqual(result['optimal_parameters'], {'param1': 0.1, 'param2': 0.2})
        self.assertEqual(result['eigenvalue'], 0.5)

    def test_compute_minimum_eigenvalue_custom_optimizer(self):
        """Test compute_minimum_eigenvalue with custom optimizer."""
        # Create a custom optimizer function that returns a custom result
        def custom_optimizer(cost_fn, initial_point, callback=None):
            # Call the cost function and callback a few times
            for i in range(3):
                params = np.array([0.1 * i, 0.2 * i])
                cost = cost_fn(params)
                if callback:
                    callback(params)
            
            # Return a custom result object
            return {'x': np.array([0.3, 0.6]), 'fun': 0.2}
        
        vqe = VQE(
            self.mock_estimator,
            self.mock_ansatz,
            optimizer=custom_optimizer,
            initial_point=np.array([0.0, 0.0])
        )
        vqe._transpiled_ansatz = self.mock_transpiled_ansatz
        
        # Configure mock to return different energy values
        energy_values = [0.8, 0.5, 0.2]
        def side_effect(*args, **kwargs):
            mock_result = Mock()
            mock_result.result.return_value = Mock()
            mock_result.result.return_value.values = [energy_values[len(vqe._energies)]]
            return mock_result
        
        self.mock_estimator.run.side_effect = side_effect
        
        mock_operator = Mock()
        result = vqe.compute_minimum_eigenvalue(mock_operator)
        
        # Check if custom optimizer was used correctly
        self.assertEqual(len(vqe._energies), 3)
        self.assertEqual(vqe._energies, [0.8, 0.5, 0.2])
        self.assertEqual(vqe._iterations, 3)
        
        # Check result structure
        self.assertEqual(result['optimal_value'], 0.2)
        self.assertTrue(np.array_equal(result['optimal_point'], np.array([0.3, 0.6])))
        self.assertEqual(result['eigenvalue'], 0.2)

    def test_properties(self):
        """Test the property getters."""
        vqe = VQE(self.mock_estimator, self.mock_ansatz)
        vqe._optimal_parameters = np.array([0.1, 0.2])
        vqe._optimal_value = 0.5
        vqe._energies = [0.8, 0.6, 0.5]
        
        self.assertTrue(np.array_equal(vqe.optimal_parameters, np.array([0.1, 0.2])))
        self.assertEqual(vqe.optimal_value, 0.5)
        self.assertEqual(vqe.energies, [0.8, 0.6, 0.5])

    @patch('qiskit_machine_learning.datasets.phase_generator.vqe.minimize')
    def test_provided_initial_point(self, mock_minimize):
        """Test that provided initial points are used correctly."""
        mock_result = OptimizeResult(
            x=np.array([0.1, 0.2]),
            fun=0.5,
            success=True
        )
        mock_minimize.return_value = mock_result
        
        initial_point = np.array([0.3, 0.4])
        vqe = VQE(
            self.mock_estimator,
            self.mock_ansatz,
            initial_point=initial_point
        )
        vqe._transpiled_ansatz = self.mock_transpiled_ansatz
        
        mock_operator = Mock()
        vqe.compute_minimum_eigenvalue(mock_operator)
        
        # Check if the provided initial point was used
        passed_initial_point = mock_minimize.call_args[0][1]
        self.assertTrue(np.array_equal(passed_initial_point, initial_point))


if __name__ == '__main__':
    unittest.main()
