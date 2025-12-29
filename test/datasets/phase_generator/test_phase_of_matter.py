"""
Unit tests for the phase_of_matter module.

This module contains comprehensive tests for the quantum phase of matter dataset generator.
"""

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import os
import tempfile

# Import the module to be tested
from qiskit_machine_learning.datasets.phase_generator import phase_of_matter


class TestPhaseOfMatter(unittest.TestCase):
    """Test cases for the phase_of_matter module."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock DataFrame that we'll use as a return value for the mocked generate_dataset
        self.mock_data = {
            'Parameter_J': [-0.5, 0.5],
            'Phase': ['AFM', 'FM'],
            'Ground_State_Energy': [-2.5, -1.5],
            'Hamiltonian': ['H1', 'H2']
        }
        self.mock_df = pd.DataFrame(self.mock_data)

        # Create more complex mock DataFrames for specific models
        self.mock_haldane_data = {
            'Parameter_J': [-0.5, 0.5],
            'Parameter_h1': [-0.2, 0.2],
            'Parameter_h2': [0.1, -0.1],
            'Phase': ['Haldane', 'Trivial'],
            'Ground_State_Energy': [-2.7, -1.8],
            'Hamiltonian': ['H3', 'H4']
        }
        self.mock_haldane_df = pd.DataFrame(self.mock_haldane_data)

        self.mock_annni_data = {
            'Parameter_J1': [-0.5, 0.5],
            'Parameter_J2': [0.3, -0.3],
            'Parameter_B': [0.2, -0.2],
            'Phase': ['Para', 'Ferro'],
            'Ground_State_Energy': [-3.0, -2.0],
            'Hamiltonian': ['H5', 'H6']
        }
        self.mock_annni_df = pd.DataFrame(self.mock_annni_data)

        self.mock_cluster_data = {
            'Parameter_J1': [-0.5, 0.5],
            'Parameter_J2': [0.4, -0.4],
            'Phase': ['SPT', 'Trivial'],
            'Ground_State_Energy': [-2.2, -1.9],
            'Hamiltonian': ['H7', 'H8']
        }
        self.mock_cluster_df = pd.DataFrame(self.mock_cluster_data)

    @patch('qiskit_machine_learning.datasets.phase_generator.phase_of_matter.QuantumDataGenerator')
    def test_generate_dataset_heisenberg(self, mock_quantum_data_generator):
        """Test generating a Heisenberg model dataset."""
        # Set up the mock
        mock_instance = mock_quantum_data_generator.return_value
        mock_instance.generate_dataset.return_value = self.mock_df

        # Call the function
        result = phase_of_matter.generate_dataset(
            model_name='heisenberg',
            num_qubits=2,
            num_samples=2
        )

        # Check the result
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)
        self.assertIn('Phase', result.columns)
        
        # Check that the QuantumDataGenerator was initialized with correct parameters
        mock_quantum_data_generator.assert_called_once_with(
            num_qubits=2,
            vqe_iterations=30,
            ansatz_depth=2,
            backend='aer_simulator'
        )
        
        # Check that generate_dataset was called with correct parameters
        mock_instance.generate_dataset.assert_called_once()
        args, kwargs = mock_instance.generate_dataset.call_args
        self.assertEqual(args[0].__name__, 'HeisenbergXXX')
        self.assertEqual(kwargs['num_samples'], 2)

    @patch('phase_of_matter.QuantumDataGenerator')
    def test_generate_dataset_haldane(self, mock_quantum_data_generator):
        """Test generating a Haldane Chain model dataset."""
        # Set up the mock
        mock_instance = mock_quantum_data_generator.return_value
        mock_instance.generate_dataset.return_value = self.mock_haldane_df

        # Call the function
        result = phase_of_matter.generate_dataset(
            model_name='haldane',
            num_qubits=3,
            num_samples=2,
            param_ranges={
                'J_range': (-0.5, 0.5),
                'h1_range': (-0.2, 0.2),
                'h2_range': (-0.1, 0.1)
            }
        )

        # Check the result
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)
        self.assertIn('Parameter_h1', result.columns)
        self.assertIn('Parameter_h2', result.columns)
        
        # Check that generate_dataset was called with correct parameters
        args, kwargs = mock_instance.generate_dataset.call_args
        self.assertEqual(args[0].__name__, 'HaldaneChain')
        self.assertEqual(kwargs['num_samples'], 2)
        self.assertEqual(args[1]['J_range'], (-0.5, 0.5))
        self.assertEqual(args[1]['h1_range'], (-0.2, 0.2))
        self.assertEqual(args[1]['h2_range'], (-0.1, 0.1))

    @patch('phase_of_matter.QuantumDataGenerator')
    def test_generate_dataset_annni(self, mock_quantum_data_generator):
        """Test generating an ANNNI model dataset."""
        # Set up the mock
        mock_instance = mock_quantum_data_generator.return_value
        mock_instance.generate_dataset.return_value = self.mock_annni_df

        # Call the function
        result = phase_of_matter.generate_dataset(
            model_name='annni',
            num_qubits=3,
            num_samples=2,
            param_ranges={
                'J1_range': (-0.5, 0.5),
                'J2_range': (-0.3, 0.3),
                'B_range': (-0.2, 0.2)
            }
        )

        # Check the result
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)
        self.assertIn('Parameter_J1', result.columns)
        self.assertIn('Parameter_J2', result.columns)
        self.assertIn('Parameter_B', result.columns)

    @patch('phase_of_matter.QuantumDataGenerator')
    def test_generate_dataset_cluster(self, mock_quantum_data_generator):
        """Test generating a Cluster model dataset."""
        # Set up the mock
        mock_instance = mock_quantum_data_generator.return_value
        mock_instance.generate_dataset.return_value = self.mock_cluster_df

        # Call the function
        result = phase_of_matter.generate_dataset(
            model_name='cluster',
            num_qubits=3,
            num_samples=2
        )

        # Check the result
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)
        self.assertIn('Parameter_J1', result.columns)
        self.assertIn('Parameter_J2', result.columns)

    @patch('phase_of_matter.QuantumDataGenerator')
    def test_generate_dataset_invalid_model(self, mock_quantum_data_generator):
        """Test generating a dataset with an invalid model name."""
        # Set up the mock
        mock_instance = mock_quantum_data_generator.return_value

        # Check that an exception is raised for an invalid model name
        with self.assertRaises(ValueError):
            phase_of_matter.generate_dataset(
                model_name='invalid_model',
                num_qubits=2,
                num_samples=2
            )

    @patch('qiskit_machine_learning.datasets.phase_generator.phase_of_matter.generate_dataset')
    def test_generate_heisenberg_dataset(self, mock_generate_dataset):
        """Test the generate_heisenberg_dataset function."""
        # Set up the mock
        mock_generate_dataset.return_value = self.mock_df

        # Call the function
        result = phase_of_matter.generate_heisenberg_dataset(
            num_qubits=2,
            num_samples=2,
            j_range=(-0.5, 0.5)
        )

        # Check the result
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)
        
        # Check that generate_dataset was called with correct parameters
        mock_generate_dataset.assert_called_once_with(
            model_name='heisenberg',
            num_qubits=2,
            num_samples=2,
            param_ranges={'J_range': (-0.5, 0.5)},
            vqe_iterations=30,
            ansatz_depth=2
        )

    @patch('phase_of_matter.generate_dataset')
    def test_generate_haldane_dataset(self, mock_generate_dataset):
        """Test the generate_haldane_dataset function."""
        # Set up the mock
        mock_generate_dataset.return_value = self.mock_haldane_df

        # Call the function
        result = phase_of_matter.generate_haldane_dataset(
            num_qubits=3,
            num_samples=2,
            j_range=(-0.5, 0.5),
            h1_range=(-0.2, 0.2),
            h2_range=(-0.1, 0.1)
        )

        # Check the result
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)
        
        # Check that generate_dataset was called with correct parameters
        mock_generate_dataset.assert_called_once_with(
            model_name='haldane',
            num_qubits=3,
            num_samples=2,
            param_ranges={
                'J_range': (-0.5, 0.5),
                'h1_range': (-0.2, 0.2),
                'h2_range': (-0.1, 0.1)
            },
            vqe_iterations=30,
            ansatz_depth=2
        )

    @patch('phase_of_matter.generate_dataset')
    def test_generate_annni_dataset(self, mock_generate_dataset):
        """Test the generate_annni_dataset function."""
        # Set up the mock
        mock_generate_dataset.return_value = self.mock_annni_df

        # Call the function
        result = phase_of_matter.generate_annni_dataset(
            num_qubits=3,
            num_samples=2,
            j1_range=(-0.5, 0.5),
            j2_range=(-0.3, 0.3),
            b_range=(-0.2, 0.2)
        )

        # Check the result
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)
        
        # Check that generate_dataset was called with correct parameters
        mock_generate_dataset.assert_called_once_with(
            model_name='annni',
            num_qubits=3,
            num_samples=2,
            param_ranges={
                'J1_range': (-0.5, 0.5),
                'J2_range': (-0.3, 0.3),
                'B_range': (-0.2, 0.2)
            },
            vqe_iterations=30,
            ansatz_depth=2
        )

    @patch('phase_of_matter.generate_dataset')
    def test_generate_cluster_dataset(self, mock_generate_dataset):
        """Test the generate_cluster_dataset function."""
        # Set up the mock
        mock_generate_dataset.return_value = self.mock_cluster_df

        # Call the function
        result = phase_of_matter.generate_cluster_dataset(
            num_qubits=3,
            num_samples=2,
            j1_range=(-0.5, 0.5),
            j2_range=(-0.4, 0.4)
        )

        # Check the result
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)
        
        # Check that generate_dataset was called with correct parameters
        mock_generate_dataset.assert_called_once_with(
            model_name='cluster',
            num_qubits=3,
            num_samples=2,
            param_ranges={
                'J1_range': (-0.5, 0.5),
                'J2_range': (-0.4, 0.4)
            },
            vqe_iterations=30,
            ansatz_depth=2
        )

    def test_save_dataset(self):
        """Test saving a dataset to a CSV file."""
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp_file:
            filename = temp_file.name

        try:
            # Create a more complex DataFrame with list and complex object columns
            df = self.mock_df.copy()
            df['list_column'] = [[1, 2], [3, 4]]
            df['complex_column'] = [complex(1, 2), complex(3, 4)]
            
            # Save the dataset
            phase_of_matter.save_dataset(df, filename)
            
            # Check that the file exists
            self.assertTrue(os.path.exists(filename))
            
            # Read the file and check its contents
            loaded_df = pd.read_csv(filename)
            self.assertEqual(len(loaded_df), 2)
            self.assertIn('Phase', loaded_df.columns)
            self.assertIn('list_column', loaded_df.columns)
            self.assertIn('complex_column', loaded_df.columns)
            
            # Check that list and complex columns were converted to strings
            self.assertTrue(isinstance(loaded_df['list_column'].iloc[0], str))
            self.assertTrue(isinstance(loaded_df['complex_column'].iloc[0], str))
            
        finally:
            # Clean up the temporary file
            if os.path.exists(filename):
                os.remove(filename)

    @patch('qiskit_machine_learning.datasets.phase_generator.phase_of_matter.generate_heisenberg_dataset')
    @patch('qiskit_machine_learning.datasets.phase_generator.phase_of_matter.generate_haldane_dataset')
    @patch('qiskit_machine_learning.datasets.phase_generator.phase_of_matter.generate_annni_dataset')
    @patch('qiskit_machine_learning.datasets.phase_generator.phase_of_matter.generate_cluster_dataset')
    @patch('qiskit_machine_learning.datasets.phase_generator.phase_of_matter.save_dataset')
    def test_run_example(self, mock_save, mock_cluster, mock_annni, mock_haldane, mock_heisenberg):
        """Test the run_example function."""
        # Set up the mocks
        mock_heisenberg.return_value = self.mock_df
        mock_haldane.return_value = self.mock_haldane_df
        mock_annni.return_value = self.mock_annni_df
        mock_cluster.return_value = self.mock_cluster_df
        
        # Call the function
        phase_of_matter.run_example()
        
        # Check that all the dataset generation functions were called
        mock_heisenberg.assert_called_once_with(num_qubits=2, num_samples=2)
        mock_haldane.assert_called_once_with(num_qubits=2, num_samples=2)
        mock_annni.assert_called_once_with(num_qubits=2, num_samples=2)
        mock_cluster.assert_called_once_with(num_qubits=2, num_samples=2)
        
        # Check that save_dataset was called for each model
        self.assertEqual(mock_save.call_count, 4)


if __name__ == '__main__':
    unittest.main()
