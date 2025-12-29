"""
Quantum Phase of Matter Dataset Generator

This module provides a unified interface for generating datasets related to
different quantum phase of matter models using VQE energy estimations.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Union, List, Tuple

# Import the Hamiltonian models
from qiskit_machine_learning.datasets.phase_generator.hamiltonians.heisenberg import HeisenbergXXX
from qiskit_machine_learning.datasets.phase_generator.hamiltonians.haldane_chain import HaldaneChain
from qiskit_machine_learning.datasets.phase_generator.hamiltonians.annni import ANNNIModel
from qiskit_machine_learning.datasets.phase_generator.hamiltonians.cluster import ClusterModel

# Import VQE implementation
from vqe import VQE

# Import the QuantumDataGenerator
from hamiltonians.hamiltonian_base import QuantumDataGenerator


def generate_dataset(
    model_name: str,
    num_qubits: int = 3,
    num_samples: int = 10,
    param_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    vqe_iterations: int = 30,
    ansatz_depth: int = 2,
    backend: str = 'aer_simulator'
) -> pd.DataFrame:
    """
    Generate a dataset for a specified quantum phase of matter model.

    Args:
        model_name: Name of the model ('heisenberg', 'haldane', 'annni', or 'cluster')
        num_qubits: Number of qubits in the system
        num_samples: Number of Hamiltonian samples to generate
        param_ranges: Dictionary of parameter ranges for the model
                      (specific to each model, see documentation)
        vqe_iterations: Maximum number of iterations for VQE optimization
        ansatz_depth: Depth of the ansatz circuit for VQE
        backend: Qiskit backend to use for simulation

    Returns:
        DataFrame containing Hamiltonian data, parameters, phase labels, and energy estimates

    Example:
        # Generate Heisenberg model dataset
        heisenberg_df = generate_dataset(
            'heisenberg',
            num_qubits=3,
            num_samples=10,
            param_ranges={'J_range': (-1, 1)}
        )
    """
    # Set default parameter ranges if not provided
    if param_ranges is None:
        param_ranges = {}

    # Initialize the quantum data generator
    dataset_gen = QuantumDataGenerator(
        num_qubits=num_qubits,
        vqe_iterations=vqe_iterations,
        ansatz_depth=ansatz_depth,
        backend=backend
    )

    # Select the appropriate model and parameter ranges
    if model_name.lower() == 'heisenberg':
        model_class = HeisenbergXXX
        # Set default J_range if not provided
        if 'J_range' not in param_ranges:
            param_ranges['J_range'] = (-1, 1)
        
    elif model_name.lower() == 'haldane':
        model_class = HaldaneChain
        # Set default parameter ranges if not provided
        if 'J_range' not in param_ranges:
            param_ranges['J_range'] = (-1, 1)
        if 'h1_range' not in param_ranges:
            param_ranges['h1_range'] = (-1, 1)
        if 'h2_range' not in param_ranges:
            param_ranges['h2_range'] = (-1, 1)
            
    elif model_name.lower() == 'annni':
        model_class = ANNNIModel
        # Set default parameter ranges if not provided
        if 'J1_range' not in param_ranges:
            param_ranges['J1_range'] = (-1, 1)
        if 'J2_range' not in param_ranges:
            param_ranges['J2_range'] = (-1, 1)
        if 'B_range' not in param_ranges:
            param_ranges['B_range'] = (-1, 1)
            
    elif model_name.lower() == 'cluster':
        model_class = ClusterModel
        # Set default parameter ranges if not provided
        if 'J1_range' not in param_ranges:
            param_ranges['J1_range'] = (-1, 1)
        if 'J2_range' not in param_ranges:
            param_ranges['J2_range'] = (-1, 1)
            
    else:
        raise ValueError(f"Unknown model name: {model_name}. Supported models: 'heisenberg', 'haldane', 'annni', 'cluster'")

    # Generate and return the dataset
    print(f"Generating dataset for {model_name} model with {num_qubits} qubits and {num_samples} samples")
    df = dataset_gen.generate_dataset(model_class, param_ranges, num_samples=num_samples)
    
    # Print dataset information
    print(f"Generated dataset with columns: {df.columns.tolist()}")
    print(f"Sample of phases: {df['Phase'].value_counts().to_dict()}")
    
    if 'Ground_State_Energy' in df.columns:
        print(f"Energy statistics: Mean={df['Ground_State_Energy'].mean():.4f}, Min={df['Ground_State_Energy'].min():.4f}")
    
    return df


def generate_heisenberg_dataset(
    num_qubits: int = 3,
    num_samples: int = 10,
    j_range: Tuple[float, float] = (-1, 1),
    vqe_iterations: int = 30,
    ansatz_depth: int = 2
) -> pd.DataFrame:
    """
    Generate a dataset for the Heisenberg XXX model with VQE energy estimates.

    Args:
        num_qubits: Number of qubits in the system
        num_samples: Number of samples to generate
        j_range: Range (min, max) for J coupling constants
        vqe_iterations: Maximum iterations for VQE optimization
        ansatz_depth: Depth of the ansatz circuit for VQE

    Returns:
        DataFrame with Hamiltonian data and energy estimates
    """
    return generate_dataset(
        model_name='heisenberg',
        num_qubits=num_qubits,
        num_samples=num_samples,
        param_ranges={'J_range': j_range},
        vqe_iterations=vqe_iterations,
        ansatz_depth=ansatz_depth
    )


def generate_haldane_dataset(
    num_qubits: int = 3,
    num_samples: int = 10,
    j_range: Tuple[float, float] = (-1, 1),
    h1_range: Tuple[float, float] = (-1, 1),
    h2_range: Tuple[float, float] = (-1, 1),
    vqe_iterations: int = 30,
    ansatz_depth: int = 2
) -> pd.DataFrame:
    """
    Generate a dataset for the Haldane Chain model with VQE energy estimates.

    Args:
        num_qubits: Number of qubits in the system
        num_samples: Number of samples to generate
        j_range: Range (min, max) for J parameter
        h1_range: Range (min, max) for h1 parameter
        h2_range: Range (min, max) for h2 parameter
        vqe_iterations: Maximum iterations for VQE optimization
        ansatz_depth: Depth of the ansatz circuit for VQE

    Returns:
        DataFrame with Hamiltonian data and energy estimates
    """
    return generate_dataset(
        model_name='haldane',
        num_qubits=num_qubits,
        num_samples=num_samples,
        param_ranges={
            'J_range': j_range,
            'h1_range': h1_range,
            'h2_range': h2_range
        },
        vqe_iterations=vqe_iterations,
        ansatz_depth=ansatz_depth
    )


def generate_annni_dataset(
    num_qubits: int = 3,
    num_samples: int = 10,
    j1_range: Tuple[float, float] = (-1, 1),
    j2_range: Tuple[float, float] = (-1, 1),
    b_range: Tuple[float, float] = (-1, 1),
    vqe_iterations: int = 30,
    ansatz_depth: int = 2
) -> pd.DataFrame:
    """
    Generate a dataset for the ANNNI model with VQE energy estimates.

    Args:
        num_qubits: Number of qubits in the system
        num_samples: Number of samples to generate
        j1_range: Range (min, max) for J1 parameter
        j2_range: Range (min, max) for J2 parameter
        b_range: Range (min, max) for B parameter
        vqe_iterations: Maximum iterations for VQE optimization
        ansatz_depth: Depth of the ansatz circuit for VQE

    Returns:
        DataFrame with Hamiltonian data and energy estimates
    """
    return generate_dataset(
        model_name='annni',
        num_qubits=num_qubits,
        num_samples=num_samples,
        param_ranges={
            'J1_range': j1_range,
            'J2_range': j2_range,
            'B_range': b_range
        },
        vqe_iterations=vqe_iterations,
        ansatz_depth=ansatz_depth
    )


def generate_cluster_dataset(
    num_qubits: int = 3,
    num_samples: int = 10,
    j1_range: Tuple[float, float] = (-1, 1),
    j2_range: Tuple[float, float] = (-1, 1),
    vqe_iterations: int = 30,
    ansatz_depth: int = 2
) -> pd.DataFrame:
    """
    Generate a dataset for the Cluster model with VQE energy estimates.

    Args:
        num_qubits: Number of qubits in the system
        num_samples: Number of samples to generate
        j1_range: Range (min, max) for J1 parameter
        j2_range: Range (min, max) for J2 parameter
        vqe_iterations: Maximum iterations for VQE optimization
        ansatz_depth: Depth of the ansatz circuit for VQE

    Returns:
        DataFrame with Hamiltonian data and energy estimates
    """
    return generate_dataset(
        model_name='cluster',
        num_qubits=num_qubits,
        num_samples=num_samples,
        param_ranges={
            'J1_range': j1_range,
            'J2_range': j2_range
        },
        vqe_iterations=vqe_iterations,
        ansatz_depth=ansatz_depth
    )


def save_dataset(df: pd.DataFrame, filename: str) -> None:
    """
    Save a dataset to a CSV file.

    Args:
        df: The DataFrame to save
        filename: The name of the file to save to
    """
    # Convert complex objects to string representation where needed
    processed_df = df.copy()
    
    # Handle Hamiltonian column (convert to string representation)
    if 'Hamiltonian' in processed_df.columns:
        processed_df['Hamiltonian'] = processed_df['Hamiltonian'].apply(str)
    
    # Handle list columns (convert to string representation)
    for col in processed_df.columns:
        if processed_df[col].apply(lambda x: isinstance(x, list)).any():
            processed_df[col] = processed_df[col].apply(str)
    
    # Save to CSV
    processed_df.to_csv(filename, index=False)
    print(f"Dataset saved to {filename}")


def run_example():
    """Run examples of dataset generation for each model type."""
    print("Running example dataset generation...")

    # Generate small example datasets
    heisenberg_df = generate_heisenberg_dataset(num_qubits=2, num_samples=2)
    haldane_df = generate_haldane_dataset(num_qubits=2, num_samples=2)
    annni_df = generate_annni_dataset(num_qubits=2, num_samples=2)
    cluster_df = generate_cluster_dataset(num_qubits=2, num_samples=2)

    # Save datasets to CSV files
    save_dataset(heisenberg_df, 'heisenberg_dataset.csv')
    save_dataset(haldane_df, 'haldane_dataset.csv')
    save_dataset(annni_df, 'annni_dataset.csv')
    save_dataset(cluster_df, 'cluster_dataset.csv')

    print("Example datasets generated and saved to CSV files.")


if __name__ == "__main__":
    run_example()
