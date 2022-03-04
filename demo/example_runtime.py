from qiskit.circuit.library import ZFeatureMap, RealAmplitudes
from qiskit import Aer

# This is an example file to create a quantum model and compute its effective dimension

# create ranges for the number of data, n
n = [5000, 8000, 10000, 40000, 60000, 100000, 150000, 200000, 500000, 1000000]

# number of times to repeat the variational circuit
blocks = 1

# number of qubits, data samples and parameter sets to estimate the effective dimension
num_qubits = 3
num_inputs = 10
num_thetas = 10

# create a feature map
fm = ZFeatureMap(feature_dimension=num_qubits, reps=1)

# create a variational circuit
vf = RealAmplitudes(num_qubits, reps=blocks)

from qiskit.utils import QuantumInstance

# declare quantum instance
qi = QuantumInstance(Aer.get_backend('aer_simulator_statevector'))

from qiskit_machine_learning.runtime.ed_runtime_client import EffDimRuntimeClient

backend = Aer.get_backend('qasm_simulator')

eff_dim = EffDimRuntimeClient(
    feat_map=fm,
    ansatz=vf,
    mock_runtime=True,
    backend=backend
)

out = eff_dim.compute_eff_dim(n, num_inputs=num_inputs, num_thetas=num_thetas)



