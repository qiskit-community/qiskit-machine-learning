from qiskit import QuantumCircuit
from qiskit.opflow import StateFn, PauliSumOp, AerPauliExpectation, Gradient
from qiskit.utils import QuantumInstance
from qiskit_machine_learning.neural_networks import OpflowQNN
from qiskit_machine_learning.circuit.library import RawFeatureVector

# expval = AerPauliExpectation()
# gradient = Gradient()
# qi_sv = QuantumInstance(Aer.get_backend('statevector_simulator'))
#
# inputs = RawFeatureVector(16)
# qc = QuantumCircuit(4)
# qc.append(inputs,range(4))
# qc_sfn = StateFn(qc)
# H1 = StateFn(PauliSumOp.from_list([('Z', 1.0)]))
# op = ~H1 @ qc_sfn
# qnn4 = OpflowQNN(operator = op, input_params = inputs.parameter, weight_params = [], exp_val = expval, gradient = gradient, quantum_instance = qi_sv)
# # --------------------------------------------
from qiskit import Aer
import numpy as np
from qiskit.circuit import Parameter

expval = AerPauliExpectation()
gradient = Gradient()
qi_sv = QuantumInstance(Aer.get_backend('statevector_simulator'))

inputs = RawFeatureVector(16)
qc = QuantumCircuit(4)
qc.append(inputs,range(4))
# qc = inputs
qc_sfn = StateFn(qc)
H1 = StateFn(PauliSumOp.from_list([('Z', 1.0)]))
op = ~H1 @ qc_sfn
print(op)
qnn4 = OpflowQNN(operator = op, input_params = inputs.parameters[:12], weight_params = inputs.parameters[12:], exp_val = expval,
                 gradient = gradient, quantum_instance = qi_sv)

# params1 = [Parameter('input1'), Parameter('weight1')]
# qc1 = QuantumCircuit(1)
# qc1.h(0)
# qc1.ry(params1[0], 0)
# qc1.rx(params1[1], 0)
# qc_sfn1 = StateFn(qc1)
# # construct cost operator
# H1 = StateFn(PauliSumOp.from_list([('Z', 1.0), ('X', 1.0)]))
# # combine operator and circuit to objective function
# op1 = ~H1 @ qc_sfn1
# print(op1)
# qnn4 = OpflowQNN(op1, [params1[0]], [params1[1]], expval, gradient, qi_sv)

# define (random) input and weights
input1 = np.random.rand(qnn4.num_inputs)
weights1 = np.random.rand(qnn4.num_weights)

# QNN forward pass
fwd_out = qnn4.forward(input1, weights1)

# QNN backward pass
bckwd_out = qnn4.backward(input1, weights1)