# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2019, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
# =============================================================================

"""Test Estimator Gradients"""

import unittest
from test import QiskitAlgorithmsTestCase

import numpy as np
from ddt import ddt, data

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import EfficientSU2, RealAmplitudes
from qiskit.circuit.library.standard_gates import RXXGate, RYYGate, RZXGate, RZZGate
from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import Session, EstimatorV2
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_machine_learning.gradients import (
    LinCombEstimatorGradient,
    ParamShiftEstimatorGradient,
    SPSAEstimatorGradient,
)

from .logging_primitives import LoggingEstimator

gradient_factories = [
    ParamShiftEstimatorGradient,
    LinCombEstimatorGradient,
]


@ddt
class TestEstimatorGradient(QiskitAlgorithmsTestCase):
    """Test Estimator Gradient"""
    def __init__(self, TestCase):
        self.estimator = Estimator()
        super().__init__(TestCase)
    @data(*gradient_factories)
    def test_gradient_operators(self, grad):
        """Test the estimator gradient for different operators"""
        a = Parameter("a")
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.p(a, 0)
        qc.h(0)
        gradient = grad(self.estimator)
        op = SparsePauliOp.from_list([("Z", 1)])
        correct_result = -1 / np.sqrt(2)
        param = [np.pi / 4]
        value = gradient.run([qc], [op], [param]).result().gradients[0]
        self.assertAlmostEqual(value[0], correct_result, 3)
        op = SparsePauliOp.from_list([("Z", 1)])
        value = gradient.run([qc], [op], [param]).result().gradients[0]
        self.assertAlmostEqual(value[0], correct_result, 3)

    @data(*gradient_factories)
    def test_single_circuit_observable(self, grad):
        """Test the estimator gradient for a single circuit and observable"""
        
        a = Parameter("a")
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.p(a, 0)
        qc.h(0)
        gradient = grad(self.estimator)
        op = SparsePauliOp.from_list([("Z", 1)])
        correct_result = -1 / np.sqrt(2)
        param = [np.pi / 4]
        value = gradient.run(qc, op, [param]).result().gradients[0]
        self.assertAlmostEqual(value[0], correct_result, 3)

    @data(*gradient_factories)
    def test_gradient_p(self, grad):
        """Test the estimator gradient for p"""
        
        a = Parameter("a")
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.p(a, 0)
        qc.h(0)
        gradient = grad(self.estimator)
        op = SparsePauliOp.from_list([("Z", 1)])
        param_list = [[np.pi / 4], [0], [np.pi / 2]]
        correct_results = [[-1 / np.sqrt(2)], [0], [-1]]
        for i, param in enumerate(param_list):
            gradients = gradient.run([qc], [op], [param]).result().gradients[0]
            for j, value in enumerate(gradients):
                self.assertAlmostEqual(value, correct_results[i][j], 3)

    @data(*gradient_factories)
    def test_gradient_u(self, grad):
        """Test the estimator gradient for u"""
        
        a = Parameter("a")
        b = Parameter("b")
        c = Parameter("c")
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.u(a, b, c, 0)
        qc.h(0)
        gradient = grad(self.estimator)
        op = SparsePauliOp.from_list([("Z", 1)])

        param_list = [[np.pi / 4, 0, 0], [np.pi / 4, np.pi / 4, np.pi / 4]]
        correct_results = [[-0.70710678, 0.0, 0.0], [-0.35355339, -0.85355339, -0.85355339]]
        for i, param in enumerate(param_list):
            gradients = gradient.run([qc], [op], [param]).result().gradients[0]
            for j, value in enumerate(gradients):
                self.assertAlmostEqual(value, correct_results[i][j], 3)

    @data(*gradient_factories)
    def test_gradient_efficient_su2(self, grad):
        """Test the estimator gradient for EfficientSU2"""
        
        qc = EfficientSU2(2, reps=1)
        op = SparsePauliOp.from_list([("ZI", 1)])
        gradient = grad(self.estimator)
        param_list = [
            [np.pi / 4 for param in qc.parameters],
            [np.pi / 2 for param in qc.parameters],
        ]
        correct_results = [
            [
                -0.35355339,
                -0.70710678,
                0,
                0.35355339,
                0,
                -0.70710678,
                0,
                0,
            ],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ]
        for i, param in enumerate(param_list):
            gradients = gradient.run([qc], [op], [param]).result().gradients[0]
            np.testing.assert_allclose(gradients, correct_results[i], atol=1e-3)

    @data(*gradient_factories)
    def test_gradient_2qubit_gate(self, grad):
        """Test the estimator gradient for 2 qubit gates"""
        
        for gate in [RXXGate, RYYGate, RZZGate, RZXGate]:
            param_list = [[np.pi / 4], [np.pi / 2]]
            correct_results = [
                [-0.70710678],
                [-1],
            ]
            op = SparsePauliOp.from_list([("ZI", 1)])
            for i, param in enumerate(param_list):
                a = Parameter("a")
                qc = QuantumCircuit(2)
                gradient = grad(self.estimator)

                if gate is RZZGate:
                    qc.h([0, 1])
                    qc.append(gate(a), [qc.qubits[0], qc.qubits[1]], [])
                    qc.h([0, 1])
                else:
                    qc.append(gate(a), [qc.qubits[0], qc.qubits[1]], [])
                gradients = gradient.run([qc], [op], [param]).result().gradients[0]
                np.testing.assert_allclose(gradients, correct_results[i], atol=1e-3)

    @data(*gradient_factories)
    def test_gradient_parameter_coefficient(self, grad):
        """Test the estimator gradient for parameter variables with coefficients"""
        
        qc = RealAmplitudes(num_qubits=2, reps=1)
        qc.rz(qc.parameters[0].exp() + 2 * qc.parameters[1], 0)
        qc.rx(3.0 * qc.parameters[0] + qc.parameters[1].sin(), 1)
        qc.u(qc.parameters[0], qc.parameters[1], qc.parameters[3], 1)
        qc.p(2 * qc.parameters[0] + 1, 0)
        qc.rxx(qc.parameters[0] + 2, 0, 1)
        gradient = grad(self.estimator)
        param_list = [[np.pi / 4 for _ in qc.parameters], [np.pi / 2 for _ in qc.parameters]]
        correct_results = [
            [-0.7266653, -0.4905135, -0.0068606, -0.9228880],
            [-3.5972095, 0.10237173, -0.3117748, 0],
        ]
        op = SparsePauliOp.from_list([("ZI", 1)])
        for i, param in enumerate(param_list):
            gradients = gradient.run([qc], [op], [param]).result().gradients[0]
            np.testing.assert_allclose(gradients, correct_results[i], atol=1e-3)

    @data(*gradient_factories)
    def test_gradient_parameters(self, grad):
        """Test the estimator gradient for parameters"""
        
        a = Parameter("a")
        b = Parameter("b")
        qc = QuantumCircuit(1)
        qc.rx(a, 0)
        qc.rx(b, 0)
        gradient = grad(self.estimator)
        param_list = [[np.pi / 4, np.pi / 2]]
        correct_results = [
            [-0.70710678],
        ]
        op = SparsePauliOp.from_list([("Z", 1)])
        for i, param in enumerate(param_list):
            gradients = gradient.run([qc], [op], [param], parameters=[[a]]).result().gradients[0]
            np.testing.assert_allclose(gradients, correct_results[i], atol=1e-3)

        # parameter order
        with self.subTest(msg="The order of gradients"):
            c = Parameter("c")
            qc = QuantumCircuit(1)
            qc.rx(a, 0)
            qc.rz(b, 0)
            qc.rx(c, 0)

            param_list = [[np.pi / 4, np.pi / 2, np.pi / 3]]
            correct_results = [
                [-0.35355339, 0.61237244, -0.61237244],
                [-0.61237244, 0.61237244, -0.35355339],
                [-0.35355339, -0.61237244],
                [-0.61237244, -0.35355339],
            ]
            param = [[a, b, c], [c, b, a], [a, c], [c, a]]
            op = SparsePauliOp.from_list([("Z", 1)])
            for i, p in enumerate(param):  # pylint: disable=invalid-name
                gradient = grad(self.estimator)
                gradients = (
                    gradient.run([qc], [op], param_list, parameters=[p]).result().gradients[0]
                )
                np.testing.assert_allclose(gradients, correct_results[i], atol=1e-3)

    @data(*gradient_factories)
    def test_gradient_multi_arguments(self, grad):
        """Test the estimator gradient for multiple arguments"""
        
        a = Parameter("a")
        b = Parameter("b")
        qc = QuantumCircuit(1)
        qc.rx(a, 0)
        qc2 = QuantumCircuit(1)
        qc2.rx(b, 0)
        gradient = grad(self.estimator)
        param_list = [[np.pi / 4], [np.pi / 2]]
        correct_results = [
            [-0.70710678],
            [-1],
        ]
        op = SparsePauliOp.from_list([("Z", 1)])
        gradients = gradient.run([qc, qc2], [op] * 2, param_list).result().gradients
        np.testing.assert_allclose(gradients, correct_results, atol=1e-3)

        c = Parameter("c")
        qc3 = QuantumCircuit(1)
        qc3.rx(c, 0)
        qc3.ry(a, 0)
        param_list2 = [[np.pi / 4], [np.pi / 4, np.pi / 4], [np.pi / 4, np.pi / 4]]
        correct_results2 = [
            [-0.70710678],
            [-0.5],
            [-0.5, -0.5],
        ]
        gradients2 = (
            gradient.run([qc, qc3, qc3], [op] * 3, param_list2, parameters=[[a], [c], None])
            .result()
            .gradients
        )
        np.testing.assert_allclose(gradients2[0], correct_results2[0], atol=1e-3)
        np.testing.assert_allclose(gradients2[1], correct_results2[1], atol=1e-3)
        np.testing.assert_allclose(gradients2[2], correct_results2[2], atol=1e-3)

    @data(*gradient_factories)
    def test_gradient_validation(self, grad):
        """Test estimator gradient's validation"""
        
        a = Parameter("a")
        qc = QuantumCircuit(1)
        qc.rx(a, 0)
        gradient = grad(self.estimator)
        param_list = [[np.pi / 4], [np.pi / 2]]
        op = SparsePauliOp.from_list([("Z", 1)])
        with self.assertRaises(ValueError):
            gradient.run([qc], [op], param_list)
        with self.assertRaises(ValueError):
            gradient.run([qc, qc], [op, op], param_list, parameters=[[a]])
        with self.assertRaises(ValueError):
            gradient.run([qc, qc], [op], param_list, parameters=[[a]])
        with self.assertRaises(ValueError):
            gradient.run([qc], [op], [[np.pi / 4, np.pi / 4]])

    def test_spsa_gradient(self):
        """Test the SPSA estimator gradient"""
        
        with self.assertRaises(ValueError):
            _ = SPSAEstimatorGradient(self.estimator, epsilon=-0.1)
        a = Parameter("a")
        b = Parameter("b")
        qc = QuantumCircuit(2)
        qc.rx(b, 0)
        qc.rx(a, 1)
        param_list = [[1, 1]]
        correct_results = [[-0.84147098, 0.84147098]]
        op = SparsePauliOp.from_list([("ZI", 1)])
        gradient = SPSAEstimatorGradient(self.estimator, epsilon=1e-6, seed=123)
        gradients = gradient.run([qc], [op], param_list).result().gradients
        np.testing.assert_allclose(gradients, correct_results, atol=1e-3)

        # multi parameters
        with self.subTest(msg="Multiple parameters"):
            gradient = SPSAEstimatorGradient(self.estimator, epsilon=1e-6, seed=123)
            param_list2 = [[1, 1], [1, 1], [3, 3]]
            gradients2 = (
                gradient.run([qc] * 3, [op] * 3, param_list2, parameters=[None, [b], None])
                .result()
                .gradients
            )
            correct_results2 = [[-0.84147098, 0.84147098], [0.84147098], [-0.14112001, 0.14112001]]
            for grad, correct in zip(gradients2, correct_results2):
                np.testing.assert_allclose(grad, correct, atol=1e-3)

        # batch size
        with self.subTest(msg="Batch size"):
            correct_results = [[-0.84147098, 0.1682942]]
            gradient = SPSAEstimatorGradient(self.estimator, epsilon=1e-6, batch_size=5, seed=123)
            gradients = gradient.run([qc], [op], param_list).result().gradients
            np.testing.assert_allclose(gradients, correct_results, atol=1e-3)

        # parameter order
        with self.subTest(msg="The order of gradients"):
            gradient = SPSAEstimatorGradient(self.estimator, epsilon=1e-6, seed=123)
            c = Parameter("c")
            qc = QuantumCircuit(1)
            qc.rx(a, 0)
            qc.rz(b, 0)
            qc.rx(c, 0)
            op = SparsePauliOp.from_list([("Z", 1)])
            param_list3 = [[np.pi / 4, np.pi / 2, np.pi / 3]]
            param = [[a, b, c], [c, b, a], [a, c], [c, a]]
            expected = [
                [-0.3535525, 0.3535525, 0.3535525],
                [0.3535525, 0.3535525, -0.3535525],
                [-0.3535525, 0.3535525],
                [0.3535525, -0.3535525],
            ]
            for i, p in enumerate(param):  # pylint: disable=invalid-name
                gradient = SPSAEstimatorGradient(self.estimator, epsilon=1e-6, seed=123)
                gradients = (
                    gradient.run([qc], [op], param_list3, parameters=[p]).result().gradients[0]
                )
                np.testing.assert_allclose(gradients, expected[i], atol=1e-3)

    @data(
        ParamShiftEstimatorGradient,
        LinCombEstimatorGradient,
        SPSAEstimatorGradient,
    )
    def test_options(self, grad):
        """Test estimator gradient's run options"""
        a = Parameter("a")
        qc = QuantumCircuit(1)
        qc.rx(a, 0)
        op = SparsePauliOp.from_list([("Z", 1)])
        estimator = Estimator(options={"shots": 100})
        with self.subTest("estimator"):
            if grad is SPSAEstimatorGradient:
                gradient = grad(estimator, epsilon=1e-6)
            else:
                gradient = grad(estimator)
            options = gradient.options
            result = gradient.run([qc], [op], [[1]]).result()
            self.assertEqual(result.options.get("shots"), 100)
            self.assertEqual(options.get("shots"), 100)

        with self.subTest("gradient init"):
            if grad is SPSAEstimatorGradient:
                gradient = grad(estimator, epsilon=1e-6, options={"shots": 200})
            else:
                gradient = grad(estimator, options={"shots": 200})
            options = gradient.options
            result = gradient.run([qc], [op], [[1]]).result()
            self.assertEqual(result.options.get("shots"), 200)
            self.assertEqual(options.get("shots"), 200)

        with self.subTest("gradient update"):
            if grad is SPSAEstimatorGradient:
                gradient = grad(estimator, epsilon=1e-6, options={"shots": 200})
            else:
                gradient = grad(estimator, options={"shots": 200})
            gradient.update_default_options(shots=100)
            options = gradient.options
            result = gradient.run([qc], [op], [[1]]).result()
            self.assertEqual(result.options.get("shots"), 100)
            self.assertEqual(options.get("shots"), 100)

        with self.subTest("gradient run"):
            if grad is SPSAEstimatorGradient:
                gradient = grad(estimator, epsilon=1e-6, options={"shots": 200})
            else:
                gradient = grad(estimator, options={"shots": 200})
            options = gradient.options
            result = gradient.run([qc], [op], [[1]], shots=300).result()
            self.assertEqual(result.options.get("shots"), 300)
            # Only default + estimator options. Not run.
            self.assertEqual(options.get("shots"), 200)

    @data(
        ParamShiftEstimatorGradient,
        LinCombEstimatorGradient,
        SPSAEstimatorGradient,
    )
    def test_operations_preserved(self, gradient_cls):
        """Test non-parameterized instructions are preserved and not unrolled."""
        x = Parameter("x")
        circuit = QuantumCircuit(2)
        circuit.initialize([0.5, 0.5, 0.5, 0.5])  # this should remain as initialize
        circuit.crx(x, 0, 1)  # this should get unrolled

        values = [np.pi / 2]
        expect = -1 / (2 * np.sqrt(2))

        observable = SparsePauliOp(["XX"])

        ops = []

        def operations_callback(op):
            ops.append(op)

        estimator = LoggingEstimator(operations_callback=operations_callback)

        if gradient_cls in [SPSAEstimatorGradient]:
            gradient = gradient_cls(self.estimator, epsilon=0.01)
        else:
            gradient = gradient_cls(self.estimator)

        job = gradient.run([circuit], [observable], [values])
        result = job.result()

        with self.subTest(msg="assert initialize is preserved"):
            self.assertTrue(all("initialize" in ops_i[0].keys() for ops_i in ops))

        with self.subTest(msg="assert result is correct"):
            self.assertAlmostEqual(result.gradients[0].item(), expect, places=5)

@ddt
class TestEstimatorGradientV2(QiskitAlgorithmsTestCase):
    """Test Estimator Gradient"""
    def __init__(self, TestCase):
        backend = GenericBackendV2(num_qubits=3, seed=123)
        session = Session(backend=backend)
        self.estimator = EstimatorV2(mode=session)
        self.pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
        super().__init__(TestCase)
    @data(*gradient_factories)
    def test_gradient_operators(self, grad):
        """Test the estimator gradient for different operators"""
        
        a = Parameter("a")
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.p(a, 0)
        qc.h(0)
        gradient = grad(self.estimator, self.pm)
        op = SparsePauliOp.from_list([("Z", 1)])
        correct_result = -1 / np.sqrt(2)
        param = [np.pi / 4]
        value = gradient.run([qc], [op], [param]).result().gradients[0]
        self.assertAlmostEqual(value[0], correct_result, 1)
        op = SparsePauliOp.from_list([("Z", 1)])
        value = gradient.run([qc], [op], [param]).result().gradients[0]
        self.assertAlmostEqual(value[0], correct_result, 1)

    @data(*gradient_factories)
    def test_single_circuit_observable(self, grad):
        """Test the estimator gradient for a single circuit and observable"""
        
        a = Parameter("a")
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.p(a, 0)
        qc.h(0)
        gradient = grad(estimator=self.estimator, pass_manager=self.pm)
        op = SparsePauliOp.from_list([("Z", 1)])
        correct_result = -1 / np.sqrt(2)
        param = [np.pi / 4]
        value = gradient.run(qc, op, [param]).result().gradients[0]
        self.assertAlmostEqual(value[0], correct_result, 1)

    @data(*gradient_factories)
    def test_gradient_p(self, grad):
        """Test the estimator gradient for p"""
        
        a = Parameter("a")
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.p(a, 0)
        qc.h(0)
        gradient = grad(estimator=self.estimator, pass_manager=self.pm)
        op = SparsePauliOp.from_list([("Z", 1)])
        param_list = [[np.pi / 4], [0], [np.pi / 2]]
        correct_results = [[-1 / np.sqrt(2)], [0], [-1]]
        for i, param in enumerate(param_list):
            gradients = gradient.run([qc], [op], [param]).result().gradients[0]
            for j, value in enumerate(gradients):
                self.assertAlmostEqual(value, correct_results[i][j], 1)

    @data(*gradient_factories)
    def test_gradient_u(self, grad):
        """Test the estimator gradient for u"""
        
        a = Parameter("a")
        b = Parameter("b")
        c = Parameter("c")
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.u(a, b, c, 0)
        qc.h(0)
        gradient = grad(estimator=self.estimator, pass_manager=self.pm)
        op = SparsePauliOp.from_list([("Z", 1)])

        param_list = [[np.pi / 4, 0, 0], [np.pi / 4, np.pi / 4, np.pi / 4]]
        correct_results = [[-0.70710678, 0.0, 0.0], [-0.35355339, -0.85355339, -0.85355339]]
        for i, param in enumerate(param_list):
            gradients = gradient.run([qc], [op], [param]).result().gradients[0]
            for j, value in enumerate(gradients):
                self.assertAlmostEqual(value, correct_results[i][j], 1)

    @data(*gradient_factories)
    def test_gradient_efficient_su2(self, grad):
        """Test the estimator gradient for EfficientSU2"""
        
        qc = EfficientSU2(2, reps=1)
        op = SparsePauliOp.from_list([("ZI", 1)])
        gradient = grad(estimator=self.estimator, pass_manager=self.pm)
        param_list = [
            [np.pi / 4 for param in qc.parameters],
            [np.pi / 2 for param in qc.parameters],
        ]
        correct_results = [
            [
                -0.35355339,
                -0.70710678,
                0,
                0.35355339,
                0,
                -0.70710678,
                0,
                0,
            ],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ]
        for i, param in enumerate(param_list):
            gradients = gradient.run([qc], [op], [param]).result().gradients[0]
            np.testing.assert_allclose(gradients, correct_results[i], atol=1e-1, rtol=1e-1)

    @data(*gradient_factories)
    def test_gradient_2qubit_gate(self, grad):
        """Test the estimator gradient for 2 qubit gates"""
        
        for gate in [RXXGate, RYYGate, RZZGate, RZXGate]:
            param_list = [[np.pi / 4], [np.pi / 2]]
            correct_results = [
                [-0.70710678],
                [-1],
            ]
            op = SparsePauliOp.from_list([("ZI", 1)])
            for i, param in enumerate(param_list):
                a = Parameter("a")
                qc = QuantumCircuit(2)
                gradient = grad(estimator=self.estimator, pass_manager=self.pm)

                if gate is RZZGate:
                    qc.h([0, 1])
                    qc.append(gate(a), [qc.qubits[0], qc.qubits[1]], [])
                    qc.h([0, 1])
                else:
                    qc.append(gate(a), [qc.qubits[0], qc.qubits[1]], [])
                gradients = gradient.run([qc], [op], [param]).result().gradients[0]
                np.testing.assert_allclose(gradients, correct_results[i], atol=1e-1, rtol=1e-1)

    @data(*gradient_factories)
    def test_gradient_parameter_coefficient(self, grad):
        """Test the estimator gradient for parameter variables with coefficients"""
        
        qc = RealAmplitudes(num_qubits=2, reps=1)
        qc.rz(qc.parameters[0].exp() + 2 * qc.parameters[1], 0)
        qc.rx(3.0 * qc.parameters[0] + qc.parameters[1].sin(), 1)
        qc.u(qc.parameters[0], qc.parameters[1], qc.parameters[3], 1)
        qc.p(2 * qc.parameters[0] + 1, 0)
        qc.rxx(qc.parameters[0] + 2, 0, 1)
        gradient = grad(estimator=self.estimator, pass_manager=self.pm)
        param_list = [[np.pi / 4 for _ in qc.parameters], [np.pi / 2 for _ in qc.parameters]]
        correct_results = [
            [-0.7266653, -0.4905135, -0.0068606, -0.9228880],
            [-3.5972095, 0.10237173, -0.3117748, 0],
        ]
        op = SparsePauliOp.from_list([("ZI", 1)])
        for i, param in enumerate(param_list):
            gradients = gradient.run([qc], [op], [param]).result().gradients[0]
            np.testing.assert_allclose(gradients, correct_results[i], atol=1e-1, rtol=1e-1)

    @data(*gradient_factories)
    def test_gradient_parameters(self, grad):
        """Test the estimator gradient for parameters"""
        
        a = Parameter("a")
        b = Parameter("b")
        qc = QuantumCircuit(1)
        qc.rx(a, 0)
        qc.rx(b, 0)
        gradient = grad(estimator=self.estimator, pass_manager=self.pm)
        param_list = [[np.pi / 4, np.pi / 2]]
        correct_results = [
            [-0.70710678],
        ]
        op = SparsePauliOp.from_list([("Z", 1)])
        for i, param in enumerate(param_list):
            gradients = gradient.run([qc], [op], [param], parameters=[[a]]).result().gradients[0]
            np.testing.assert_allclose(gradients, correct_results[i], atol=1e-1, rtol=1e-1)

        # parameter order
        with self.subTest(msg="The order of gradients"):
            c = Parameter("c")
            qc = QuantumCircuit(1)
            qc.rx(a, 0)
            qc.rz(b, 0)
            qc.rx(c, 0)

            param_list = [[np.pi / 4, np.pi / 2, np.pi / 3]]
            correct_results = [
                [-0.35355339, 0.61237244, -0.61237244],
                [-0.61237244, 0.61237244, -0.35355339],
                [-0.35355339, -0.61237244],
                [-0.61237244, -0.35355339],
            ]
            param = [[a, b, c], [c, b, a], [a, c], [c, a]]
            op = SparsePauliOp.from_list([("Z", 1)])
            for i, p in enumerate(param):  # pylint: disable=invalid-name
                gradient = grad(estimator=self.estimator, pass_manager=self.pm)
                gradients = (
                    gradient.run([qc], [op], param_list, parameters=[p]).result().gradients[0]
                )
                np.testing.assert_allclose(gradients, correct_results[i], atol=1e-1, rtol=1e-1)

    @data(*gradient_factories)
    def test_gradient_multi_arguments(self, grad):
        """Test the estimator gradient for multiple arguments"""
        
        a = Parameter("a")
        b = Parameter("b")
        qc = QuantumCircuit(1)
        qc.rx(a, 0)
        qc2 = QuantumCircuit(1)
        qc2.rx(b, 0)
        gradient = grad(estimator=self.estimator, pass_manager=self.pm)
        param_list = [[np.pi / 4], [np.pi / 2]]
        correct_results = [
            [-0.70710678],
            [-1],
        ]
        op = SparsePauliOp.from_list([("Z", 1)])
        gradients = gradient.run([qc, qc2], [op] * 2, param_list).result().gradients
        np.testing.assert_allclose(gradients, correct_results, atol=1e-1, rtol=1e-1)

        c = Parameter("c")
        qc3 = QuantumCircuit(1)
        qc3.rx(c, 0)
        qc3.ry(a, 0)
        param_list2 = [[np.pi / 4], [np.pi / 4, np.pi / 4], [np.pi / 4, np.pi / 4]]
        correct_results2 = [
            [-0.70710678],
            [-0.5],
            [-0.5, -0.5],
        ]
        gradients2 = (
            gradient.run([qc, qc3, qc3], [op] * 3, param_list2, parameters=[[a], [c], None])
            .result()
            .gradients
        )
        np.testing.assert_allclose(gradients2[0], correct_results2[0], atol=1e-1, rtol=1e-1)
        np.testing.assert_allclose(gradients2[1], correct_results2[1], atol=1e-1, rtol=1e-1)
        np.testing.assert_allclose(gradients2[2], correct_results2[2], atol=1e-1, rtol=1e-1)

    @data(*gradient_factories)
    def test_gradient_validation(self, grad):
        """Test estimator gradient's validation"""
        
        a = Parameter("a")
        qc = QuantumCircuit(1)
        qc.rx(a, 0)
        gradient = grad(estimator=self.estimator, pass_manager=self.pm)
        param_list = [[np.pi / 4], [np.pi / 2]]
        op = SparsePauliOp.from_list([("Z", 1)])
        with self.assertRaises(ValueError):
            gradient.run([qc], [op], param_list)
        with self.assertRaises(ValueError):
            gradient.run([qc, qc], [op, op], param_list, parameters=[[a]])
        with self.assertRaises(ValueError):
            gradient.run([qc, qc], [op], param_list, parameters=[[a]])
        with self.assertRaises(ValueError):
            gradient.run([qc], [op], [[np.pi / 4, np.pi / 4]])
    @unittest.skip("Skipping due to noise.")
    def test_spsa_gradient(self):
        """Test the SPSA estimator gradient"""
        
        with self.assertRaises(ValueError):
            _ = SPSAEstimatorGradient(estimator=self.estimator, pass_manager=self.pm, epsilon=-0.1)
        a = Parameter("a")
        b = Parameter("b")
        qc = QuantumCircuit(2)
        qc.rx(b, 0)
        qc.rx(a, 1)
        param_list = [[1, 1]]
        correct_results = [[-0.84147098, 0.84147098]]
        op = SparsePauliOp.from_list([("ZI", 1)])
        gradient = SPSAEstimatorGradient(estimator=self.estimator, pass_manager=self.pm, epsilon=1e-6, seed=123)
        gradients = gradient.run([qc], [op], param_list).result().gradients
        np.testing.assert_allclose(gradients, correct_results, atol=1e-1, rtol=1e-1)

        # multi parameters
        with self.subTest(msg="Multiple parameters"):
            gradient = SPSAEstimatorGradient(estimator=self.estimator, pass_manager=self.pm, epsilon=1e-6, seed=123)
            param_list2 = [[1, 1], [1, 1], [3, 3]]
            gradients2 = (
                gradient.run([qc] * 3, [op] * 3, param_list2, parameters=[None, [b], None])
                .result()
                .gradients
            )
            correct_results2 = [[-0.84147098, 0.84147098], [0.84147098], [-0.14112001, 0.14112001]]
            for grad, correct in zip(gradients2, correct_results2):
                np.testing.assert_allclose(grad, correct, atol=1e-1, rtol=1e-1)

        # batch size
        with self.subTest(msg="Batch size"):
            correct_results = [[-0.84147098, 0.1682942]]
            gradient = SPSAEstimatorGradient(estimator=self.estimator, pass_manager=self.pm, epsilon=1e-6, batch_size=5, seed=123)
            gradients = gradient.run([qc], [op], param_list).result().gradients
            np.testing.assert_allclose(gradients, correct_results, atol=1e-1, rtol=1e-1)

        # parameter order
        with self.subTest(msg="The order of gradients"):
            gradient = SPSAEstimatorGradient(estimator=self.estimator, pass_manager=self.pm, epsilon=1e-6, seed=123)
            c = Parameter("c")
            qc = QuantumCircuit(1)
            qc.rx(a, 0)
            qc.rz(b, 0)
            qc.rx(c, 0)
            op = SparsePauliOp.from_list([("Z", 1)])
            param_list3 = [[np.pi / 4, np.pi / 2, np.pi / 3]]
            expected = [
                [-0.3535525, 0.3535525, 0.3535525],
                [0.3535525, 0.3535525, -0.3535525],
                [-0.3535525, 0.3535525],
                [0.3535525, -0.3535525],
            ]
            for i, p in enumerate(param):  # pylint: disable=invalid-name
                gradient = SPSAEstimatorGradient(estimator=self.estimator, pass_manager=self.pm, epsilon=1e-6, seed=123)
                gradients = (
                    gradient.run([qc], [op], param_list3, parameters=[p]).result().gradients[0]
                )
                np.testing.assert_allclose(gradients, expected[i], atol=1e-1, rtol=1e-1)

    @data(
        ParamShiftEstimatorGradient,
        LinCombEstimatorGradient,
        SPSAEstimatorGradient,
    )
    @unittest.skip("Options needs to be added for V2.")
    def test_options(self, grad):
        """Test estimator gradient's run options"""
        a = Parameter("a")
        qc = QuantumCircuit(1)
        qc.rx(a, 0)
        op = SparsePauliOp.from_list([("Z", 1)])
        estimator = Estimator(options={"shots": 100})
        with self.subTest("estimator"):
            if grad is SPSAEstimatorGradient:
                gradient = grad(estimator=self.estimator, pass_manager=self.pm, epsilon=1e-6)
            else:
                gradient = grad(estimator=self.estimator, pass_manager=self.pm)
            options = gradient.options
            result = gradient.run([qc], [op], [[1]]).result()
            self.assertEqual(result.options.get("shots"), 100)
            self.assertEqual(options.get("shots"), 100)

        with self.subTest("gradient init"):
            if grad is SPSAEstimatorGradient:
                gradient = grad(estimator=self.estimator, pass_manager=self.pm, epsilon=1e-6, options={"shots": 200})
            else:
                gradient = grad(estimator=self.estimator, pass_manager=self.pm, options={"shots": 200})
            options = gradient.options
            result = gradient.run([qc], [op], [[1]]).result()
            self.assertEqual(result.options.get("shots"), 200)
            self.assertEqual(options.get("shots"), 200)

        with self.subTest("gradient update"):
            if grad is SPSAEstimatorGradient:
                gradient = grad(estimator=self.estimator, pass_manager=self.pm, epsilon=1e-6, options={"shots": 200})
            else:
                gradient = grad(estimator=self.estimator, pass_manager=self.pm, options={"shots": 200})
            gradient.update_default_options(shots=100)
            options = gradient.options
            result = gradient.run([qc], [op], [[1]]).result()
            self.assertEqual(result.options.get("shots"), 100)
            self.assertEqual(options.get("shots"), 100)

        with self.subTest("gradient run"):
            if grad is SPSAEstimatorGradient:
                gradient = grad(estimator=self.estimator, pass_manager=self.pm, epsilon=1e-6, options={"shots": 200})
            else:
                gradient = grad(estimator=self.estimator, pass_manager=self.pm, options={"shots": 200})
            options = gradient.options
            result = gradient.run([qc], [op], [[1]], shots=300).result()
            self.assertEqual(result.options.get("shots"), 300)
            # Only default + estimator options. Not run.
            self.assertEqual(options.get("shots"), 200)

    @data(
        ParamShiftEstimatorGradient,
        LinCombEstimatorGradient,
        SPSAEstimatorGradient,
    )
    def test_operations_preserved(self, gradient_cls):
        """Test non-parameterized instructions are preserved and not unrolled."""
        x = Parameter("x")
        circuit = QuantumCircuit(2)
        circuit.initialize([0.5, 0.5, 0.5, 0.5])  # this should remain as initialize
        circuit.crx(x, 0, 1)  # this should get unrolled

        values = [np.pi / 2]
        expect = -1 / (2 * np.sqrt(2))

        observable = SparsePauliOp(["XX"])

        ops = []

        def operations_callback(op):
            ops.append(op)

        estimator = LoggingEstimator(operations_callback=operations_callback)

        if gradient_cls in [SPSAEstimatorGradient]:
            gradient = gradient_cls(estimator, epsilon=0.01)
        else:
            gradient = gradient_cls(estimator)

        job = gradient.run([circuit], [observable], [values])
        result = job.result()

        with self.subTest(msg="assert initialize is preserved"):
            self.assertTrue(all("initialize" in ops_i[0].keys() for ops_i in ops))

        with self.subTest(msg="assert result is correct"):
            self.assertAlmostEqual(result.gradients[0].item(), expect, places=5)

if __name__ == "__main__":
    unittest.main()
