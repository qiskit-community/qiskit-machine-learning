# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""An implementation of Quantum K-Means Clustering"""

import warnings
from typing import Tuple, Union, Optional, Callable

import numpy as np
import scipy.sparse as sp
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ControlledGate
from qiskit.extensions import Initialize
from qiskit.utils import QuantumInstance
from sklearn.base import ClusterMixin
from sklearn.cluster import kmeans_plusplus
from sklearn.utils.sparsefuncs import mean_variance_axis

from ...exceptions import QiskitMachineLearningError, QiskitMachineLearningWarning


def _swap_test(psi: QuantumCircuit, phi: QuantumCircuit) -> QuantumCircuit:
    """Method that creates the circuit for Euclidean
        distance calculation between two states using `cswap`.

    Args:
        psi: QuantumCircuit
            First state.
        phi: QuantumCircuit
            Second state.

    Returns:
        swap_test: QuantumCircuit
            The circuit that calculates the distance between the two states.
    """

    control_qubit = QuantumRegister(1, name="control")
    cq_measure = ClassicalRegister(1)
    qc = QuantumCircuit(2 * len(psi.qubits), name="SwapTest")
    qc.compose(psi, qubits=qc.qubits[: len(psi.qubits)], inplace=True)
    qc.compose(phi, qubits=qc.qubits[len(psi.qubits) :], inplace=True)
    qc.add_register(control_qubit, cq_measure)
    qc.barrier()
    qc.h(control_qubit)
    for i in range(len(psi.qubits)):
        qc.cswap(
            control_qubit=control_qubit,
            target_qubit1=qc.qubits[i],
            target_qubit2=qc.qubits[i + len(psi.qubits)],
        )
    qc.h(control_qubit)
    qc.barrier()
    qc.measure(control_qubit, cq_measure)
    return qc


class QKMeans(ClusterMixin):
    """Quantum K-Means Clustering. Implements Scikit-Learn compatible methods for
    clustering and extends ``ClusterMixin``. See `Scikit-Learn <https://scikit-learn.org>`__
    for more details.

    **References**
        [1] Dawid Kopczyk, Quantum machine learning for data scientists.
            `QML for Data Scientists <https://arxiv.org/pdf/1804.10068>`_
    """

    def __init__(
        self,
        n_clusters: int = 3,
        init: Union[str, np.ndarray, Callable] = "k-means++",
        max_iter: Optional[int] = 100,
        tol: float = 1e-4,
        verbose: bool = False,
        quantum_instance: QuantumInstance = None,
    ) -> None:
        """
        Args:
            n_clusters: The number of clusters to form as well as the number of centroids to generate.
            init: Method for initialization :

                    'k-means++' : selects initial cluster centers for k-mean clustering
                    in a smart way to speed up convergence.
                    'random': choose `n_clusters` observations (rows) at random from data
                    for the initial centroids.
                    If an array is passed, it should be of shape (n_clusters, n_features)
                    and gives the initial centers.
                    If a callable is passed, it should take arguments X and n_clusters and
                    return an array of shape (n_clusters, n_features).
            max_iter: Maximum number of iterations of the k-means algorithm for a single run.
            tol: Relative tolerance with regard to Frobenius norm of the difference between
                the previous and new centroids.
            verbose: Verbosity mode.
            quantum_instance: The quantum instance to execute circuits on.
                Note: For better accuracy when calculating the distances,
                the number of shots should be increased.
        """
        self._n_clusters = n_clusters
        self._init = init
        self._max_iter = max_iter
        self._tol = tol
        self._verbose = verbose
        self._quantum_instance = quantum_instance
        self._num_qubits = None

        # defining other variables
        self.cluster_centers_ = None
        self.labels_ = None

    # pylint: disable=invalid-name
    def _check_params(self, X: np.ndarray):
        """Check the parameters passed to the clustering algorithm.

        Args:
            X: numpy.ndarray of shape (n_samples, n_features)
                The input data to be clustered.

        Raises:
            ValueError: If the number of clusters is not positive.
            ValueError: If the number of clusters is greater than the number of samples.
            ValueError: If the maximum number of iterations is not positive.
            ValueError: If init is not one of the supported initializations.
            QiskitMachineLearningError: If the number of qubits in the quantum instance is not
                greater than the number of qubits required to represent the data.
        """
        # n_clusters must be a positive integer
        if self._n_clusters <= 0:
            raise ValueError("n_clusters must be a positive integer")

        # n_clusters must be less than the number of samples
        if X.shape[0] < self._n_clusters:
            raise ValueError(f"n_samples={X.shape[0]} should be >= n_clusters={self._n_clusters}.")

        # init
        if not (
            hasattr(self._init, "__array__")
            or callable(self._init)
            or (isinstance(self._init, str) and self._init in ["k-means++", "random"])
        ):
            raise ValueError(
                "init should be either 'k-means++', 'random', a ndarray or a "
                f"callable, got '{self._init}' instead."
            )
        # if init is a numpy array, check if it is of the correct shape
        if hasattr(self._init, "__array__"):
            if self._init.shape[1] != X.shape[1]:
                raise ValueError(
                    "The number of features of the initial centers must be equal to the number of "
                    "features of the data "
                )
            if self._init.shape[0] != self._n_clusters:
                raise ValueError("The number of initial centers must be equal to n_clusters")
            # Since amplitude encoding is used to represent classical data,
            # the vector with all zeros cannot be encoded as it does not have
            # a unit norm.
            if np.any(np.all(self._init == 0, axis=1)):
                raise QiskitMachineLearningError(
                    "Any initial center cannot be all zeros since it does not have a unit norm."
                )

        # max_iter
        if self._max_iter <= 0:
            raise ValueError(f"max_iter should be > 0, got {self._max_iter} instead.")

        # tol
        self._tol = _tolerance(X, self._tol)

        # quantum_instance
        if self._quantum_instance is None:
            raise ValueError("quantum_instance is required.")
        self._num_qubits = (
            int(np.log2(X.shape[1]))
            if float(np.log2(X.shape[1])).is_integer()
            else int(np.log2(X.shape[1])) + 1
        )
        backend_qubits = self._quantum_instance.backend.configuration().n_qubits
        if backend_qubits < 2 * self._num_qubits + 1:
            raise QiskitMachineLearningError(
                "The number of qubits in the quantum_instance - {} is less than the"
                " number of qubits required - {}.".format(backend_qubits, 2 * self._num_qubits + 1)
            )

        # The dataset must also not contain any 0 vectors since they cannot be
        # represented as amplitudes.
        if np.any(np.all(X == 0, axis=1)):
            raise QiskitMachineLearningError(
                "The dataset contains a 0 vector since which cannot be represented as amplitudes."
            )

    def _prepare_input(self, X: np.ndarray):
        """Prepare the input data for clustering.

        Args:
            X: numpy.ndarray of shape (n_features,)
                The input data to be clustered.

        Returns:
            X: numpy.ndarray of shape (2**n_qubits,)
                The input data to be clustered.
        """

        # pad the input data with zeros if the number of features is not a power of 2
        if not float(np.log2(X.shape[0])).is_integer():
            X = np.pad(X, (0, 2 ** self._num_qubits - X.shape[0]), "constant")

        # if all 0's, then return as is
        if np.all(X == 0):
            return X
        # normalize each vector to have unit norm
        X = X / np.linalg.norm(X)

        return X

    def _gate_for_init(self, x: np.ndarray) -> ControlledGate:
        """Internal method that initializes the state |x>.

        Args:
            x: numpy.ndarray of shape (2**n_qubits,)
                The input data to be initialized.

        Returns:
            `ControlledGate`: The controlled version of the gate that initializes the state |x>.
        """
        # initialize the state
        init_state = Initialize(x)

        # call to generate the circuit that takes the desired vector to zero
        dgc = init_state.gates_to_uncompute()
        # invert the circuit to create the desired vector from zero (assuming
        # the qubits are in the zero state)
        initialize_instr = dgc.to_instruction().inverse()
        q = QuantumRegister(self._num_qubits, "q")
        initialize_circuit = QuantumCircuit(q, name="x_init")
        initialize_circuit.append(initialize_instr, q[:])
        x_con = initialize_circuit.to_gate().control(num_ctrl_qubits=1, label="x_con")

        return x_con

    def _prepare_states(
        self, a: np.ndarray, b: np.ndarray
    ) -> Tuple[QuantumCircuit, QuantumCircuit]:
        """Internal method that prepares the states |psi> and |phi> for a pair of vectors.

        Args:
            a: numpy.ndarray of shape (2**n_qubits,)
                Vector 1.
            b: numpy.ndarray of shape (2**n_qubits,)
                Vector 2.

        Returns:
            psi: QuantumCircuit
                The circuit that prepares the state |psi>.
            phi: QuantumCircuit
                The circuit that prepares the state |phi>.
        """

        # getting the controlled init gates
        a_con = self._gate_for_init(a)
        b_con = self._gate_for_init(b)

        # defining psi
        psi_con = QuantumRegister(1, name="psi_con")
        psi_state = QuantumRegister(self._num_qubits, name="psi_state")
        psi = QuantumCircuit(psi_con, psi_state, name="psi")
        psi.h(psi_con)
        psi.append(a_con, psi.qubits)
        psi.x(psi_con)
        psi.append(b_con, psi.qubits)

        # defining phi
        phi = QuantumCircuit(self._num_qubits + 1, name="phi")
        phi.h(0)

        return psi, phi

    def _distance_calc(self, a: np.ndarray, b: np.ndarray) -> float:
        """Internal method that calculates the Euclidean distance between two vectors.

        Args:
            a: numpy.ndarray of shape (2**n_qubits,)
                Vector 1.
            b: numpy.ndarray of shape (2**n_qubits,)
                Vector 2.

        Returns:
            distance: float
                The Euclidean distance between the two vectors.
        """

        # preparing the input data
        a = self._prepare_input(a)
        b = self._prepare_input(b)

        # preparing the states
        psi, phi = self._prepare_states(a, b)

        # performing SwapTest subroutine and executing it
        swap_test = _swap_test(psi, phi)
        swap_test_result = self._quantum_instance.execute(swap_test)

        # calculating the distance
        counts = swap_test_result.get_counts(swap_test)
        z = 2  # since both states are normalized
        dist = 4 * z * (counts["0"] / self._quantum_instance.run_config.shots - 0.5)
        return dist

    # pylint: disable=invalid-name
    def _validate_center_shape(self, X, centers):
        """Check if centers is compatible with X and n_clusters."""
        if centers.shape[0] != self._n_clusters:
            raise ValueError(
                f"The shape of the initial centers {centers.shape} does not "
                f"match the number of clusters {self._n_clusters}."
            )
        if centers.shape[1] != X.shape[1]:
            raise ValueError(
                f"The shape of the initial centers {centers.shape} does not "
                f"match the number of features of the data {X.shape[1]}."
            )

    # pylint: disable=invalid-name
    def _init_centroids(self, X: np.ndarray):
        """Internal method that computes the initial centroids.

        Args:
            X: numpy.ndarray of shape (n_samples, n_features)
                The input dataset.

        Returns:
            centroids: numpy.ndarray of shape (n_clusters, n_features)
                The initial centroids.
        """
        n_samples = X.shape[0]
        n_clusters = self._n_clusters

        if isinstance(self._init, str) and self._init == "k-means++":
            centers, _ = kmeans_plusplus(
                X,
                n_clusters,
            )
        elif isinstance(self._init, str) and self._init == "random":
            seeds = np.random.choice(n_samples, n_clusters, replace=False)
            centers = X[seeds]
        elif hasattr(self._init, "__array__"):
            centers = self._init
        elif callable(self._init):
            centers = self._init(X, n_clusters)
            self._validate_center_shape(X, centers)

        return centers

    # pylint: disable=invalid-name
    def _kmeans_run(self, X: np.ndarray, centers_init: np.ndarray):
        """Internal method that runs the K-Means algorithm.

        Args:
            X: numpy.ndarray of shape (n_samples, n_features)
                The input dataset.
            centers_init: numpy.ndarray of shape (n_clusters, n_features)
                The initial centroids.

        Returns:
            centers: numpy.ndarray of shape (n_clusters, n_features)
                The final centroids.
            labels: numpy.ndarray of shape (n_samples,)
                The labels of each sample.
        """
        n_clusters = centers_init.shape[0]

        centers = centers_init
        centers_new = np.zeros_like(centers)
        labels = np.full(X.shape[0], -1, dtype=np.int32)

        for i in range(self._max_iter):
            labels_old = labels.copy()
            # iterate over all samples
            for j in range(X.shape[0]):
                # initialize empty distance array
                distances = np.zeros(n_clusters)
                # iterate over all clusters
                for k in range(n_clusters):
                    # calculate distance between sample and cluster
                    distances[k] = self._distance_calc(X[j], centers[k])
                # assign sample to the closest cluster
                labels[j] = np.argmin(distances)
            # update cluster centers
            for k in range(n_clusters):
                # calculate mean of all samples assigned to cluster
                # to calculate the new cluster center
                if X[labels == k].shape[0] > 0:
                    centers_new[k] = np.mean(X[labels == k], axis=0)
                else:
                    centers_new[k] = centers[k]
            # check for strict convergence
            if np.array_equal(labels, labels_old):
                if self._verbose:
                    print(f"Strict convergence reached after {i} iterations.")
                break
            # check for convergence based on tolerance
            if np.linalg.norm(centers - centers_new) < self._tol:
                centers = centers_new
                if self._verbose:
                    print(f"Converged after {i} iterations.")
                break
            centers = centers_new.copy()

        return centers, labels

    def fit(self, X: np.ndarray):
        """Compute Quantum K-Means clustering.

        Args:
            X: numpy.ndarray of shape (n_samples, n_features)
                The input dataset.

        Returns:
            self: object

        Raises:
            TypeError: If the input data is not a numpy.ndarray.
            ValueError: If the input data is not 2D.
        """
        # validate data
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy.ndarray.")
        if len(X.shape) != 2:
            raise ValueError("X must be a 2D array.")

        self._check_params(X)

        # initialize centroids
        centers_init = self._init_centroids(X)

        # run k-means
        centers, labels = self._kmeans_run(X, centers_init)

        # save results
        self.cluster_centers_ = centers
        self.labels_ = labels

        distinct_clusters = len(set(labels))
        if distinct_clusters < self._n_clusters:
            warnings.warn(
                "The number of distinct clusters ({}) is smaller than the "
                "number of requested clusters ({}). Possibly due to "
                " duplicate samples.".format(distinct_clusters, self._n_clusters),
                QiskitMachineLearningWarning,
            )

        return self

    def fit_predict(self, X, y=None):
        return super().fit_predict(X, y=y)

    @property
    def n_clusters(self) -> int:
        """Getter for the number of clusters."""
        return self._n_clusters

    @n_clusters.setter
    def n_clusters(self, n_clusters: int):
        """Setter for the number of clusters."""
        self._n_clusters = n_clusters

    @property
    def init(self) -> Union[str, np.ndarray]:
        """Getter for the initial centroids."""
        return self._init

    @init.setter
    def init(self, init: Union[str, np.ndarray]):
        """Setter for the initial centroids."""
        self._init = init

    @property
    def max_iter(self) -> int:
        """Getter for the maximum number of iterations."""
        return self._max_iter

    @max_iter.setter
    def max_iter(self, max_iter: int):
        """Setter for the maximum number of iterations."""
        self._max_iter = max_iter

    @property
    def tol(self) -> float:
        """Getter for the tolerance."""
        return self._tol

    @tol.setter
    def tol(self, tol: float):
        """Setter for the tolerance."""
        self._tol = tol


# pylint: disable=invalid-name
def _tolerance(X, tol):
    """Return a tolerance which is dependent on the dataset."""
    if tol == 0:
        return 0
    if sp.issparse(X):
        variances = mean_variance_axis(X, axis=0)[1]
    else:
        variances = np.var(X, axis=0)
    return np.mean(variances) * tol
