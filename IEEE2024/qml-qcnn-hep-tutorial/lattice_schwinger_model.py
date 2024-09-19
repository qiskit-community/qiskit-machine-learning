# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Lattice Schwinger model class."""

from qiskit.quantum_info import Pauli, SparsePauliOp


class LatticeSchwingerModel:
    """Lattice Schwinger model."""

    def __init__(
        self,
        mass: float,
        coupling: float,
        num_sites: int,
        spacing: float,
        left_gauge: float = 0,
    ) -> None:
        """
        Args:
            mass: Particle mass.
            coupling: Coupling constant (e.g., -e for QED).
            num_sites: Number of lattice sites.
            spacing: Lattice constant.
            left_gauge: Left gauge.
        """
        assert (
            num_sites % 2 == 0
        ), f"Number of sites is {num_sites} but should be even for staggered fermions!"

        self._mass = mass
        self._couple = coupling
        self._num_sites = num_sites
        self._spacing = spacing
        self._left_gauge = left_gauge

    def _build_hopping_ops(self) -> tuple[list[SparsePauliOp], list[SparsePauliOp]]:
        """Constructs all hopping terms of the Hamiltonian after qubit-mapping.

        These are (sigma^+
        sigma^_ + h.c.) terms which simplify to (YY + XX) terms and are split into mutually
        non-commuting even and odd sums of Paulis within which all terms commute.
        """
        coeff = 1.0 / (4 * self._spacing)
        ops: tuple[list[SparsePauliOp], list[SparsePauliOp]] = ([], [])
        for parity in [0, 1]:
            for j in range(parity, self._num_sites - 1, 2):
                for pauli in ["X", "Y"]:
                    ops[parity].append(
                        SparsePauliOp(
                            [f"{'I' * (self._num_sites - 1 - j - 1)}{pauli}{pauli}{'I' * j}"],
                            [coeff],
                        )
                    )

        return ops

    def _build_z_ops(self) -> list[SparsePauliOp]:
        """
        Constructs all Pauli operators constituting the mass term as well as the gauge-field term of
        the Hamiltonian with staggered fermions. These operators consist of only I and Z.
        """
        op_z = []

        coeff_mass = 0.5 * self._mass
        coeff_couple = 0.5 * self._couple * self._couple * self._spacing
        n_mod2 = self._num_sites % 2
        coeff_const = -coeff_mass * n_mod2 + coeff_couple * (
            (self._num_sites - 1) * self._left_gauge * self._left_gauge
            + (0.125 - 0.5 * self._left_gauge) * (self._num_sites - n_mod2)
        )

        for j in range(self._num_sites - 1):
            op_z_j = []

            # Mass terms
            op_z_j.append(
                SparsePauliOp(
                    [f"{'I' * (self._num_sites - 1 - j)}{'Z'}{'I' * j}"],
                    [coeff_mass * (1 if j % 2 == 0 else -1)],
                )
            )

            coeff_j = self._left_gauge + 0.5 * ((j + 1) % 2)

            # Gauge terms from Gauss' law.
            for l in range(j + 1):  # need +1 as otherwise sum stops at N-3 instead of N-2.
                op_z_j.append(
                    SparsePauliOp(
                        [f"{'I' * (self._num_sites - 1 - l)}{'Z'}{'I' * l}"],
                        [coeff_j * coeff_couple],
                    )
                )
                pauli1 = Pauli(f"{'I' * (self._num_sites - 1 - l)}{'Z'}{'I' * l}")

                for ll in range(j + 1):
                    if l == ll:
                        coeff_const += 0.25 * coeff_couple
                    else:
                        pauli2 = Pauli(f"{'I' * (self._num_sites - 1 - ll)}{'Z'}{'I' * ll}")
                        op_z_j.append(SparsePauliOp([pauli1.dot(pauli2)], [0.25 * coeff_couple]))

            op_z.append(op_z_j)

        # j = N mass term
        op_z.append(
            SparsePauliOp(
                [f"{'Z'}{'I' * (self._num_sites - 1)}"],
                [coeff_mass * (1 if (self._num_sites - 1) % 2 == 0 else -1)],
            )
        )

        # Constant term is left last, so it can be easily discarded.
        # op_z.append(SparsePauliOp([self._num_sites*'I'], [coeff_const])

        return op_z

    def get_hamiltonian(self) -> list[SparsePauliOp]:
        """Returns the Hamiltonian as a list of sparse Pauli operators.

        The operators are grouped such that within each sum, all Paulis commute but the elements of
        the list (i.e., Pauli sums) do not commute mutually.
        """
        hamiltonian = []

        hopping_ops = self._build_hopping_ops()
        coupling_ops = self._build_z_ops()

        for op_list in hopping_ops:
            hamiltonian.append(sum(op_list))

        summed_coupling_op = sum(sum(op_list) for op_list in coupling_ops)
        hamiltonian.append(summed_coupling_op)

        return hamiltonian
