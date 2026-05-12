# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
H Atom Pauli Forms (Developer-Only)
"""

import numpy as np
import pickle as pkl
import os
import itertools

from qiskit.quantum_info import SparsePauliOp

"""
- H2: Usually found at a `0.735 Å` equilibrium bond distance
- H3+: Usually found both in an equilateral triangle configuration with `0.9 Å` between each pair
- H6: Usually modelled as a linear chain of 6 atoms with bond lengths `1 Å` between each pair
"""
_molecules = {
    "H2": {"atom": "H 0 0 0; H 0 0 0.735", "basis": "sto-3g", "charge": 0, "spin": 0},
    "H3": {
        "atom": "H 0 0 -0.45; H 0 0 0.45; H 0 0.78 0;",
        "basis": "sto-3g",
        "charge": 1,
        "spin": 0,
    },
    "H6": {
        "atom": "H 0 0 0; H 0 0 1; H 0 0 2; H 0 0 3; H 0 0 4; H 0 0 5;",
        "basis": "sto-3g",
        "charge": 0,
        "spin": 0,
    },
}

# Pauli multiplication table
_mul_table = {
    ("I", "I"): (1, "I"),
    ("I", "X"): (1, "X"),
    ("I", "Y"): (1, "Y"),
    ("I", "Z"): (1, "Z"),
    ("X", "I"): (1, "X"),
    ("X", "X"): (1, "I"),
    ("X", "Y"): (1j, "Z"),
    ("X", "Z"): (-1j, "Y"),
    ("Y", "I"): (1, "Y"),
    ("Y", "X"): (-1j, "Z"),
    ("Y", "Y"): (1, "I"),
    ("Y", "Z"): (1j, "X"),
    ("Z", "I"): (1, "Z"),
    ("Z", "X"): (1j, "Y"),
    ("Z", "Y"): (-1j, "X"),
    ("Z", "Z"): (1, "I"),
}


def _a_p(p, n):
    """JW Substituion for Annihilation Operator"""
    z_str, t_str = ["Z"] * p, ["I"] * (n - p - 1)
    return {tuple(z_str + ["X"] + t_str): 0.5, tuple(z_str + ["Y"] + t_str): 0.5j}


def _a_p_dag(p, n):
    """JW Substitution for Creation Operator"""
    z_str, t_str = ["Z"] * p, ["I"] * (n - p - 1)
    return {tuple(z_str + ["X"] + t_str): 0.5, tuple(z_str + ["Y"] + t_str): -0.5j}


def _n_p(p, n):
    """JW Substitution for 1-Body Diagonal Entries"""
    id_str = ("I",) * n
    z_str = tuple("Z" if i == p else "I" for i in range(n))
    return {id_str: 0.5, z_str: -0.5}


def _z_string(idxs, n):
    """Helper for Z string generation"""
    return {tuple("Z" if i in idxs else "I" for i in range(n)): 1.0}


def _mul_strs(s1, s2):
    """Multiply Pauli Strings of form (Q0_op, Q1_op, Q2_op...)"""
    phase, res = 1, []
    for a, b in zip(s1, s2):
        ph, op = _mul_table[(a, b)]
        phase *= ph
        res.append(op)
    return phase, tuple(res)


def _mul_pauli(t1, t2):
    """Multiply Pauli Summation Dicts of form {Pauli String: Coefficient}"""
    res = {}
    for s1, c1 in t1.items():
        for s2, c2 in t2.items():
            ph, s = _mul_strs(s1, s2)
            res[s] = res.get(s, 0) + ph * c1 * c2
            if abs(res[s]) == 0:
                del res[s]
    return res


def _add_pauli(H, t, coef=1.0):
    """Add Pauli Summation Dicts of form {Pauli String: Coefficient}"""
    for s, c in t.items():
        H[s] = H.get(s, 0) + coef * c
        if abs(H[s]) == 0:
            del H[s]


def _JW_map(E_nuc, h_so, g_so, eps=1e-15):
    """Jordan Wigner mapping of each Spin Orbital to a Qubit"""
    n = h_so.shape[0]
    H = {("I",) * n: E_nuc}

    for p in range(n):
        _add_pauli(H, _n_p(p, n), h_so[p, p])

    for p in range(n):
        for q in range(p + 1, n):
            c = 0.5 * (h_so[p, q] + np.conj(h_so[q, p]))
            if abs(c) < eps:
                continue
            _add_pauli(H, _mul_pauli(_a_p_dag(p, n), _a_p(q, n)), c)
            _add_pauli(H, _mul_pauli(_a_p_dag(q, n), _a_p(p, n)), np.conj(c))

    for p, q, r, s in itertools.product(range(n), repeat=4):
        g = 0.5 * g_so[p, q, r, s]
        if abs(g) < eps:
            continue
        uniq = len({p, q, r, s})

        if uniq == 4:
            term1 = _mul_pauli(
                _mul_pauli(_a_p_dag(p, n), _a_p_dag(q, n)), _mul_pauli(_a_p(r, n), _a_p(s, n))
            )
            term2 = _mul_pauli(
                _mul_pauli(_a_p_dag(s, n), _a_p_dag(r, n)), _mul_pauli(_a_p(q, n), _a_p(p, n))
            )
            _add_pauli(H, term1, g)
            _add_pauli(H, term2, np.conj(g))
        elif uniq == 3:
            term = _mul_pauli(
                _mul_pauli(_a_p_dag(p, n), _a_p_dag(q, n)), _mul_pauli(_a_p(r, n), _a_p(s, n))
            )
            _add_pauli(H, term, g)
            _add_pauli(H, _mul_pauli(_z_string([p], n), term), -g)
            _add_pauli(H, _mul_pauli(_z_string([q], n), term), -g)
        elif uniq == 2 and p != q and r == s:
            coeff = 0.25 * g
            z_p = _z_string([p], n)
            z_q = _z_string([q], n)
            z_pq = _z_string([p, q], n)
            _add_pauli(H, {("I",) * n: -coeff})
            _add_pauli(H, z_p, coeff)
            _add_pauli(H, z_q, coeff)
            _add_pauli(H, z_pq, -coeff)

    return {k: v for k, v in H.items() if abs(v) > eps}


def _save_H_atom_pauli_forms():
    r"""Generates and saves

    This is a Developer-only module. The results of this code has already been
    cached in the repository. This has been included in the repository for
    ready availability of the generation process for future developments.

    Running this needs installing PySCF which is not included as a
    requirement in requirements-dev.txt. If you are using a Windows
    environment, it is recommended that you proceed with this module
    on a different operating system and drop-in import the results.

    This saves H Atom Hamiltonians in Pauli form for further usage
    by other dataset generators

    """
    try:
        from pyscf import gto, scf, ao2mo
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            """This Developer-Only Module requires PySCF. Please install it with 
            pip install --prefer-binary pyscf. If you are using a Windows
            environment, it is recommended that you proceed with this module 
            on a different operating system and drop-in import the results"""
        )

    dir_path = os.path.dirname(__file__)

    for label, params in _molecules.items():

        # PySCF Simulations
        mol = gto.M(**params)
        mf = scf.RHF(mol).run(conv_tol=1e-12)

        # E_nuc, 1-body and 2-body Integrals
        E_nuc = mol.energy_nuc()
        h_ao = mf.get_hcore()
        g_ao = mol.intor("int2e")

        # Converting to Spatial Orbitals
        C = mf.mo_coeff
        h_mo = C.T @ h_ao @ C
        g_mo8 = ao2mo.kernel(mol, C)
        g_mo = ao2mo.restore(1, g_mo8, C.shape[1])

        # Converting to Spin Orbitals
        eps = 1e-18
        n_mo = h_mo.shape[0]
        n_so = 2 * n_mo

        h_so = np.zeros((n_so, n_so), dtype=complex)
        g_so = np.zeros((n_so, n_so, n_so, n_so), dtype=complex)

        # Single Body Terms
        for p in range(n_mo):
            for q in range(n_mo):
                val = h_mo[p, q]
                if abs(val) < eps:
                    continue
                h_so[2 * p, 2 * q] = val
                h_so[2 * p + 1, 2 * q + 1] = val

        # Two Body Terms
        for p, q, r, s in itertools.product(range(n_mo), repeat=4):
            val = g_mo[p, q, r, s]
            if abs(val) < eps:
                continue
            g_so[2 * p, 2 * q, 2 * r, 2 * s] = val
            g_so[2 * p + 1, 2 * q + 1, 2 * r + 1, 2 * s + 1] = val
            g_so[2 * p, 2 * q + 1, 2 * r, 2 * s + 1] = val
            g_so[2 * p + 1, 2 * q, 2 * r + 1, 2 * s] = val

        # Jordan Wigner Transform uses O(n) depth.
        JW_H = _JW_map(E_nuc, h_so, g_so)
        # Alternatively Bravyi-Kaetev Transform can give O(logn)

        # Convert to SparsePauliOp

        # We used qubits in (0,1...) indexing in pauli strings while qiskit
        # uses (...1,0). Hence we reverse here
        pauli_list = [("".join(reversed(k)), v) for k, v in JW_H.items()]
        spo = SparsePauliOp.from_list(pauli_list)

        fname = f"h_molecule_hamiltonians/{label}.bin"
        finalpath = os.path.join(dir_path, fname)

        with open(finalpath, "wb") as f:
            pkl.dump(JW_H, f)

    print("Hamiltonians saved.")


if __name__ == "__main__":
    _save_H_atom_pauli_forms()
