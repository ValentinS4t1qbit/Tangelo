# Copyright 2023 Good Chemistry Company.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Combinatorial mapping as described in references (1) and (2). In contrast to
qubit mappings such as JW, BK that use occupation/parity, the mappings in this
file use the Fock configurations as the elements of the basis set. Thus, the
number of required qubits scales with the number of electronic configuration
instead of the number of spinorbitals.

References:
    1. Streltsov, A. I., Alon, O. E. & Cederbaum, L. S. General mapping for
        bosonic and fermionic operators in Fock space. Phys. Rev. A 81, 022124
        (2010).
    2. Chamaki, D., Metcalf, M. & de Jong, W. A. Compact Molecular Simulation on
        Quantum Computers via Combinatorial Mapping and Variational State
        Preparation. Preprint at https://doi.org/10.48550/arXiv.2205.11742
        (2022).
"""
import gc
import os
import string
import json
import pickle
import sys
import itertools
import math
import time
from collections import OrderedDict
from itertools import product
from multiprocessing import cpu_count, Pool
from functools import partial

#import codon
#from numba import jit
import numpy as np
from scipy.special import comb
from scipy.sparse import lil_matrix, coo_matrix, coo_array
from openfermion.transforms import chemist_ordered

from tangelo.toolboxes.operators import QubitOperator


# os.environ["JULIA_NUM_THREADS"] = str(4) # str(cpu_count())
#
# import julia
# jl = julia.Julia(compiled_modules=False)
#
# from julia import Main
# Main.include("combinatorial.jl")


ZERO_TOLERANCE = 1e-8


#@codon.jit(debug=True)
def int_to_tuple(integer, n_qubits):
    """ Convert a qubit Hamiltonian term in integer encoding (stabilizer representation) into
    an Openfermion-style tuple.

    Bits in the binary representation of the integer encode the Pauli operators that need to be applied
    to each qubit. Each consecutive pair of bits encode information for a specific qubit.

    Args:
        integer (int): integer to decode. Its binary representation has 2*n_qubits bits
        n_qubits (int): number of qubits in the term

    Returns:
        tuple: OpenFermion-style term including up to n_qubits qubits
    """

    term = []
    for i in range(1, n_qubits+1):
        shift_x = 2*(i-1)
        shift_z = shift_x + 1

        x_term = (integer & (1 << shift_x)) >> shift_x
        z_term = (integer & (1 << shift_z)) >> shift_z

        if (x_term, z_term) == (0, 0):
            continue
        elif (x_term, z_term) == (1, 0):
            term.append((i-1, 'X'))
        elif (x_term, z_term) == (0, 1):
            term.append((i-1, 'Z'))
        else:
            term.append((i-1, 'Y'))

    return tuple(term)

#@profile
#@codon.jit(debug=True)
def int_to_tuple2(integer, n_qubits):
    """ Convert a qubit Hamiltonian term in integer encoding (stabilizer representation) into
    an Openfermion-style tuple.

    Bits in the binary representation of the integer encode the Pauli operators that need to be applied
    to each qubit. Each consecutive pair of bits encode information for a specific qubit.

    Args:
        integer (int): integer to decode. Its binary representation has 2*n_qubits bits
        n_qubits (int): number of qubits in the term

    Returns:
        tuple: OpenFermion-style term including up to n_qubits qubits
    """

    #bs = format(integer, f'0{str(2 * n_qubits)}b')[::-1]
    #bs += (2*n_qubits-len(bs))*'0'

    bs = bin(integer)[2:][::-1]
    if len(bs)%2:
        bs += '0'
    term = []

    for i in range(len(bs)//2): #range(n_qubits):
        xz_term = bs[2*i:2*i+2]

        if xz_term == '10':
            term.append((i, 'X'))
        elif xz_term == '01':
            term.append((i, 'Z'))
        elif xz_term == '11':
            term.append((i, 'Y'))

    return tuple(term)

#@codon.jit(debug=True)
def tensor_product_pauli_dicts(pa, pb):
    """ Perform the tensor product of 2 Pauli operators by using their underlying dictionary of terms.

    Args:
        pa (dict[int -> complex]): first Pauli dictionary
        pb (dict[int -> complex]): second Pauli dictionary

    Returns:
        dict[int -> complex]: tensor product of Pauli operators
    """

    pp = dict()
    for ta, ca in pa.items():
        for tb, cb in pb.items():
            pp[ta ^ tb] = ca*cb
    return pp


def one_body_op_on_state(op, state_in):
    """Function to apply a a^{\dagger}_i a_j operator as described in Phys. Rev.
    A 81, 022124 (2010) eq. (8).

    Args:
        op (tuple): Operator, written as ((qubit_i, 1), (qubit_j, 0)), where 0/1
            means annihilation/creation on the specified qubit.
        state_in (tuple): Electronic configuration described as tuple of
            spinorbital indices where there is an electron.

    Returns:
        tuple: Resulting state with the same form as in the input state.
            Can be 0.
        int: Phase shift. Can be -1 or 1.
    """

    # Copy the state, then transform it into a list (it will be mutated).
    state = list(state_in)

    # Unpack the creation and annihilation operators.
    creation_op, annihilation_op = op
    creation_qubit, creation_dagger = creation_op
    annihilation_qubit, annihilation_dagger = annihilation_op

    # annihilation logics on the state.
    if annihilation_qubit in state:
        state.remove(annihilation_qubit)
    else:
        return (), 0

    # Creation logics on the state.
    if creation_qubit not in state:
        state.append(creation_qubit)
    else:
        return (), 0

    # Compute the phase shift.
    if annihilation_qubit > creation_qubit:
        d = sum(creation_qubit < i < annihilation_qubit for i in state)
    elif annihilation_qubit < creation_qubit:
        d = sum(annihilation_qubit < i < creation_qubit for i in state)
    else:
        d = 0

    return tuple(sorted(state)), (-1)**d


def recursive_mapping(M):
    """ Maps an arbitrary square matrix representing an operator via Pauli decomposition.

    Args:
        M (np.array(np, complex, np.complex)): Square matrix representing the operator.

    Returns:
        dict[int -> complex]: Pauli operator encoded as a dictionary
    """

    n_rows, n_cols = M.shape

    # Bottom of recursion: 2x2 matrix case
    if n_rows == 2:
        res = {0: 0.5*(M[0,0]+M[1,1]), 1: 0.5*(M[0,1]+M[1,0]), 2: 0.5*(M[0,0]-M[1,1]), 3: 0.5j*(M[0,1]-M[1,0])}
        return res
    else:
        n_qubits = int(math.log2(n_rows))
        piv = n_rows//2
        shift_x = 2*(n_qubits-1)
        shift_z = shift_x + 1

        # 1/2 (I +-Z)
        z_op = (1 << shift_z)
        i_plus_z = {0: 0.5, z_op: 0.5}
        i_minus_z = {0: 0.5, z_op: -0.5}

        # 1/2 (X +-iY)
        x_op = (1 << shift_x)
        y_op = x_op | (1 << shift_z)
        x_plus_iy = {x_op: 0.5, y_op: 0.5j}
        x_minus_iy = {x_op: 0.5, y_op: -0.5j}

        M_00 = tensor_product_pauli_dicts(recursive_mapping(M[:piv, :piv]), i_plus_z)
        M_11 = tensor_product_pauli_dicts(recursive_mapping(M[piv:, piv:]), i_minus_z)
        M_01 = tensor_product_pauli_dicts(recursive_mapping(M[:piv, piv:]), x_plus_iy)
        M_10 = tensor_product_pauli_dicts(recursive_mapping(M[piv:, :piv]), x_minus_iy)

        # Merge the 4 outputs into one additively
        for d in M_01, M_10, M_11:
            for (k, v) in d.items():
                M_00[k] = M_00.get(k, 0.) + v
        return M_00

#@profile
def recursive_mapping_dict(M, n, s): # n is n_rows and n_cols here

    # Bottom of recursion: 2x2 matrix case
    if n == 2:
        M2 = {(k[0]%2, k[1]%2): v for k, v in M.items()}
        m00, m01, m10, m11 = M2.get((0, 0), 0.), M2.get((0, 1), 0.), M2.get((1, 0), 0.), M2.get((1, 1), 0.)
        res = {0: 0.5 * (m00 + m11), 1: 0.5 * (m01 + m10), 2: 0.5 * (m00 - m11), 3: 0.5j * (m01 - m10)}
        #print(res)
        return res
    else:

        n_qubits = int(math.log2(n))

        shift_x = 2 * (n_qubits - 1)
        shift_z = shift_x + 1

        # 1/2 (I +-Z)
        z_op = (1 << shift_z)
        i_plus_z = {0: 0.5, z_op: 0.5}
        i_minus_z = {0: 0.5, z_op: -0.5}

        # 1/2 (X +-iY)
        x_op = (1 << shift_x)
        y_op = x_op | (1 << shift_z)
        x_plus_iy = {x_op: 0.5, y_op: 0.5j}
        x_minus_iy = {x_op: 0.5, y_op: -0.5j}

        piv = n // 2

        # # Split into smaller dicts
        # Ms_00, Ms_01, Ms_10, Ms_11 = dict(), dict(), dict(), dict()
        #
        # for ((x, y),v) in M.items():
        #     if x < piv + s[0]:
        #         if y < piv + s[1]:
        #             Ms_00[(x, y)] = v
        #         else:
        #             Ms_01[(x, y)] = v
        #     else:
        #         if y < piv + s[1]:
        #             Ms_10[(x, y)] = v
        #         else:
        #             Ms_11[(x, y)] = v

        # Ms = [Ms_00, Ms_11, Ms_01, Ms_10]


        def get_quadrant_matrix(M, quadrant, piv, s):

            xp, yp = piv + s[0], piv + s[1]
            if quadrant == 0: #00
                Mq = {(x, y): v for ((x, y), v) in M.items() if x < xp and y < yp}
            elif quadrant == 1: #01
                Mq = {(x, y): v for ((x, y), v) in M.items() if x < xp and y >= yp}
            elif quadrant == 2: #10
                Mq = {(x, y): v for ((x, y), v) in M.items() if x >= xp and y < yp}
            else: #11
                Mq = {(x, y): v for ((x, y), v) in M.items() if x >= xp and y >= yp}
            return Mq

        quadrants = [0, 3, 1, 2]
        shifts = [(s[0], s[1]), (s[0]+piv, s[1]+piv), (s[0], s[1]+piv), (s[0]+piv, s[1])]
        values = [i_plus_z, i_minus_z, x_plus_iy, x_minus_iy]

        # res = dict()
        # for q, ss, v in zip(quadrants, shifts, values):
        #
        #     m = get_quadrant_matrix(M, q, piv, (s[0], s[1]))
        #     if m:
        #         d = tensor_product_pauli_dicts(recursive_mapping_dict(m, n//2, ss), v)
        #         for (k, v) in d.items(): res[k] = res.get(k, 0.) + v
        # res = {k: v for k, v in res.items() if v != 0}
        # return res

        res = dict()
        for q, ss, v in zip(quadrants, shifts, values):

            m = get_quadrant_matrix(M, q, piv, (s[0], s[1]))
            d = tensor_product_pauli_dicts(recursive_mapping_dict(m, n // 2, ss), v)
            d = {k: v for k, v in d.items() if abs(v) > 1e-10}

            # High level case: potentially use a lot of memory, dump into file and agglomerate later
            if n_qubits > 10:
                with open(f'comb_{n_qubits}_{q}.pkl', 'wb') as file:
                    #file.write(json.dumps(d))
                    t1 = time.time()
                    pickle.dump(d, file, protocol=5)
                    t2 = time.time()
                    print(f'comb_{n_qubits}_{q}.pkl \t written in ({t2-t1:.2f} s)')
            # Low level case: should be easy to immediately agglomerate
            else:
                if m:
                    for (k, v) in d.items(): res[k] = res.get(k, 0.) + v

        # If high level case: read from files and agglomerate partial results
        if n_qubits > 10:
            for f in [f'comb_{n_qubits}_{q}.pkl' for q in range(4)]:
                with open(f, 'rb') as file:
                    #d = json.load(file)
                    t1 = time.time()
                    d = pickle.load(file)
                    t2 = time.time()
                    print(f'{f} \t loaded in ({t2-t1:.2f} s)')
                    for (k, v) in d.items(): res[k] = res.get(k, 0.) + v
                os.remove(f)

        # remove zero entries and return result
        res = {k: v for k, v in res.items() if v != 0}
        return res

def prep(n):
    # print(n)
    n_qubits = int(math.log2(n))
    shift_x = 2 * (n_qubits - 1)
    shift_z = shift_x + 1

    # 1/2 (I +-Z)
    z_op = (1 << shift_z)
    i_plus_z = {0: 0.5, z_op: 0.5}
    i_minus_z = {0: 0.5, z_op: -0.5}

    # 1/2 (X +-iY)
    x_op = (1 << shift_x)
    y_op = x_op | (1 << shift_z)
    x_plus_iy = {x_op: 0.5, y_op: 0.5j}
    x_minus_iy = {x_op: 0.5, y_op: -0.5j}

    return i_plus_z, i_minus_z, x_plus_iy, x_minus_iy

def combinatorial_jl(ferm_op, n_modes, n_electrons):
    """Function to transform the fermionic Hamiltonian into a basis constructed
    in the Fock space.

    Args:
        ferm_op (FermionOperator). Fermionic operator, with alternate ordering
            as followed in the openfermion package
        n_modes (int): Number of relevant molecular orbitals, i.e. active molecular
            orbitals.
        n_electrons (int): Number of active electrons.

    Returns:
        QubitOperator: Self-explanatory.
    """

    # The chemist ordering splits some 1-body and 2-body terms.
    ferm_op = chemist_ordered(ferm_op)

    # Specify the number of alpha and beta electrons.
    if isinstance(n_electrons, tuple) and len(n_electrons) == 2:
        n_alpha, n_beta = n_electrons
    elif isinstance(n_electrons, int) and n_electrons % 2 == 0:
        n_alpha = n_beta = n_electrons // 2
    else:
        raise ValueError(f"{n_electrons} is not a valid entry for n_electrons, must be a tuple or an int.")

    # Get the number of qubits n.
    n_choose_alpha = comb(n_modes, n_alpha, exact=True)
    n_choose_beta = comb(n_modes, n_beta, exact=True)
    n = math.ceil(np.log2(n_choose_alpha * n_choose_beta))

    # Construct the basis set where each configutation is mapped to a unique
    # integer.
    basis_set_alpha = basis(n_modes, n_alpha)
    basis_set_beta = basis(n_modes, n_beta)
    basis_set = dict()

    for sigma_alpha, int_alpha in basis_set_alpha.items():
        for sigma_beta, int_beta in basis_set_beta.items():
            # Alternate ordering (like FermionOperator in openfermion).
            sigma = tuple(sorted([2*sa for sa in sigma_alpha] + [2*sb+1 for sb in sigma_beta]))
            unique_int = (int_alpha * n_choose_beta) + int_beta
            basis_set[sigma] = unique_int

    #print(f"N qubits: {n}")
    #print(f"Min int: {min(basis_set.values())}")
    #print(f"Max int: {max(basis_set.values())}")
    #print(f"Length int: {len(basis_set.values())}")

    qu_op_dict = Main.get_qubit_op(ferm_op.terms, basis_set, n)

    qu_op = QubitOperator()
    qu_op.terms = qu_op_dict

    return qu_op


def compute2(tM):
    m00, m01, m10, m11 = tM.get((0, 0), 0.), tM.get((0, 1), 0.), tM.get((1, 0), 0.), tM.get((1, 1), 0.)
    res = {0: 0.5 * (m00 + m11), 1: 0.5 * (m01 + m10), 2: 0.5 * (m00 - m11), 3: 0.5j * (m01 - m10)}
    return res


def agglomerate(values, ops):
    res = dict()
    for i in range(4):
        if values[i] is not None:
            d = tensor_product_pauli_dicts(values[i], ops[i])
            for (k, v) in d.items(): res[k] = res.get(k, 0.) + v
    res = {k: v for k, v in res.items() if v != 0}
    return res

#@profile
def combinatorial5(ferm_op, n_modes, n_electrons):

    t1_quop = time.time()
    # The chemist ordering splits some 1-body and 2-body terms.
    ferm_op_chemist = chemist_ordered(ferm_op)

    # Specify the number of alpha and beta electrons.
    if isinstance(n_electrons, tuple) and len(n_electrons) == 2:
        n_alpha, n_beta = n_electrons
    elif isinstance(n_electrons, int) and n_electrons % 2 == 0:
        n_alpha = n_beta = n_electrons // 2
    else:
        raise ValueError(f"{n_electrons} is not a valid entry for n_electrons, must be a tuple or an int.")

    # Get the number of qubits n.
    n_choose_alpha = comb(n_modes, n_alpha, exact=True)
    n_choose_beta = comb(n_modes, n_beta, exact=True)
    n = math.ceil(np.log2(n_choose_alpha * n_choose_beta))
    print(f"[combinatorial] n_qubits = {n}")

    # Construct the basis set where each configuration is mapped to a unique integer.
    basis_set_alpha = basis(n_modes, n_alpha)
    basis_set_beta = basis(n_modes, n_beta)
    basis_set = OrderedDict()
    for sigma_alpha, int_alpha in basis_set_alpha.items():
        for sigma_beta, int_beta in basis_set_beta.items():
            # Alternate ordering (like FermionOperator in Openfermion).
            sigma = tuple(sorted([2*sa for sa in sigma_alpha] + [2*sb+1 for sb in sigma_beta]))
            unique_int = (int_alpha * n_choose_beta) + int_beta
            basis_set[sigma] = unique_int

    quop_matrix = dict()
    cte = ferm_op_chemist.terms.pop(tuple()) if ferm_op_chemist.constant else 0.
    n_basis = len(basis_set)
    confs, ints = list(zip(*basis_set.items())) # No need for these, we can draw them one by one

    # Get the effect of each operator to the basis set items.
    for i in range(n_basis):
        conf, unique_int = confs[i], ints[i]

        filtered_ferm_op = {k: v for (k, v) in ferm_op_chemist.terms.items() if k[-1][0] in conf}
        for (term, coeff) in filtered_ferm_op.items():
            new_state, phase = one_body_op_on_state(term[-2:], conf)

            if len(term) == 4 and new_state:
                new_state, phase_two = one_body_op_on_state(term[:2], new_state)
                phase *= phase_two

            if not new_state:
                continue

            new_unique_int = basis_set[new_state]
            quop_matrix[(unique_int, new_unique_int)] = quop_matrix.get((unique_int, new_unique_int), 0.) + phase*coeff

    # Valentin: quop is a Dict[(int, int) -> complex]
    print(f'combinatorial5 :: quop dict size {len(quop_matrix)} \t (memory :: {sys.getsizeof(quop_matrix)//10**6} Mbytes)')
    t2_quop = time.time()
    print(f"quop built (elapsed : {t2_quop - t1_quop}s)")

    # Converts matrix back into qubit operator object
    gsize = 2**n # total size
    get_tensor_ops = {2**k: prep(2**k) for k in range(1, n+1)}

    # Split data across all 2x2 matrices
    t1_ground = time.time()
    t = dict()
    t[2] = dict()

    for ((x, y), v) in quop_matrix.items():
        xl, yl, xr, yr = x//2, y//2, x%2, y%2
        t[2][xl, yl] = t[2].get((xl, yl), dict())
        t[2][xl, yl][(xr, yr)] = v

    quop_matrix.clear() #del quop_matrix; gc.collect()
    t2_ground = time.time()
    print(f"Ground level built (elapsed : {t2_ground - t1_ground}s)")

    # Agglomerate lowermost level
    t1_l2 = time.time()
    t[4] = dict()
    ops = get_tensor_ops[4]

    for (x, y) in t[2]:
        xn, yn = x//2, y//2
        if (xn, yn) not in t[4]:

            xl, yl = 2*xn, 2*yn
            values = []
            for xy in [(xl, yl), (xl + 1, yl + 1), (xl, yl + 1), (xl + 1, yl)]:
                v = t[2].get(xy, None)
                v = compute2(v) if v else None
                values.append(v)
            if values != [None] * 4:
                t[4][x // 2, y // 2] = agglomerate(values, ops)

    t2_l2 = time.time()
    print(f"Level 4 built (elapsed : {t2_l2 - t1_l2: 8.1f}s)")

    # Agglomerate levels above iteratively
    t[2].clear() #del t[2]; gc.collect()
    l, s = 8, 2**(n-2)

    while l <= gsize:
        t1_level = time.time()
        t[l] = dict()
        ops = get_tensor_ops[l]

        for (x, y) in t[l//2]:
            xn, yn = x // 2, y // 2
            if (xn, yn) not in t[l]:

                xl, yl = 2*xn, 2*yn
                values = []
                for xy in [(xl, yl), (xl + 1, yl + 1), (xl, yl + 1), (xl + 1, yl)]:
                    values.append(t[l//2].get(xy, None))
                if values != [None] * 4:
                    t[l][x // 2, y // 2] = agglomerate(values, ops)

        # Next iteration
        t[l // 2].clear()  # del t[l//2]; gc.collect()
        l, s = 2 * l, s // 2

        t2_level = time.time()
        print(f"Level {l:7d} built (elapsed : {t2_level - t1_level: 8.1f}s)")

    quop_ints = t[l//2][(0, 0)]

    # Construct final operator
    t1 = time.time()
    quop = QubitOperator()
    for (term, coeff) in quop_ints.items():
        coeff = coeff.real if abs(coeff.imag < ZERO_TOLERANCE) else coeff
        if not (abs(coeff) < ZERO_TOLERANCE):
            t = int_to_tuple2(term, n)
            quop.terms[t] = coeff
    quop.terms[tuple()] = quop.terms.get(tuple(), 0.) + cte

    t2 = time.time()
    print(f"Final operator built (elapsed : {t2 - t1: 8.1f}s)")

    return quop

#@profile
def combinatorial4(ferm_op, n_modes, n_electrons):

    t1_quop = time.time()
    # The chemist ordering splits some 1-body and 2-body terms.
    ferm_op_chemist = chemist_ordered(ferm_op)

    # Specify the number of alpha and beta electrons.
    if isinstance(n_electrons, tuple) and len(n_electrons) == 2:
        n_alpha, n_beta = n_electrons
    elif isinstance(n_electrons, int) and n_electrons % 2 == 0:
        n_alpha = n_beta = n_electrons // 2
    else:
        raise ValueError(f"{n_electrons} is not a valid entry for n_electrons, must be a tuple or an int.")

    # Get the number of qubits n.
    n_choose_alpha = comb(n_modes, n_alpha, exact=True)
    n_choose_beta = comb(n_modes, n_beta, exact=True)
    n = math.ceil(np.log2(n_choose_alpha * n_choose_beta))
    print(f"[combinatorial] n_qubits = {n}")

    # Construct the basis set where each configuration is mapped to a unique integer.
    basis_set_alpha = basis(n_modes, n_alpha)
    basis_set_beta = basis(n_modes, n_beta)
    basis_set = OrderedDict()
    for sigma_alpha, int_alpha in basis_set_alpha.items():
        for sigma_beta, int_beta in basis_set_beta.items():
            # Alternate ordering (like FermionOperator in Openfermion).
            sigma = tuple(sorted([2*sa for sa in sigma_alpha] + [2*sb+1 for sb in sigma_beta]))
            unique_int = (int_alpha * n_choose_beta) + int_beta
            basis_set[sigma] = unique_int

    quop_matrix = dict()
    cte = ferm_op_chemist.terms.pop(tuple()) if ferm_op_chemist.constant else 0.
    n_basis = len(basis_set)
    confs, ints = list(zip(*basis_set.items())) # No need for these, we can draw them one by one

    # Get the effect of each operator to the basis set items.
    for i in range(n_basis):
        conf, unique_int = confs[i], ints[i]

        filtered_ferm_op = {k: v for (k, v) in ferm_op_chemist.terms.items() if k[-1][0] in conf}
        for (term, coeff) in filtered_ferm_op.items():
            new_state, phase = one_body_op_on_state(term[-2:], conf)

            if len(term) == 4 and new_state:
                new_state, phase_two = one_body_op_on_state(term[:2], new_state)
                phase *= phase_two

            if not new_state:
                continue

            new_unique_int = basis_set[new_state]
            quop_matrix[(unique_int, new_unique_int)] = quop_matrix.get((unique_int, new_unique_int), 0.) + phase*coeff

    # Valentin: quop is a Dict[(int, int) -> complex]
    print(f'combinatorial4 :: quop dict size {len(quop_matrix)} \t (memory :: {sys.getsizeof(quop_matrix)//10**6} Mbytes)')
    t2_quop = time.time()
    print(f"quop built (elapsed : {t2_quop - t1_quop}s)")

    # Converts matrix back into qubit operator object
    gsize = 2**n # total size
    get_tensor_ops = {2**k: prep(2**k) for k in range(1, n+1)}

    t1_ground = time.time()
    # Split data across all 2x2 matrices
    t = dict()
    t[2] = dict()

    for ((x, y), v) in quop_matrix.items():
        xl, yl, xr, yr = x//2, y//2, x%2, y%2

        #t[2][(xl, yl)][(xr, yr)] = v
        # Version that does not require initialization of all empty dicts
        t[2][xl, yl] = t[2].get((xl, yl), dict())
        t[2][xl, yl][(xr, yr)] = v

    quop_matrix.clear() #del quop_matrix; gc.collect()
    t2_ground = time.time()
    print(f"Ground level built (elapsed : {t2_ground - t1_ground}s)")

    # Agglomerate lowermost level
    t1_l2 = time.time()
    t[4] = dict()
    ops = get_tensor_ops[4]
    for (x, y) in product(range(0, 2**(n-1), 2), range(0, 2**(n-1), 2)):
        #values = [compute2(t[2][xy]) for xy in [(x, y), (x+1, y+1), (x, y+1), (x+1, y)]]
        values = []
        for xy in [(x, y), (x + 1, y + 1), (x, y + 1), (x + 1, y)]:
            v = t[2].get(xy, None)
            v = compute2(v) if v else None
            values.append(v)
        if values != [None] * 4:
            t[4][x//2, y//2] = agglomerate(values, ops)
    t2_l2 = time.time()
    print(f"Level 4 built (elapsed : {t2_l2 - t1_l2: 8.1f}s)")

    # def agglo_wrapper1(x, y):
    #     values = [compute2(t[2][xy]) for xy in [(x, y), (x + 1, y + 1), (x, y + 1), (x + 1, y)]]
    #     t[4][x // 2, y // 2] = agglomerate(values, ops)
    #
    # pool = Pool(4)
    # pool.starmap(agglomerate, product(range(0, 2**(n-1), 2), range(0, 2**(n-1), 2)), chunksize=1)
    # agglo2 = partial(agglomerate_level2, t_in=t[2], t_out=t[4], ops=ops)
    # datas = list(product(range(0, 2**(n-1), 2), range(0, 2**(n-1), 2)))
    # pool.map(agglo2, datas)

    # Agglomerate levels above iteratively
    t[2].clear() #del t[2]; gc.collect()
    l, s = 8, 2**(n-2)

    while l <= gsize:
        t1_level = time.time()
        #t[l] = {(xl, yl): dict() for (xl, yl) in product(range(0, s//2), range(0, s//2))}
        t[l] = dict()
        ops = get_tensor_ops[l]

        for (x, y) in product(range(0, s, 2), range(0, s, 2)):
            #values = [t[l//2][xy] for xy in [(x, y), (x + 1, y + 1), (x, y + 1), (x + 1, y)]]
            values = []
            for xy in [(x, y), (x + 1, y + 1), (x, y + 1), (x + 1, y)]:
                values.append(t[l//2].get(xy, None))
            if values != [None] * 4:
                t[l][x // 2, y // 2] = agglomerate(values, ops)

        # Next iteration
        t[l // 2].clear()  # del t[l//2]; gc.collect()
        l, s = 2 * l, s // 2

        t2_level = time.time()
        print(f"Level {l:7d} built (elapsed : {t2_level - t1_level: 8.1f}s)")

    quop_ints = t[l//2][(0, 0)]

    # Construct final operator
    t1 = time.time()
    quop = QubitOperator()
    for (term, coeff) in quop_ints.items():
        coeff = coeff.real if abs(coeff.imag < ZERO_TOLERANCE) else coeff
        if not (abs(coeff) < ZERO_TOLERANCE):
            t = int_to_tuple(term, n)
            quop.terms[t] = coeff
    quop.terms[tuple()] = quop.terms.get(tuple(), 0.) + cte

    t2 = time.time()
    print(f"Final operator built (elapsed : {t2 - t1: 8.1f}s)")

    return quop

#@profile
def combinatorial_dict(ferm_op, n_modes, n_electrons):

    t1_quop = time.time()

    # The chemist ordering splits some 1-body and 2-body terms.
    ferm_op_chemist = chemist_ordered(ferm_op)

    # Specify the number of alpha and beta electrons.
    if isinstance(n_electrons, tuple) and len(n_electrons) == 2:
        n_alpha, n_beta = n_electrons
    elif isinstance(n_electrons, int) and n_electrons % 2 == 0:
        n_alpha = n_beta = n_electrons // 2
    else:
        raise ValueError(f"{n_electrons} is not a valid entry for n_electrons, must be a tuple or an int.")

    # Get the number of qubits n.
    n_choose_alpha = comb(n_modes, n_alpha, exact=True)
    n_choose_beta = comb(n_modes, n_beta, exact=True)
    n = math.ceil(np.log2(n_choose_alpha * n_choose_beta))
    print(f"[combinatorial dict] n_qubits = {n}")

    # Construct the basis set where each configuration is mapped to a unique integer.
    basis_set_alpha = basis(n_modes, n_alpha)
    basis_set_beta = basis(n_modes, n_beta)
    basis_set = OrderedDict()
    for sigma_alpha, int_alpha in basis_set_alpha.items():
        for sigma_beta, int_beta in basis_set_beta.items():
            # Alternate ordering (like FermionOperator in Openfermion).
            sigma = tuple(sorted([2*sa for sa in sigma_alpha] + [2*sb+1 for sb in sigma_beta]))
            unique_int = (int_alpha * n_choose_beta) + int_beta
            basis_set[sigma] = unique_int

    quop_matrix = dict()
    cte = ferm_op_chemist.terms.pop(tuple()) if ferm_op_chemist.constant else 0.
    n_basis = len(basis_set)
    confs, ints = list(zip(*basis_set.items())) # No need for these, we can draw them one by one
    max_int = max(ints)
    n_terms = len(ferm_op_chemist.terms)

    # Get the effect of each operator to the basis set items.
    for i in range(n_basis):
        conf, unique_int = confs[i], ints[i]

        filtered_ferm_op = {k: v for (k, v) in ferm_op_chemist.terms.items() if k[-1][0] in conf}
        for (term, coeff) in filtered_ferm_op.items():
            new_state, phase = one_body_op_on_state(term[-2:], conf)

            if len(term) == 4 and new_state:
                new_state, phase_two = one_body_op_on_state(term[:2], new_state)
                phase *= phase_two

            if not new_state:
                continue

            new_unique_int = basis_set[new_state]
            quop_matrix[(unique_int, new_unique_int)] = quop_matrix.get((unique_int, new_unique_int), 0.) + phase*coeff

    # Print size of dict (should be lower than dense array)
    print(f'combinatorial dict :: quop dict size {len(quop_matrix)} \t (memory :: {sys.getsizeof(quop_matrix)//10**6} Mbytes)')
    t2_quop = time.time()
    print(f"quop built (elapsed : {t2_quop - t1_quop}s)")

    # Converts matrix back into qubit operator object
    quop_ints = recursive_mapping_dict(quop_matrix, 2**n, (0,0))
    quop = QubitOperator()
    for (term, coeff) in quop_ints.items():
        coeff = coeff.real if abs(coeff.imag < ZERO_TOLERANCE) else coeff
        if not (abs(coeff) < ZERO_TOLERANCE):
            t = int_to_tuple2(term, n)
            quop.terms[t] = coeff
    quop.terms[tuple()] = quop.terms.get(tuple(), 0.) + cte

    return quop

def combinatorial_sp(ferm_op, n_modes, n_electrons):
    """ Function to transform the fermionic Hamiltonian into a basis constructed
    in the Fock space.

    Args:
        ferm_op (FermionOperator). Fermionic operator, with alternate ordering
            as followed in the openfermion package
        n_modes (int): Number of relevant molecular orbitals, i.e. active molecular
            orbitals.
        n_electrons (int): Number of active electrons.

    Returns:
        QubitOperator: Self-explanatory.
    """

    # The chemist ordering splits some 1-body and 2-body terms.
    ferm_op_chemist = chemist_ordered(ferm_op)

    # Specify the number of alpha and beta electrons.
    if isinstance(n_electrons, tuple) and len(n_electrons) == 2:
        n_alpha, n_beta = n_electrons
    elif isinstance(n_electrons, int) and n_electrons % 2 == 0:
        n_alpha = n_beta = n_electrons // 2
    else:
        raise ValueError(f"{n_electrons} is not a valid entry for n_electrons, must be a tuple or an int.")

    # Get the number of qubits n.
    n_choose_alpha = comb(n_modes, n_alpha, exact=True)
    n_choose_beta = comb(n_modes, n_beta, exact=True)
    n = math.ceil(np.log2(n_choose_alpha * n_choose_beta))
    print(f"[combinatorial] n_qubits = {n}")

    # Construct the basis set where each configuration is mapped to a unique integer.
    basis_set_alpha = basis(n_modes, n_alpha)
    basis_set_beta = basis(n_modes, n_beta)
    basis_set = OrderedDict()
    for sigma_alpha, int_alpha in basis_set_alpha.items():
        for sigma_beta, int_beta in basis_set_beta.items():
            # Alternate ordering (like FermionOperator in OpenFermion).
            sigma = tuple(sorted([2*sa for sa in sigma_alpha] + [2*sb+1 for sb in sigma_beta]))
            unique_int = (int_alpha * n_choose_beta) + int_beta
            basis_set[sigma] = unique_int

    quop_matrix = np.zeros((2**n, 2**n), dtype=np.complex64)
    cte = ferm_op_chemist.terms.pop(tuple()) if ferm_op_chemist.constant else 0.
    n_basis = len(basis_set)
    confs, ints = list(zip(*basis_set.items()))

    # Get the effect of each operator to the basis set items.
    for i in range(n_basis):
        conf, unique_int = confs[i], ints[i]

        filtered_ferm_op = {k: v for (k, v) in ferm_op_chemist.terms.items() if k[-1][0] in conf}
        for (term, coeff) in filtered_ferm_op.items():
            new_state, phase = one_body_op_on_state(term[-2:], conf)

            if len(term) == 4 and new_state:
                new_state, phase_two = one_body_op_on_state(term[:2], new_state)
                phase *= phase_two

            if not new_state:
                continue

            new_unique_int = basis_set[new_state]
            quop_matrix[unique_int, new_unique_int] += phase*coeff

    # Convert dense matrix to sparse matrix, delete dense matrix. Quantify reduction in memory
    nz = np.count_nonzero(quop_matrix)
    print(f'Dense matrix = {nz} ({100*nz/4**n:5.2f}%) \t (memory :: {quop_matrix.nbytes/(10**6)} Mbytes)')
    quop_matrix_sp = lil_matrix(quop_matrix)
    del quop_matrix; gc.collect()
    print(f'Sparse matrix (memory :: {quop_matrix_sp.dtype.nbyte*quop_matrix_sp.size/(10**6)} Mbytes)')

    # Convert matrix back into qubit operator object
    quop_ints = recursive_mapping(quop_matrix_sp)
    quop = QubitOperator()
    for (term, coeff) in quop_ints.items():
        coeff = coeff.real if abs(coeff.imag < ZERO_TOLERANCE) else coeff
        if not (abs(coeff) < ZERO_TOLERANCE):
            quop.terms[int_to_tuple(term, n)] = coeff
    quop.terms[tuple()] = quop.terms.get(tuple(), 0.) + cte

    return quop


def combinatorial(ferm_op, n_modes, n_electrons):
    """ Function to transform the fermionic Hamiltonian into a basis constructed
    in the Fock space.

    Args:
        ferm_op (FermionOperator). Fermionic operator, with alternate ordering
            as followed in the openfermion package
        n_modes (int): Number of relevant molecular orbitals, i.e. active molecular
            orbitals.
        n_electrons (int): Number of active electrons.

    Returns:
        QubitOperator: Self-explanatory.
    """

    t1_quop = time.time()

    # The chemist ordering splits some 1-body and 2-body terms.
    ferm_op_chemist = chemist_ordered(ferm_op)

    # Specify the number of alpha and beta electrons.
    if isinstance(n_electrons, tuple) and len(n_electrons) == 2:
        n_alpha, n_beta = n_electrons
    elif isinstance(n_electrons, int) and n_electrons % 2 == 0:
        n_alpha = n_beta = n_electrons // 2
    else:
        raise ValueError(f"{n_electrons} is not a valid entry for n_electrons, must be a tuple or an int.")

    # Get the number of qubits n.
    n_choose_alpha = comb(n_modes, n_alpha, exact=True)
    n_choose_beta = comb(n_modes, n_beta, exact=True)
    n = math.ceil(np.log2(n_choose_alpha * n_choose_beta))
    print(f"[combinatorial] n_qubits = {n}")

    # Construct the basis set where each configuration is mapped to a unique integer.
    basis_set_alpha = basis(n_modes, n_alpha)
    basis_set_beta = basis(n_modes, n_beta)
    basis_set = OrderedDict()
    for sigma_alpha, int_alpha in basis_set_alpha.items():
        for sigma_beta, int_beta in basis_set_beta.items():
            # Alternate ordering (like FermionOperator in OpenFermion).
            sigma = tuple(sorted([2*sa for sa in sigma_alpha] + [2*sb+1 for sb in sigma_beta]))
            unique_int = (int_alpha * n_choose_beta) + int_beta
            basis_set[sigma] = unique_int

    quop_matrix = np.zeros((2**n, 2**n), dtype=np.complex64)
    cte = ferm_op_chemist.terms.pop(tuple()) if ferm_op_chemist.constant else 0.
    n_basis = len(basis_set)
    confs, ints = list(zip(*basis_set.items()))

    # Get the effect of each operator to the basis set items.
    for i in range(n_basis):
        conf, unique_int = confs[i], ints[i]

        filtered_ferm_op = {k: v for (k, v) in ferm_op_chemist.terms.items() if k[-1][0] in conf}
        for (term, coeff) in filtered_ferm_op.items():
            new_state, phase = one_body_op_on_state(term[-2:], conf)

            if len(term) == 4 and new_state:
                new_state, phase_two = one_body_op_on_state(term[:2], new_state)
                phase *= phase_two

            if not new_state:
                continue

            new_unique_int = basis_set[new_state]
            quop_matrix[unique_int, new_unique_int] += phase*coeff

    # Valentin: quantify how much memory dense matrix is consuming and how sparse it is.
    t2_quop = time.time()
    print(f"quop built (elapsed : {t2_quop - t1_quop}s)")
    nz = np.count_nonzero(quop_matrix)
    print(f'Dense matrix = {nz} ({100*nz/4**n:5.2f}%) \t (memory :: {quop_matrix.nbytes/(10**6)} Mbytes)')

    # Convert matrix back into qubit operator object
    quop_ints = recursive_mapping(quop_matrix)
    quop = QubitOperator()
    for (term, coeff) in quop_ints.items():
        coeff = coeff.real if abs(coeff.imag < ZERO_TOLERANCE) else coeff
        if not (abs(coeff) < ZERO_TOLERANCE):
            quop.terms[int_to_tuple(term, n)] = coeff
    quop.terms[tuple()] = quop.terms.get(tuple(), 0.) + cte

    return quop


def basis(M, N):
    """ Function to construct the combinatorial basis set, i.e. a basis set
    respecting the number of electrons and the total spin.

    Args:
        M (int): Number of spatial orbitals.
        N (int): Number of alpha or beta electrons.

    Returns:
        OrderedDict: Lexicographically sorted basis set, mapping electronic
            configuration to unique integers.
    """

    mapping = [(sigma, conf_to_integer(sigma, M)) for sigma in itertools.combinations(range(M), N)]
    return OrderedDict(mapping)


def conf_to_integer(sigma, M):
    """ Function to map an electronic configuration to a unique integer, as done
    in arXiv.2205.11742 eq. (14).

    Args:
        sigma (tuple of int): Orbital indices where the electron are in the
            electronic configuration.
        M (int): Number of modes, i.e. number of spatial orbitals.

    Returns:
        int: Unique integer for the input electronic state.
    """

    # Equivalent to the number of electrons.
    N = len(sigma)

    # Eq. (14) in the reference.
    terms_k = [comb(M - sigma[N - 1 - k] - 1, k + 1) for k in range(N)]
    unique_int = comb(M, N) - 1 - np.sum(terms_k)

    return int(unique_int)
