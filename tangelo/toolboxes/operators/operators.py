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

"""This module defines various kinds of operators used in vqe. It can later be
broken down in several modules if needed.
"""

from math import sqrt
from collections import OrderedDict
from copy import deepcopy

import numpy as np
from scipy.special import comb
import openfermion as of

from tangelo.helpers import assert_freq_dict_almost_equal
from tangelo.toolboxes.molecular_computation.coefficients import spatial_from_spinorb

COEFFICIENT_TYPES = (int, float, complex, np.integer, np.floating)


# Define products of all Pauli operators for symbolic multiplication.
_PAULI_OPERATOR_PRODUCTS = {
    ('I', 'I'): (1., 'I'),
    ('I', 'X'): (1., 'X'),
    ('X', 'I'): (1., 'X'),
    ('I', 'Y'): (1., 'Y'),
    ('Y', 'I'): (1., 'Y'),
    ('I', 'Z'): (1., 'Z'),
    ('Z', 'I'): (1., 'Z'),
    ('X', 'X'): (1., 'I'),
    ('Y', 'Y'): (1., 'I'),
    ('Z', 'Z'): (1., 'I'),
    ('X', 'Y'): (1.j, 'Z'),
    ('X', 'Z'): (-1.j, 'Y'),
    ('Y', 'X'): (-1.j, 'Z'),
    ('Y', 'Z'): (1.j, 'X'),
    ('Z', 'X'): (1.j, 'Y'),
    ('Z', 'Y'): (-1.j, 'X')
}


class FermionOperator(of.FermionOperator):
    """Custom FermionOperator class. Based on openfermion's, with additional functionalities.
    """

    def __init__(self, term=None, coefficient=1., n_spinorbitals=None, n_electrons=None, spin=None):
        super(FermionOperator, self).__init__(term, coefficient)

        self.n_spinorbitals = n_spinorbitals
        self.n_electrons = n_electrons
        self.spin = spin

    def __imul__(self, other):
        if isinstance(other, FermionOperator):
            # Raise error if attributes are not the same across Operators.
            if (self.n_spinorbitals, self.n_electrons, self.spin) != (other.n_spinorbitals, other.n_electrons, other.spin):
                raise RuntimeError("n_spinorbitals, n_electrons and spin must be the same for all FermionOperators.")
            else:
                return super(FermionOperator, self).__imul__(other)

        elif isinstance(other, of.FermionOperator):
            if (self.n_spinorbitals, self.n_electrons, self.spin) != (None, None, None):
                raise RuntimeError("openfermion FermionOperator did not define a necessary attribute")
            else:
                f_op = FermionOperator()
                f_op.terms = other.terms.copy()
                return super(FermionOperator, self).__imul__(f_op)

        else:
            return super(FermionOperator, self).__imul__(other)

    def __mul__(self, other):
        return self.__imul__(other)

    def __iadd__(self, other):
        if isinstance(other, FermionOperator):
            # Raise error if attributes are not the same across Operators.
            if (self.n_spinorbitals, self.n_electrons, self.spin) != (other.n_spinorbitals, other.n_electrons, other.spin):
                raise RuntimeError("n_spinorbitals, n_electrons and spin must be the same for all FermionOperators.")
            else:
                return super(FermionOperator, self).__iadd__(other)

        elif isinstance(other, of.FermionOperator):
            if (self.n_spinorbitals, self.n_electrons, self.spin) != (None, None, None):
                raise RuntimeError("openfermion FermionOperator did not define a necessary attribute")
            else:
                f_op = FermionOperator()
                f_op.terms = other.terms.copy()
                return super(FermionOperator, self).__iadd__(f_op)

        elif isinstance(other, COEFFICIENT_TYPES):
            self.constant += other
            return self

        else:
            raise RuntimeError(f"You cannot add FermionOperator and {other.__class__}.")

    def __add__(self, other):
        return self.__iadd__(other)

    def __radd__(self, other):
        return self.__iadd__(other)

    def __isub__(self, other):
        return self.__iadd__(-1. * other)

    def __sub__(self, other):
        return self.__isub__(other)

    def __rsub__(self, other):
        return -1 * self.__isub__(other)

    def __eq__(self, other):
        # Additional checks for == operator.
        if isinstance(other, FermionOperator):
            if (self.n_spinorbitals, self.n_electrons, self.spin) == (other.n_spinorbitals, other.n_electrons, other.spin):
                return super(FermionOperator, self).__eq__(other)
            else:
                return False
        else:
            return super(FermionOperator, self).__eq__(other)

    def get_coeffs(self, coeff_threshold=1e-8, spatial=False):
        """Method to get the coefficient tensors from a fermion operator.

        Args:
            coeff_threshold (float): Ignore coefficient below the threshold.
                Default value is 1e-8.
            spatial (bool): Spatial orbital or spin orbital.

        Returns:
            (float, array float, array of float): Core constant, one- (N*N) and
                two-body coefficient matrices (N*N*N*N), where N is the number
                of spinorbitals or spatial orbitals.
        """
        n_sos = of.count_qubits(self)

        constant = 0.
        one_body = np.zeros((n_sos, n_sos), complex)
        two_body = np.zeros((n_sos, n_sos, n_sos, n_sos), complex)

        # Loop through terms and assign to matrix.
        for term in self.terms:
            coefficient = self.terms[term]

            # Ignore this term if the coefficient is zero
            if abs(coefficient) < coeff_threshold:
                continue

            # Handle constant shift.
            if len(term) == 0:
                constant = coefficient
            # Handle one-body terms.
            elif len(term) == 2:
                if [operator[1] for operator in term] == [1, 0]:
                    p, q = [operator[0] for operator in term]
                    one_body[p, q] = coefficient
            # Handle two-body terms.
            elif len(term) == 4:
                if [operator[1] for operator in term] == [1, 1, 0, 0]:
                    p, q, r, s = [operator[0] for operator in term]
                    two_body[p, q, r, s] = coefficient

        if spatial:
            one_body, two_body = spatial_from_spinorb(one_body, two_body)

        return constant, one_body, two_body

    def to_openfermion(self):
        """Converts Tangelo FermionOperator to openfermion"""
        ferm_op = of.FermionOperator()
        ferm_op.terms = self.terms.copy()
        return ferm_op


class BosonOperator(of.BosonOperator):
    """Currently, this class is coming from openfermion. Can be later on be
    replaced by our own implementation.
    """
    pass


class QubitOperator(of.QubitOperator):
    """Currently, this class is coming from openfermion. Can be later on be
    replaced by our own implementation.
    """

    @property
    def n_terms(self):
        """ Return the number of terms present in the operator

        Returns:
            integer: self-descriptive
        """
        return len(self.terms)

    @property
    def n_qubits(self):
        """ Return the number of qubits present in the operator as the largest qubit index present

        Returns:
            integer: self-descriptive
        """
        return count_qubits(self)

    @property
    def qubit_indices(self):
        """ Return a set of integers corresponding to qubit indices the qubit operator acts on.

        Returns:
            set: Set of qubit indices.
        """

        qubit_indices = set()
        for term in self.terms:
            if term:
                indices = list(zip(*term))[0]
                qubit_indices.update(indices)

        return qubit_indices

    @classmethod
    def from_openfermion(cls, of_qop):
        """ Enable instantiation of a QubitOperator from an openfermion QubitOperator object.

        Args:
            of_qop (openfermion QubitOperator): an existing qubit operator defined with Openfermion

        Returns:
            corresponding QubitOperator object.
        """
        qop = cls()
        qop.terms = of_qop.terms.copy()
        return qop

    def frobenius_norm_compression(self, epsilon, n_qubits):
        """Reduces the number of operator terms based on its Frobenius norm
        and a user-defined threshold, epsilon. The eigenspectrum of the
        compressed operator will not deviate more than epsilon. For more
        details, see J. Chem. Theory Comput. 2020, 16, 2, 1055-1063.

        Args:
            epsilon (float): Parameter controlling the degree of compression
                and resulting accuracy.
            n_qubits (int): Number of qubits in the register.

        Returns:
            QubitOperator: The compressed qubit operator.
        """

        compressed_op = dict()
        coef2_sum = 0.
        frob_factor = 2**(n_qubits // 2)

        # Arrange the terms of the qubit operator in ascending order
        self.terms = OrderedDict(sorted(self.terms.items(), key=lambda x: abs(x[1]), reverse=False))

        for term, coef in self.terms.items():
            coef2_sum += abs(coef)**2
            # while the sum is less than epsilon / factor, discard the terms
            if sqrt(coef2_sum) > epsilon / frob_factor:
                compressed_op[term] = coef
        self.terms = compressed_op
        self.compress()

    def get_max_number_hamiltonian_terms(self, n_qubits):
        """Compute the possible number of terms for a qubit Hamiltonian. In the
        absence of an external magnetic field, each Hamiltonian term must have
        an even number of Pauli Y operators to preserve time-reversal symmetry.
        See J. Chem. Theory Comput. 2020, 16, 2, 1055-1063 for more details.

        Args:
            n_qubits (int): Number of qubits in the register.

        Returns:
            int: The maximum number of possible qubit Hamiltonian terms.
        """

        return sum([comb(n_qubits, 2*i, exact=True) * 3**(n_qubits-2*i) for i in range(n_qubits//2)])

    def to_openfermion(self):
        """Converts Tangelo QubitOperator to openfermion"""
        qu_op = of.QubitOperator()
        qu_op.terms = self.terms.copy()
        return qu_op

class QubitOperator2(of.QubitOperator):
    """Currently, this class is coming from openfermion. Can be later on be
    replaced by our own implementation.
    """

    def __eq__(self, other):
        if not isinstance(other, (QubitOperator2, of.QubitOperator)):
            return False
        else:
            return assert_freq_dict_almost_equal(self.terms, other.terms, atol=1e-8)

    def __iadd__(self, other):
        if isinstance(other, COEFFICIENT_TYPES):
            self.terms[()] = self.terms.get((), 0.) + other
        elif isinstance(other, self.__class__):
            for k, v in other.terms.items():
                self.terms[k] = self.terms.get(k, 0.) + v
        return self

    def __add__(self, other):
        new = deepcopy(self)
        new += other
        return new

    def __radd__(self, other):
        return self.__add__(other)

    def __isub__(self, other):
        return self.__iadd__(-1. * other)

    def __sub__(self, other):
        return self.__add__(-1 * other)

    def __rsub__(self, other):
        return -1 * self.__isub__(other)

    def _simplify3(self, term, coefficient=1.0):
        """Simplify a term using commutator and anti-commutator relations."""
        if not term:
            return coefficient, term

        term = sorted(term, key=lambda factor: factor[0])

        new_term = []
        l_factor = term[0]
        for r_factor in term[1:]:

            l_index, l_action = l_factor
            r_index, r_action = r_factor

            if l_index != r_index:
                if l_action != 'I':
                    new_term.append(l_factor)
                l_factor = r_factor
            else:
                new_coefficient, new_action = _PAULI_OPERATOR_PRODUCTS[l_action, r_action]
                l_factor = (l_index, new_action)
                coefficient *= new_coefficient

        # Save result of final iteration.
        if l_factor[1] != 'I':
            new_term.append(l_factor)

        return coefficient, tuple(new_term)

    #@profile
    def _simplify2(self, term, coefficient=1.0):
        """Simplify a term using commutator and anti-commutator relations."""
        if not term:
            return coefficient, term

        # Compute new actions and coefficients for each qubit index present
        d_act = dict()
        coef = 1.
        for ind, act in term:
            res = _PAULI_OPERATOR_PRODUCTS[(d_act.get(ind, 'I'), act)]
            coef *= res[0]
            d_act[ind] = res[1]

        # Recast into ordered tuple of (int, str)
        d_act = {k: v for k, v in d_act.items() if v != 'I'}
        t = tuple((i, d_act[i]) for i in sorted(d_act.keys()))

        return coef, t

    #@profile
    def _simplify(self, term, coefficient=1.0):
        """Simplify a term using commutator and anti-commutator relations."""
        if not term:
            return coefficient, term

        term = sorted(term, key=lambda factor: factor[0])

        new_term = []
        l_factor = term[0]
        for r_factor in term[1:]:

            l_index, l_action = l_factor
            r_index, r_action = r_factor

            # Still on the same qubit, keep simplifying.
            if l_index == r_index:
                new_coefficient, new_action = _PAULI_OPERATOR_PRODUCTS[l_action, r_action]
                l_factor = (l_index, new_action)
                coefficient *= new_coefficient

            # Reached different qubit, save result and re-initialize.
            else:
                if l_action != 'I':
                    new_term.append(l_factor)
                l_factor = r_factor

        # Save result of final iteration.
        if l_factor[1] != 'I':
            new_term.append(l_factor)

        return coefficient, tuple(new_term)

    def __mul__(self, other):
        if isinstance(other, COEFFICIENT_TYPES):
            d = {k: v * other for k, v in self.terms.items()}
            qop = QubitOperator2()
            qop.terms = d
            return qop
        elif isinstance(other, self.__class__):
            new_terms = dict()
            for t1, c1 in self.terms.items():
                for t2, c2 in other.terms.items():
                    new_c, new_t = self._simplify(t1+t2, coefficient=c1*c2)
                    new_terms[new_t] = new_terms.get(new_t, 0.) + new_c
            self.terms = new_terms
            return self
        else:
            raise TypeError(f'Cannot multiply type {self.__class__.__name__} with {type(self)}.')

    def __imul__(self, other):
        return self.__mul__(other)

    @property
    def n_terms(self):
        """ Return the number of terms present in the operator

        Returns:
            integer: self-descriptive
        """
        return len(self.terms)

    @property
    def n_qubits(self):
        """ Return the number of qubits present in the operator as the largest qubit index present

        Returns:
            integer: self-descriptive
        """
        return count_qubits(self)

    @property
    def qubit_indices(self):
        """ Return a set of integers corresponding to qubit indices the qubit operator acts on.

        Returns:
            set: Set of qubit indices.
        """

        qubit_indices = set()
        for term in self.terms:
            if term:
                indices = list(zip(*term))[0]
                qubit_indices.update(indices)

        return qubit_indices

    @classmethod
    def from_dict(cls, d):
        """ Enable instantiation of a QubitOperator from a dictionary of terms.

        Args:
            d (Openfermion-style dictionary of terms): dictionary to build the operator from.

        Returns:
            Operator with desired terms
        """
        op = cls()
        op.terms = d.copy()
        return op

    @classmethod
    def from_openfermion(cls, of_qop):
        """ Enable instantiation of a QubitOperator from an openfermion QubitOperator object.

        Args:
            of_qop (openfermion QubitOperator): an existing qubit operator defined with Openfermion

        Returns:
            corresponding QubitOperator object.
        """
        qop = cls()
        qop.terms = of_qop.terms.copy()
        return qop

    def norm(self, ord=1, threshold=1e-10):
        """ Compute the equivalent of a number of numpy norms using the coefficients appearing in the operator.
        (https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html)

        Args:
            ord (Integer | string): Desired norm. Can be a positive integer, "inf" or "-inf".

        Returns:
            float: the desired norm
        """
        if isinstance(ord, int):
            if ord > 0:
                norm = sum(abs(coef)**ord for coef in self.terms.values())
                return norm**(1. / ord)
            elif ord == 0:
                return len([coef for coef in self.terms.values() if abs(coef) > threshold])
            else:
                return ValueError('Allowed values for "ord" are positive integers, "inf" or "-inf"')
        elif isinstance(ord, str):
            if ord == 'inf':
                return max(abs(coef) for coef in self.terms.values())
            elif ord == '-inf':
                return min(abs(coef) for coef in self.terms.values())
            else:
                return ValueError('Allowed values for "ord" are positive integers, "inf" or "-inf"')
        else:
            return ValueError('Allowed values for "ord" are positive integers, "inf" or "-inf"')

    def frobenius_norm_compression(self, epsilon, n_qubits):
        """Reduces the number of operator terms based on its Frobenius norm
        and a user-defined threshold, epsilon. The eigenspectrum of the
        compressed operator will not deviate more than epsilon. For more
        details, see J. Chem. Theory Comput. 2020, 16, 2, 1055-1063.

        Args:
            epsilon (float): Parameter controlling the degree of compression
                and resulting accuracy.
            n_qubits (int): Number of qubits in the register.

        Returns:
            QubitOperator: The compressed qubit operator.
        """

        frob_factor = 2**(n_qubits // 2)
        threshold = epsilon / frob_factor

        # Arrange the terms of the qubit operator in ascending order
        # VS Why make a new dictionary when sorted already returns a list ?
        self.terms = OrderedDict(sorted(self.terms.items(), key=lambda x: abs(x[1]), reverse=False))

        compressed_op = dict()
        coef2_sum = 0.
        for term, coef in self.terms.items():
            coef2_sum += abs(coef)**2
            # while the sum is less than epsilon / factor, discard the terms
            if sqrt(coef2_sum) > threshold:
                compressed_op[term] = coef
        self.terms = compressed_op
        self.compress()

    def get_max_number_hamiltonian_terms(self, n_qubits):
        """Compute the possible number of terms for a qubit Hamiltonian. In the
        absence of an external magnetic field, each Hamiltonian term must have
        an even number of Pauli Y operators to preserve time-reversal symmetry.
        See J. Chem. Theory Comput. 2020, 16, 2, 1055-1063 for more details.

        Args:
            n_qubits (int): Number of qubits in the register.

        Returns:
            int: The maximum number of possible qubit Hamiltonian terms.
        """

        return sum([comb(n_qubits, 2*i, exact=True) * 3**(n_qubits-2*i) for i in range(n_qubits//2)])

    def to_openfermion(self):
        """Converts Tangelo QubitOperator to openfermion"""
        qu_op = of.QubitOperator()
        qu_op.terms = self.terms.copy()
        return qu_op


class QubitHamiltonian(QubitOperator):
    """QubitHamiltonian objects are essentially openfermion.QubitOperator
    objects, with extra attributes. The mapping procedure (mapping) and the
    qubit ordering (up_then_down) are incorporated into the class. In addition
    to QubitOperator, several checks are done when performing arithmetic
    operations on QubitHamiltonians.

    Attributes:
        term (openfermion-like): Same as openfermion term formats.
        coefficient (complex): Coefficient for this term.
        mapping (string): Mapping procedure for fermionic to qubit encoding
            (ex: "JW", "BK", etc.).
        up_then_down (bool): Whether spin ordering is all up then all down.

    Properties:
        n_terms (int): Number of terms in this qubit Hamiltonian.
    """

    def __init__(self, term=None, coefficient=1., mapping=None, up_then_down=None):
        super(QubitOperator, self).__init__(term, coefficient)

        self.mapping = mapping
        self.up_then_down = up_then_down

    @property
    def n_terms(self):
        return len(self.terms)

    @property
    def n_qubits(self):
        return count_qubits(self)

    def __iadd__(self, other_hamiltonian):

        # Raise error if attributes are not the same across Hamiltonians. This
        # check is ignored if comparing to a QubitOperator or a bare
        # QubitHamiltonian.
        if self.mapping is not None and self.up_then_down is not None and \
                                other_hamiltonian.mapping is not None and \
                                other_hamiltonian.up_then_down is not None:

            if self.mapping.upper() != other_hamiltonian.mapping.upper():
                raise RuntimeError("Mapping must be the same for all QubitHamiltonians.")
            elif self.up_then_down != other_hamiltonian.up_then_down:
                raise RuntimeError("Spin ordering must be the same for all QubitHamiltonians.")

        return super(QubitOperator, self).__iadd__(other_hamiltonian)

    def __eq__(self, other_hamiltonian):

        # Additional checks for == operator. This check is ignored if comparing
        # to a QubitOperator or a bare QubitHamiltonian.
        if self.mapping is not None and self.up_then_down is not None and \
                                other_hamiltonian.mapping is not None and \
                                other_hamiltonian.up_then_down is not None:
            if (self.mapping.upper() != other_hamiltonian.mapping.upper()) or (self.up_then_down != other_hamiltonian.up_then_down):
                return False

        return super(QubitOperator, self).__eq__(other_hamiltonian)

    def to_qubitoperator(self):
        qubit_op = QubitOperator()
        qubit_op.terms = self.terms.copy()
        return qubit_op


def count_qubits(qb_op):
    """Return the number of qubits used by the qubit operator based on the
    highest index found in the terms.
    """
    if (len(qb_op.terms.keys()) == 0) or ((len(qb_op.terms.keys()) == 1) and (len(list(qb_op.terms.keys())[0]) == 0)):
        return 0
    else:
        return max([(sorted(pw))[-1][0] for pw in qb_op.terms.keys() if len(pw) > 0]) + 1


def normal_ordered(fe_op):
    """ Input: a Fermionic operator of class
    toolboxes.operators.FermionicOperator or openfermion.FermionicOperator for
    reordering.

    Returns:
        FermionicOperator: Normal ordered operator.
    """

    # Obtain normal ordered fermionic operator as list of terms
    norm_ord_terms = of.transforms.normal_ordered(fe_op).terms

    # Regeneratore full operator using class of tangelo.toolboxes.operators.FermionicOperator
    norm_ord_fe_op = FermionOperator()
    for term in norm_ord_terms:
        norm_ord_fe_op += FermionOperator(term, norm_ord_terms[term])
    return norm_ord_fe_op


def squared_normal_ordered(all_terms):
    """Input: a list of terms to generate toolboxes.operators.FermionOperator
    or openfermion.FermionOperator

    Returns:
        FermionOperator: squared (i.e. fe_op*fe_op) and
            normal ordered.
    """

    # Obtain normal ordered fermionic operator as list of terms
    fe_op = list_to_fermionoperator(all_terms)
    fe_op *= fe_op
    return normal_ordered(fe_op)


def list_to_fermionoperator(all_terms):
    """Input: a list of terms to generate FermionOperator

    Returns:
        FermionOperator: Single merged operator.
    """

    fe_op = FermionOperator()
    for item in all_terms:
        fe_op += FermionOperator(item[0], item[1])
    return fe_op


def qubitop_to_qubitham(qubit_op, mapping, up_then_down):
    """Function to convert a QubitOperator into a QubitHamiltonian.

    Args:
        qubit_op (QubitOperator): Self-explanatory.
        mapping (string): Qubit mapping procedure.
        up_then_down (bool): Whether or not spin ordering is all up then
            all down.

    Returns:
        QubitHamiltonian: Self-explanatory.
    """
    qubit_ham = QubitHamiltonian(mapping=mapping, up_then_down=up_then_down)
    qubit_ham.terms = qubit_op.terms.copy()

    return qubit_ham
