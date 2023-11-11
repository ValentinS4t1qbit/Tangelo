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

import unittest
from copy import deepcopy
from itertools import product
from time import time

import numpy as np
import openfermion as of

from tangelo.toolboxes.operators import QubitHamiltonian, FermionOperator, \
    QubitOperator, count_qubits, qubitop_to_qubitham

from tangelo.toolboxes.operators import QubitOperator2

c = 666.
q1 = 3*QubitOperator2("X0 Y1")
q2 = 2*QubitOperator2("Z0 Z1") + QubitOperator2("X0 Y1")

# Build artificially large operator made of all possible "full" Pauli words (no 'I') of length n_qubits
n_qubits_op = 7
n_terms = 3 ** n_qubits_op
terms = {tuple(zip(range(n_qubits_op), pw)): 1.0 for pw in product(['X', 'Y', 'Z'], repeat=n_qubits_op)}
bq1_ref = QubitOperator()
bq1_ref.terms = terms
bq2_ref = deepcopy(bq1_ref)
bq1, bq2 = QubitOperator2.from_dict(terms), QubitOperator2.from_dict(terms)

class OperatorsUtilitiesTest(unittest.TestCase):

    def test_count_qubits(self):
        """Test count_qubits function."""

        n_qubits_3 = count_qubits(QubitHamiltonian("X0 Y1 Z2", mapping="JW", up_then_down=True))
        self.assertEqual(n_qubits_3, 3)

        n_qubits_5 = count_qubits(QubitHamiltonian("X0 Y1 Z4", mapping="JW", up_then_down=True))
        self.assertEqual(n_qubits_5, 5)

    def test_qubitop_to_qubitham(self):
        """Test qubitop_to_qubitham function."""

        qubit_ham = qubitop_to_qubitham(0.5*QubitOperator("X0 Y1 Z2"), "BK", False)

        reference_attributes = {"terms": {((0, 'X'), (1, 'Y'), (2, 'Z')): 0.5},
                                "mapping": "BK", "up_then_down": False}

        self.assertDictEqual(qubit_ham.__dict__, reference_attributes)


class FermionOperatorTest(unittest.TestCase):

    def test_get_coeff(self):
        """Test the get_coeff method of the FermionOperator class."""

        # Constant.
        dummy_ferm_op = FermionOperator("", 1.)

        # 1-body terms.
        dummy_ferm_op += FermionOperator("0^ 0", 2.)
        dummy_ferm_op += FermionOperator("0^ 1", 3.)
        dummy_ferm_op += FermionOperator("1^ 0", 4.)
        dummy_ferm_op += FermionOperator("1^ 1", 5.)

        # 2-body term.
        dummy_ferm_op += FermionOperator("0^ 0^ 0 0", 7.)
        dummy_ferm_op += FermionOperator("1^ 1^ 1 1", 8.)
        dummy_ferm_op += FermionOperator("0^ 1^ 0 1", 9.)

        ref_cte = 1.
        ref_one_body = np.arange(2, 6, 1.).reshape((2, 2))
        ref_two_body = np.zeros((2, 2, 2, 2))
        ref_two_body[0, 0, 0, 0] = 7.
        ref_two_body[1, 1, 1, 1] = 8.
        ref_two_body[0, 1, 0, 1] = 9.

        cte, one_body, two_body = dummy_ferm_op.get_coeffs()

        self.assertAlmostEqual(ref_cte, cte)
        np.testing.assert_array_almost_equal(ref_one_body, one_body)
        np.testing.assert_array_almost_equal(ref_two_body, two_body)

    def test_eq(self):
        fop_1 = FermionOperator("0^ 0", 2., spin=1)
        fop_2 = of.FermionOperator("0^ 0", 2.)
        fop_3 = FermionOperator("0^ 0", 2., spin=0)
        fop_4 = FermionOperator("0^ 0", 1., spin=0)
        self.assertEqual(fop_1, fop_2)
        self.assertNotEqual(fop_1, fop_3)
        self.assertNotEqual(fop_3, fop_4)

    def test_add(self):
        # addition between two compatible tangelo FermionOperator
        FermionOperator("0^ 0", 2.) + FermionOperator("0^ 1", 3.)

        # addition between two incompatible tangelo FermionOperator
        fop_1 = FermionOperator("0^ 0", 2., spin=1)
        fop_2 = FermionOperator("0^ 1", 3., spin=0)
        with self.assertRaises(RuntimeError):
            fop_1 + fop_2

        # addition between openfermion FermionOperator and Tangelo equivalent
        fop_1 = FermionOperator("0^ 0", 2.) + of.FermionOperator("0^ 1", 3.)
        self.assertTrue(isinstance(fop_1, FermionOperator))

        # Reverse order addition test
        fop_2 = of.FermionOperator("0^ 0", 2.) + FermionOperator("0^ 1", 3.)
        self.assertEqual(fop_1, fop_2)

        # Test in-place addition
        fop = FermionOperator("0^ 0", 2.)
        fop += FermionOperator("0^ 1", 3.)
        self.assertEqual(fop, fop_1)

        # Test addition with coefficient
        self.assertEqual(2. + fop_1, fop_2 + 2.)

        # Test addition with non-compatible type
        with self.assertRaises(RuntimeError):
            fop + "a"

    def test_mul(self):
        # Test in-place multiplication
        fop_1 = FermionOperator("0^ 0", 2.)
        fop_1 *= of.FermionOperator("0^ 1", 3.)
        # Test multiplication
        fop_2 = FermionOperator("0^ 0", 2.) * of.FermionOperator("0^ 1", 3.)
        self.assertEqual(fop_1, fop_2)

        # Test reverse multiplication
        fop_3 = of.FermionOperator("0^ 0", 2.) * FermionOperator("0^ 1", 3.)
        self.assertEqual(fop_2, fop_3)

        # Test multiplication by number
        fop_4 = 6. * FermionOperator("0^ 0 0^ 1", 1.)
        self.assertEqual(fop_3, fop_4)

    def test_sub(self):
        # Test in-place subtraction
        fop_1 = FermionOperator("0^ 0", 2.)
        fop_1 -= of.FermionOperator("0^ 1", 3.)
        # Test subtraction
        fop_2 = FermionOperator("0^ 0", 2.) - of.FermionOperator("0^ 1", 3.)
        self.assertEqual(fop_1, fop_2)

        # Test reverse subtraction
        fop_3 = of.FermionOperator("0^ 1", 3.) - FermionOperator("0^ 0", 2.)
        self.assertEqual(fop_2, -1*fop_3)

    def test_div(self):
        # Test in-place division
        fop_1 = FermionOperator("0^ 0", 2.)
        fop_1 /= 2.
        # Test division
        fop_2 = FermionOperator("0^ 0", 2.) / 2.
        self.assertEqual(fop_1, fop_2)

        # Test error for division by operator
        with self.assertRaises(TypeError):
            fop_1 / fop_2


class QubitOperatorTest(unittest.TestCase):

    def test_qubit_indices(self):
        """Test the qubit_indices property of the QubitOperator class."""

        q1 = QubitOperator()
        self.assertEqual(q1.qubit_indices, set())

        q2 = QubitOperator("Z0 Z1")
        self.assertEqual(q2.qubit_indices, {0, 1})

        q3 = QubitOperator("Z0 Z1") + QubitOperator("Z1 Z2")
        self.assertEqual(q3.qubit_indices, {0, 1, 2})

    def test_init_from_dict(self):
        d = {((0, 'X'), (1, 'Y')): 3.}
        qop = QubitOperator2.from_dict(d)
        self.assertEqual(qop, q1)

    def test_add_sub(self):
        # Addition with constant terms and qubit operators
        d = {(): c, ((0, 'X'), (1, 'Y')): 4., ((0, 'Z'), (1, 'Z')): 2.}
        self.assertEqual(c + q1 + q2, QubitOperator2.from_dict(d))

        # Substraction with constant terms and qubit operators, both on left and right
        d = {(): c, ((0, 'X'), (1, 'Y')): -4., ((0, 'Z'), (1, 'Z')): -2.}
        self.assertEqual(-q1 - (q2 - c), QubitOperator2.from_dict(d))

        # iadd
        q3 = deepcopy(q2)
        q3 += q1
        self.assertEqual(q3, q1+q2)

    def test_mult(self):
        # Multiplication with constant terms
        d = {(): 5*c, ((0, 'X'), (1, 'Y')): 5., ((0, 'Z'), (1, 'Z')): 10.}
        self.assertEqual(5*(q2+c), QubitOperator2.from_dict(d))

        # Multiplication with other qubit operator
        q1_of, q2_of = of.QubitOperator(), of.QubitOperator()
        q1_of.terms, q2_of.terms = q1.terms.copy(), q2.terms.copy()
        d_ref = {(): 3.0, ((0, 'Y'), (1, 'X')): (6+0j)}
        self.assertEqual(q1 * q2, QubitOperator2.from_dict(d_ref))

    def test_perf_mul(self):

        t1 = time()
        bq = bq1 * bq2
        t2 = time()
        print(f'Elapsed NEW :: {t2-t1} sec')

        t1 = time()
        bq_ref = bq1_ref * bq2_ref
        t2 = time()
        print(f'Elapsed REF :: {t2-t1} sec')

        self.assertEqual(bq, bq_ref)


class QubitHamiltonianTest(unittest.TestCase):

    def test_instantiate_QubitHamiltonian(self):
        """Test initialization of QubitHamiltonian class."""

        qubit_ham = 0.5 * QubitHamiltonian("X0 Y1 Z2", mapping="BK", up_then_down=False)
        reference_attributes = {"terms": {((0, 'X'), (1, 'Y'), (2, 'Z')): 0.5},
                                "mapping": "BK", "up_then_down": False}

        self.assertDictEqual(qubit_ham.__dict__, reference_attributes)

    def test_add_QubitHamiltonian(self):
        """Test error raising when 2 incompatible QubitHamiltonian are summed up
        together.
        """

        qubit_ham = QubitHamiltonian("X0 Y1 Z2", mapping="JW", up_then_down=True)

        with self.assertRaises(RuntimeError):
            qubit_ham + QubitHamiltonian("Z0 X1 Y2", mapping="BK", up_then_down=True)

        with self.assertRaises(RuntimeError):
            qubit_ham + QubitHamiltonian("Z0 X1 Y2", mapping="JW",  up_then_down=False)

    def test_to_qubit_operator(self):
        """Test exportation of QubitHamiltonian to QubitOperator."""

        qubit_ham = 1. * QubitHamiltonian("X0 Y1 Z2", mapping="JW",  up_then_down=True)

        self.assertEqual(qubit_ham.to_qubitoperator(), QubitOperator(term="X0 Y1 Z2", coefficient=1.))

    def test_get_operators(self):
        """Test get_operators methods, defined in QubitOperator class."""

        terms = [QubitHamiltonian("X0 Y1 Z2", mapping="JW", up_then_down=True),
                 QubitHamiltonian("Z0 X1 Y2", mapping="JW", up_then_down=True),
                 QubitHamiltonian("Y0 Z1 X2", mapping="JW", up_then_down=True)]

        H = terms[0] + terms[1] + terms[2]

        for ref_term, term in zip(terms, H.get_operators()):
            self.assertEqual(ref_term, term)


if __name__ == "__main__":
    unittest.main()
