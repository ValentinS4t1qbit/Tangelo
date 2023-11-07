

import os
import unittest
from time import time

from openfermion import load_operator

from tangelo import SecondQuantizedMolecule
from tangelo.helpers import assert_freq_dict_almost_equal
from tangelo.molecule_library import mol_H2_sto3g
from tangelo.toolboxes.operators import QubitOperator as QOp
from tangelo.toolboxes.qubit_mappings import combinatorial, combinatorial_dict, combinatorial5, combinatorial4
from tangelo.toolboxes.qubit_mappings.combinatorial import basis, conf_to_integer, one_body_op_on_state


path_data = os.path.dirname(os.path.abspath(__file__)) + '/data'


xyz_h2 = """
H  0.0000 0.0000 0.0000
H  0.0000 0.0000 0.7414
"""

xyz_lih = """
Li 0.0000 0.0000 0.0000
H  0.0000 0.0000 1.5949
"""

xyz_h2o = """
O  0.0000  0.0000  0.1173
H  0.0000  0.7572 -0.4692
H  0.0000 -0.7572 -0.4692
"""

xyz = xyz_lih
basis = "6-31g"

mol = SecondQuantizedMolecule(xyz, q=0, spin=0, basis=basis, frozen_orbitals=[])

t1 = time()
qop1 = combinatorial_dict(mol.fermionic_hamiltonian, mol.n_active_mos, mol.n_active_electrons)
t2=time()
print(f'Time elapsed {t2-t1:.4f} s. \t #terms = {len(qop1.terms)}\n')

t1 = time()
qop2 = combinatorial5(mol.fermionic_hamiltonian, mol.n_active_mos, mol.n_active_electrons)
t2=time()
print(f'Time elapsed {t2-t1:.4f} s. \t #terms = {len(qop2.terms)}\n')

assert_freq_dict_almost_equal(qop1.terms, qop2.terms, atol=1e-4)
