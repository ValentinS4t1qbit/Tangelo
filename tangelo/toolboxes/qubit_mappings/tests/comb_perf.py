import os
from time import time

from tangelo import SecondQuantizedMolecule
from tangelo.helpers import assert_freq_dict_almost_equal
from tangelo.toolboxes.qubit_mappings import combinatorial, combinatorial_dict, combinatorial_jl, combinatorial5, combinatorial4

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

xyz = xyz_h2o
basis = "sto-3g"

mol = SecondQuantizedMolecule(xyz, q=0, spin=0, basis=basis, frozen_orbitals=[])

t1 = time()
qop1 = combinatorial_jl(mol.fermionic_hamiltonian, mol.n_active_mos, mol.n_active_electrons)
t2 = time()
print(f'Time elapsed {t2-t1:.4f} s. \t #terms = {len(qop1.terms)}\n')

t1 = time()
qop2 = combinatorial_dict(mol.fermionic_hamiltonian, mol.n_active_mos, mol.n_active_electrons)
t2 = time()
print(f'Time elapsed {t2-t1:.4f} s. \t #terms = {len(qop2.terms)}\n')

# assert_freq_dict_almost_equal(qop1.terms, qop2.terms, atol=1e-4)
