""" Module for data structure for DMET fragments. """

from dataclasses import dataclass, field
import openfermion
import numpy as np
import pyscf
from pyscf import ao2mo

from qsdk.toolboxes.operators import FermionOperator
from qsdk.toolboxes.qubit_mappings.mapping_transform import get_fermion_operator


@dataclass
class SecondQuantizedDMETFragment:
    """ Mimicking SecondQuantizedMolecule for DMET fragments. It has the minimal
        number of attributes and methods to be parsed by electronic solvers.
    """

    molecule: pyscf.gto
    mean_field: pyscf.scf

    n_active_electrons: int
    n_active_sos: int
    q: int
    spin: int

    basis: str = field(init=False)
    n_active_mos: int = field(init=False)
    fermionic_hamiltonian: FermionOperator = field(init=False, repr=False)
    frozen_mos: None = field(init=False)

    def __post_init__(self):
        self.basis = self.molecule.basis
        self.n_active_mos = self.n_active_sos // 2
        self.fermionic_hamiltonian = self._get_fermionic_hamiltonian()
        self.frozen_mos = None

    def _get_fermionic_hamiltonian(self):
        """ This method returns the fermionic hamiltonian. It written to take into account
            calls for this function is without argument, and attributes are parsed into it.

            Returns:
                FermionOperator: Self-explanatory.
        """

        dummy_of_molecule = openfermion.MolecularData([["C", (0., 0. ,0.)]], "sto-3g", self.spin+1, self.q)

        # Overwrting nuclear repulsion term.
        dummy_of_molecule.nuclear_repulsion = self.mean_field.mol.energy_nuc()

        canonical_orbitals = self.mean_field.mo_coeff
        h_core = self.mean_field.get_hcore()
        n_orbitals = len(self.mean_field.mo_energy)

        # Overwriting 1-electron integrals.
        dummy_of_molecule._one_body_integrals = canonical_orbitals.T @ h_core @ canonical_orbitals

        twoint = self.mean_field._eri
        eri = ao2mo.restore(8, twoint, n_orbitals)
        eri = ao2mo.incore.full(eri, canonical_orbitals)
        eri = ao2mo.restore(1, eri, n_orbitals)

        # Overwriting 2-electrons integrals.
        dummy_of_molecule._two_body_integrals = np.asarray(eri.transpose(0, 2, 3, 1), order="C")

        fragment_hamiltonian = dummy_of_molecule.get_molecular_hamiltonian()

        return get_fermion_operator(fragment_hamiltonian)

    def to_pyscf(self, basis=None):
        """ Method to output the PySCF molecule.

            Returns
                pyscf.gto.Mole: PySCF molecule.
        """
        return self.molecule