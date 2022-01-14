# Copyright 2021 Good Chemistry Company.
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

"""Module containing datastructures for interfacing with this package
functionalities.
"""

import copy
from dataclasses import dataclass, field
import numpy as np
from pyscf import gto, scf, ao2mo
import openfermion
import openfermionpyscf
from openfermionpyscf import run_pyscf
from openfermion.ops.representations.interaction_operator import get_active_space_integrals

from tangelo.toolboxes.molecular_computation.frozen_orbitals import get_frozen_core
from tangelo.toolboxes.qubit_mappings.mapping_transform import get_fermion_operator


def atom_string_to_list(atom_string):
    """Convert atom coordinate string (typically stored in text files) into a
    list/tuple representation suitable for openfermion.MolecularData.
    """

    geometry = []
    for line in atom_string.split("\n"):
        data = line.split()
        if len(data) == 4:
            atom = data[0]
            coordinates = (float(data[1]), float(data[2]), float(data[3]))
            geometry += [(atom, coordinates)]
    return geometry


def molecule_to_secondquantizedmolecule(mol, basis_set="sto-3g", frozen_orbitals=None):
    """Function to convert a Molecule into a SecondQuantizedMolecule.

    Args:
        mol (Molecule): Self-explanatory.
        basis_set (string): String representing the basis set.
        frozen_orbitals (int or list of int): Number of MOs or MOs indexes to
            freeze.

    Returns:
        SecondQuantizedMolecule: Mean-field data structure for a molecule.
    """

    converted_mol = SecondQuantizedMolecule(mol.xyz, mol.q, mol.spin,
                                            basis=basis_set,
                                            frozen_orbitals=frozen_orbitals)
    return converted_mol


@dataclass
class Molecule:
    """Custom datastructure to store information about a Molecule. This contains
    only physical information.

    Attributes:
        xyz (array-like or string): Nested array-like structure with elements
            and coordinates (ex:[ ["H", (0., 0., 0.)], ...]). Can also be a
            multi-line string.
        q (float): Total charge.
        spin (int): Absolute difference between alpha and beta electron number.
        n_atom (int): Self-explanatory.
        n_electrons (int): Self-explanatory.
        n_min_orbitals (int): Number of orbitals with a minimal basis set.

    Properties:
        elements (list): List of all elements in the molecule.
        coords (array of float): N x 3 coordinates matrix.
    """
    xyz: list or str
    q: int = 0
    spin: int = 0

    # Defined in __post_init__.
    n_atoms: int = field(init=False)
    n_electrons: int = field(init=False)
    n_min_orbitals: int = field(init=False)

    def __post_init__(self):
        self.xyz = atom_string_to_list(self.xyz) if isinstance(self.xyz, str) else self.xyz
        mol = self.to_pyscf(basis="sto-3g")
        self.n_atoms = mol.natm
        self.n_electrons = mol.nelectron
        self.n_min_orbitals = mol.nao_nr()

    @property
    def elements(self):
        return [self.xyz[i][0] for i in range(self.n_atoms)]

    @property
    def coords(self):
        return np.array([self.xyz[i][1] for i in range(self.n_atoms)])

    def to_pyscf(self, basis="sto-3g"):
        """Method to return a pyscf.gto.Mole object.

        Args:
            basis (string): Basis set.

        Returns:
            pyscf.gto.Mole: PySCF compatible object.
        """

        mol = gto.Mole()
        mol.atom = self.xyz
        mol.basis = basis
        mol.charge = self.q
        mol.spin = self.spin
        mol.build()

        return mol

    def to_openfermion(self, basis="sto-3g"):
        """Method to return a openfermion.MolecularData object.

        Args:
            basis (string): Basis set.

        Returns:
            openfermion.MolecularData: Openfermion compatible object.
        """

        return openfermion.MolecularData(self.xyz, basis, self.spin+1, self.q)


@dataclass
class SecondQuantizedMolecule(Molecule):
    """Custom datastructure to store information about a mean field derived
    from a molecule. This class inherits from Molecule and add a number of
    attributes defined by the second quantization.

    Attributes:
        basis (string): Basis set.
        mf_energy (float): Mean-field energy (RHF or ROHF energy depending
            on the spin).
        mo_energies (list of float): Molecular orbital energies.
        mo_occ (list of float): Molecular orbital occupancies (between 0.
            and 2.).
        mean_field (pyscf.scf): Mean-field object (used by CCSD and FCI).
        n_mos (int): Number of molecular orbitals with a given basis set.
        n_sos (int): Number of spin-orbitals with a given basis set.
        active_occupied (list of int): Occupied molecular orbital indexes
            that are considered active.
        frozen_occupied (list of int): Occupied molecular orbital indexes
            that are considered frozen.
        active_virtual (list of int): Virtual molecular orbital indexes
            that are considered active.
        frozen_virtual (list of int): Virtual molecular orbital indexes
            that are considered frozen.
        fermionic_hamiltonian (FermionOperator): Self-explanatory.

    Methods:
        freeze_mos: Change frozen orbitals attributes. It can be done inplace
            or not.

    Properties:
        n_active_electrons (int): Difference between number of total
            electrons and number of electrons in frozen orbitals.
        n_active_sos (int): Number of active spin-orbitals.
        n_active_mos (int): Number of active molecular orbitals.
        frozen_mos (list or None): Frozen MOs indexes.
        actives_mos (list): Active MOs indexes.
    """
    basis: str = "sto-3g"
    frozen_orbitals: list or int = field(default="frozen_core", repr=False)

    # Defined in __post_init__.
    mf_energy: float = field(init=False)
    mo_energies: list = field(init=False)
    mo_occ: list = field(init=False)

    mean_field: scf = field(init=False)

    n_mos: int = field(init=False)
    n_sos: int = field(init=False)

    active_occupied: list = field(init=False)
    frozen_occupied: list = field(init=False)
    active_virtual: list = field(init=False)
    frozen_virtual: list = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self._compute_mean_field()
        self.freeze_mos(self.frozen_orbitals)

    @property
    def n_active_electrons(self):
        return int(sum([self.mo_occ[i] for i in self.active_occupied]))

    @property
    def n_active_sos(self):
        return 2*len(self.active_mos)

    @property
    def n_active_mos(self):
        return len(self.active_mos)

    @property
    def fermionic_hamiltonian(self):
        return self._get_fermionic_hamiltonian()

    @property
    def frozen_mos(self):
        """This property returns MOs indexes for the frozen orbitals. It was
        written to take into account if one of the two possibilities (occ or
        virt) is None. In fact, list + None, None + list or None + None return
        an error. An empty list cannot be sent because PySCF mp2 returns
        "IndexError: list index out of range".

        Returns:
            list: MOs indexes frozen (occupied + virtual).
        """
        if self.frozen_occupied and self.frozen_virtual:
            return self.frozen_occupied + self.frozen_virtual
        elif self.frozen_occupied:
            return self.frozen_occupied
        elif self.frozen_virtual:
            return self.frozen_virtual
        else:
            return None

    @property
    def active_mos(self):
        """This property returns MOs indexes for the active orbitals.

        Returns:
            list: MOs indexes that are active (occupied + virtual).
        """
        return self.active_occupied + self.active_virtual

    def _compute_mean_field(self):
        """Computes the mean-field for the molecule. It supports open-shell
        mean-field calculation through openfermionpyscf. Depending on the
        molecule spin, it does a restricted or a restriction open-shell
        Hartree-Fock calculation.

        It is also used for defining attributes related to the mean-field
        (mf_energy, mo_energies, mo_occ, n_mos and n_sos).
        """
        of_molecule = self.to_openfermion(self.basis)
        of_molecule = run_pyscf(of_molecule, run_scf=True, run_mp2=False,
                                run_cisd=False, run_ccsd=False, run_fci=False)

        self.mf_energy = of_molecule.hf_energy
        self.mo_energies = of_molecule.orbital_energies
        self.mo_occ = of_molecule._pyscf_data["scf"].mo_occ

        self.n_mos = of_molecule._pyscf_data["mol"].nao_nr()
        self.n_sos = 2*self.n_mos

        self.mean_field = of_molecule._pyscf_data["scf"]

    def _get_fermionic_hamiltonian(self):
        """This method returns the fermionic hamiltonian. It written to take
        into account calls for this function is without argument, and attributes
        are parsed into it.

        Returns:
            FermionOperator: Self-explanatory.
        """

        occupied_indices = self.frozen_occupied
        active_indices = self.active_mos

        of_molecule = self.to_openfermion(self.basis)
        of_molecule = run_pyscf(of_molecule, run_scf=True, run_mp2=False,
                                run_cisd=False, run_ccsd=False, run_fci=False)

        molecular_hamiltonian = of_molecule.get_molecular_hamiltonian(occupied_indices, active_indices)

        return get_fermion_operator(molecular_hamiltonian)

    def _convert_frozen_orbitals(self, frozen_orbitals):
        """This method converts an int or a list of frozen_orbitals into four
        categories:
        - Active and occupied MOs;
        - Active and virtual MOs;
        - Frozen and occupied MOs;
        - Frozen and virtual MOs.
        Each of them are list with MOs indexes (first one is 0). Note that they
        are MOs labelled, not spin-orbitals (MOs * 2) indexes.

        Args:
            frozen_orbitals (int or list of int): Number of MOs or MOs indexes
                to freeze.

        Returns:
            list: Nested list of active occupied, frozen occupied, active
                virtual and frozen virtual orbital indexes.
        """

        if frozen_orbitals == "frozen_core":
            frozen_orbitals = get_frozen_core(self.to_pyscf(self.basis))
        elif frozen_orbitals is None:
            frozen_orbitals = 0

        # First case: frozen_orbitals is an int.
        # The first n MOs are frozen.
        if isinstance(frozen_orbitals, int):
            frozen_orbitals = list(range(frozen_orbitals))
        # Second case: frozen_orbitals is a list of int.
        # All MOs with indexes in this list are frozen (first MO is 0, second is 1, ...).
        # Everything else raise an exception.
        elif not (isinstance(frozen_orbitals, list) and all(isinstance(_, int) for _ in frozen_orbitals)):
            raise TypeError("frozen_orbitals argument must be an (or a list of) integer(s).")

        occupied = [i for i in range(self.n_mos) if self.mo_occ[i] > 0.]
        virtual = [i for i in range(self.n_mos) if self.mo_occ[i] == 0.]

        frozen_occupied = [i for i in frozen_orbitals if i in occupied]
        frozen_virtual = [i for i in frozen_orbitals if i in virtual]

        # Redefined active orbitals based on frozen ones.
        active_occupied = [i for i in occupied if i not in frozen_occupied]
        active_virtual = [i for i in virtual if i not in frozen_virtual]

        # Exception raised here if n_occupied <= frozen_orbitals (int), because it means that there is no active electron.
        # An exception is raised also if all occupied orbitals are in the frozen_orbitals (list).
        if (len(active_occupied) == 0) or (len(active_virtual) == 0):
            raise ValueError("All electrons or virtual orbitals are frozen in the system.")

        return active_occupied, frozen_occupied, active_virtual, frozen_virtual

    def freeze_mos(self, frozen_orbitals, inplace=True):
        """This method recomputes frozen orbitals with the provided input."""

        list_of_active_frozen = self._convert_frozen_orbitals(frozen_orbitals)

        if inplace:
            self.frozen_orbitals = frozen_orbitals

            self.active_occupied = list_of_active_frozen[0]
            self.frozen_occupied = list_of_active_frozen[1]
            self.active_virtual = list_of_active_frozen[2]
            self.frozen_virtual = list_of_active_frozen[3]

            return None
        else:
            # Shallow copy is enough to copy the same object and changing frozen
            # orbitals (deepcopy also returns errors).
            copy_self = copy.copy(self)

            copy_self.frozen_orbitals = frozen_orbitals

            copy_self.active_occupied = list_of_active_frozen[0]
            copy_self.frozen_occupied = list_of_active_frozen[1]
            copy_self.active_virtual = list_of_active_frozen[2]
            copy_self.frozen_virtual = list_of_active_frozen[3]

            return copy_self

    def energy_from_rdms(self, one_rdm, two_rdm):
        """Computes the molecular energy from one- and two-particle reduced
        density matrices (RDMs). Coefficients (integrals) are computed
        on-the-fly using a pyscf object and the mean-field. Frozen orbitals
        are supported with this method.

        Args:
            one_rdm (numpy.array): One-particle density matrix in MO basis.
            two_rdm (numpy.array): Two-particle density matrix in MO basis.

        Returns:
            float: Molecular energy.
        """

        # Pyscf molecule to get integrals.
        pyscf_mol = self.to_pyscf(self.basis)

        # Corresponding to nuclear repulsion energy and static coulomb energy.
        core_constant = float(pyscf_mol.energy_nuc())

        # get_hcore is equivalent to int1e_kin + int1e_nuc.
        one_electron_integrals = self.mean_field.mo_coeff.T @ self.mean_field.get_hcore() @ self.mean_field.mo_coeff

        # Getting 2-body integrals in atomic and converting to molecular basis.
        two_electron_integrals = ao2mo.kernel(pyscf_mol.intor("int2e"), self.mean_field.mo_coeff)
        two_electron_integrals = ao2mo.restore(1, two_electron_integrals, len(self.mean_field.mo_coeff))

        # PQRS convention in openfermion:
        # h[p,q]=\int \phi_p(x)* (T + V_{ext}) \phi_q(x) dx
        # h[p,q,r,s]=\int \phi_p(x)* \phi_q(y)* V_{elec-elec} \phi_r(y) \phi_s(x) dxdy
        # The convention is not the same with PySCF integrals. So, a change is
        # made and reverse back after performing the truncation for frozen
        # orbitals.
        two_electron_integrals = two_electron_integrals.transpose(0, 2, 3, 1)
        core_offset, one_electron_integrals, two_electron_integrals = get_active_space_integrals(one_electron_integrals,
                                                                                                 two_electron_integrals,
                                                                                                 self.frozen_occupied,
                                                                                                 self.active_mos)
        two_electron_integrals = two_electron_integrals.transpose(0, 3, 1, 2)

        # Adding frozen electron contribution to core constant.
        core_constant += core_offset

        # Computing the total energy from integrals and provided RDMs.
        e = core_constant + np.sum(one_electron_integrals * one_rdm) + 0.5*np.sum(two_electron_integrals * two_rdm)

        return e.real