import unittest
import os
import numpy as np

from qsdk.toolboxes.post_processing import mcweeny_purify_2rdm

path_data = os.path.dirname(__file__) + '/data'


class McWeenyPurificationTest(unittest.TestCase):

    def test_mcweeny_exact(self):
        """ Check McWeeny's RDM purification technique using exact simulation results.
        Input, output and code as used in https://arxiv.org/pdf/2102.07045.pdf, R=1.1 A (pot 7) of the H10 curve """
        with open(f"{path_data}/rdm2_spin_pot7_exact.npy", 'rb') as f:
            rdm2_spin = np.load(f)
        with open(f"{path_data}/rdm1_nw_exact.npy", 'rb') as f:
            rdm1_mw_ref = np.load(f)
        with open(f"{path_data}/rdm2_nw_exact.npy", 'rb') as f:
            rdm2_mw_ref = np.load(f)

        rdm1_mw, rdm2_mw = mcweeny_purify_2rdm(rdm2_spin)
        np.testing.assert_array_almost_equal(rdm1_mw, rdm1_mw_ref)
        np.testing.assert_array_almost_equal(rdm2_mw, rdm2_mw_ref)

    def test_mcweeny_experimental(self):
        """ Check McWeeny's RDM purification technique using experimental simulation results.
        Input, output and code as used in https://arxiv.org/pdf/2102.07045.pdf, R=1.1 A (pot 7) of the H10 curve """
        with open(f"{path_data}/rdm2_spin_pot7_experimental.npy", 'rb') as f:
            rdm2_spin = np.load(f)
        with open(f"{path_data}/rdm1_nw.npy", 'rb') as f:
            rdm1_mw_ref = np.load(f)
        with open(f"{path_data}/rdm2_nw.npy", 'rb') as f:
            rdm2_mw_ref = np.load(f)

        rdm1_mw, rdm2_mw = mcweeny_purify_2rdm(rdm2_spin)
        np.testing.assert_array_almost_equal(rdm1_mw, rdm1_mw_ref)
        np.testing.assert_array_almost_equal(rdm2_mw, rdm2_mw_ref)


if __name__ == "__main__":
    unittest.main()