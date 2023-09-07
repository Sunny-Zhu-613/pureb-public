import numpy as np

import pyqet

def test_isotropic_state():
    np_rng = np.random.default_rng()
    for d in range(2,10):
        alpha = np_rng.uniform(-(1/(d**2-1)), 1)
        rho = pyqet.isotropic_state(d, alpha)
        u0 = pyqet.random.rand_haar_unitary(d, np_rng)
        tmp0 = np.kron(u0, u0.conj())
        assert np.abs(tmp0 @ rho @ tmp0.T.conj() - rho).max() < 1e-7
    for d in range(2, 10):
        rho_entangled = pyqet.isotropic_state(d, np_rng.uniform(1/(d+1), 1))
        rho_separable = pyqet.isotropic_state(d, np_rng.uniform(-(1/(d**2-1)), 1/(d+1)))
        assert pyqet.ppt.is_positive_partial_transpose(rho_separable)
        assert not pyqet.ppt.is_positive_partial_transpose(rho_entangled)


def test_werner_state():
    np_rng = np.random.default_rng()
    for d in range(2, 10):
        alpha = np_rng.uniform(-1, 1)
        rho = pyqet.werner_state(d, alpha)
        u0 = pyqet.random.rand_haar_unitary(d, np_rng)
        tmp0 = np.kron(u0, u0)
        assert np.abs(tmp0 @ rho @ tmp0.T.conj() - rho).max() < 1e-7
    for d in range(2, 10):
        rho_entangled = pyqet.werner_state(d, np_rng.uniform(1/d, 1))
        rho_separable = pyqet.werner_state(d, np_rng.uniform(-1, 1/d))
        assert pyqet.ppt.is_positive_partial_transpose(rho_separable)
        assert not pyqet.ppt.is_positive_partial_transpose(rho_entangled)
    
