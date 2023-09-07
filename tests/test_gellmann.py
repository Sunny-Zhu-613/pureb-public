import numpy as np

import pyqet

def test_matrix_to_gellman_basis():
    for N0 in [3,4,5]:
        np0 = np.random.rand(N0,N0) + np.random.rand(N0,N0)*1j
        coeff = pyqet.gellmann.matrix_to_gellmann_basis(np0)
        tmp0 = pyqet.gellmann.all_gellmann_matrix(N0)
        ret0 = sum(x*y for x,y in zip(coeff,tmp0))
        assert np.abs(np0-ret0).max()<1e-7


def test_all_gellmann_matrix():
    # https://arxiv.org/abs/1705.01523
    for d in [3,4,5]:
        all_term = pyqet.gellmann.all_gellmann_matrix(d, with_I=False)
        for ind0,x in enumerate(all_term):
            for ind1,y in enumerate(all_term):
                assert abs(np.trace(x @ y)-2*(ind0==ind1)) < 1e-7
