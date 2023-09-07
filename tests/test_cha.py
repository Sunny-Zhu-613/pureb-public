import numpy as np

import pyqet

# about 1 minutes
def test_convex_hull_approximation_iterative():
    rho = pyqet.upb.load_upb('tiles', return_bes=True)[1]
    alpha_history,ketA,ketB,lambda_ = pyqet.cha.convex_hull_approximation_iterative(rho, dimA=3, maxiter=100)
    # alpha=0.86327, eps=0.0490, 160 seconds
    # alpha_history (list,float)
    # ketA(np,complex128,(N,3))
    # ketB(np,complex128,(N,3))
    # lambda_(np,float,N)
    assert alpha_history[-1]>0.863 #if not achieved, there must be bugs somewhere

    assert np.abs(np.linalg.norm(ketA, axis=1)-1).max() < 1e-7
    assert np.abs(np.linalg.norm(ketB, axis=1)-1).max() < 1e-7
    tmp0 = (ketA[:,:,np.newaxis] * ketB[:,np.newaxis]).reshape(ketA.shape[0], -1, 1)
    reconstructed_rho = np.sum((tmp0 * tmp0.transpose(0,2,1).conj())*lambda_[:,np.newaxis,np.newaxis], axis=0)
    if alpha_history[-1]<1: #most likely to be an entangled state
        # alpha*rho + (1-alpha)*eye = sum_i (lambda_i * ketA_i * ketB_i)
        tmp0 = np.eye(rho.shape[0])/rho.shape[0]
        z0 = alpha_history[-1]*rho + (1-alpha_history[-1])*tmp0 - reconstructed_rho
        assert np.abs(z0).max()<1e-7
