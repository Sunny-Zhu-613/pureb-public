import collections
import numpy as np
import scipy.linalg
import torch

from .gellmann import gellmann_basis_to_dm, dm_to_gellmann_basis
from ._torch_op import TorchMatrixSqrtm, TorchMatrixLogm

op_torch_logm = TorchMatrixLogm(num_sqrtm=6, pade_order=8)

def hf_intertropalte_dm(dm0, alpha=None, beta=None):
    """
    Interpolate a density matrix through the direction on the Gell-Mann basis.
    input: dm0, the initial density matrix
              alpha, the interpolation parameter
                beta, the interpolation parameter
    output: the interpolated density matrix
    """
    assert (alpha is not None) or (beta is not None)
    if alpha is not None:
        N0 = dm0.shape[0]
        ret = alpha*dm0 + ((1-alpha)/N0)*np.eye(N0)
    else:
        tmp0 = dm_to_gellmann_basis(dm0)
        ret = gellmann_basis_to_dm(tmp0*(beta/np.linalg.norm(tmp0)))
    return ret


def density_matrix_boundary(dm_or_vec, return_alpha=False, return_both=False):
    """
    Get the upper and lower bounds of the density matrix boundary given the desired direction in quantum state space, and coreesponding states
    input: dm_or_vec, the density matrix or the vector in the Gell-Mann basis
    output: (alpha_u,beta_u,dm_u) or (beta_u,dm_u) or (alpha_u,beta_u,dm_u,alpha_l,beta_l,dm_l) or (beta_u,dm_u,beta_l,dm_l)
    """
    if dm_or_vec.ndim==1:
        vec0 = dm_or_vec
        # dm0 = gellmann_basis_to_dm(dm_or_vec)
    else:
        # dm0 = dm_or_vec
        vec0 = dm_to_gellmann_basis(dm_or_vec)
    vec0_norm = np.linalg.norm(vec0)
    dm_norm = gellmann_basis_to_dm(vec0 / vec0_norm)
    N0 = dm_norm.shape[0]
    tmp0 = np.linalg.eigvalsh(dm_norm)
    beta_u = 1/(1-N0*tmp0[0])
    beta_l = 1/(1-N0*tmp0[-1])
    alpha_u = beta_u/vec0_norm
    alpha_l = beta_l/vec0_norm
    rho0 = np.eye(N0)/N0
    tmp0 = dm_norm - rho0
    dm_u = rho0 + beta_u*tmp0
    dm_l = rho0 + beta_l*tmp0
    ret = (alpha_u,beta_u,dm_u) if return_alpha else (beta_u,dm_u)
    if return_both:
        tmp0 = (alpha_l,beta_l,dm_l) if return_alpha else (beta_l,dm_l)
        ret = ret + tmp0
    return ret


def density_matrix_plane(dm0, dm1):
    """
    Get the plane of the two density matrices on polar coordinate and the function computes the density matrix for (r,theta).
    input: dm0, the first density matrix
            dm1, the second density matrix
    output: theta0, the angle of the first density matrix
            norm0, the norm of the first density matrix
            theta1, the angle of the second density matrix
            norm1, the norm of the second density matrix
            hf0, the function computes the density matrix for (r,theta)
    """
    vec0 = dm_to_gellmann_basis(dm0)
    vec1 = dm_to_gellmann_basis(dm1)
    norm0 = np.linalg.norm(vec0)
    basis0 = vec0 / norm0
    norm1 = np.linalg.norm(vec1)
    tmp0 = vec1 - np.dot(basis0, vec1) * basis0
    basis0 = vec0 / norm0
    basis1 = tmp0 / np.linalg.norm(tmp0)
    theta0 = 0
    theta1 = np.arccos(np.dot(vec0, vec1)/(norm0*norm1))
    def hf0(theta, norm=1):
        tmp0 = norm*(basis0*np.cos(theta) + basis1*np.sin(theta))
        ret = gellmann_basis_to_dm(tmp0)
        return ret
    return theta0,norm0,theta1,norm1,hf0


def beta_to_alpha(dm, beta):
    vec0 = dm_to_gellmann_basis(dm)
    tmp0 = np.linalg.norm(vec0)
    ret = beta / tmp0
    return ret


def alpha_to_beta(dm, alpha):
    ret = np.linalg.norm(dm_to_gellmann_basis(hf_intertropalte_dm(dm, alpha)))
    return ret


def quantum_relative_entropy(rho0, rho1, kind='error', zero_tol=1e-5):
    is_torch = isinstance(rho0, torch.Tensor)
    if is_torch:
        tmp0 = op_torch_logm(rho0)
        tmp1 = op_torch_logm(rho1)
        ret = torch.trace(rho0 @ tmp0) - torch.trace(rho0 @ tmp1)
    else:
        tmp0 = scipy.linalg.logm(rho0)
        tmp1 = scipy.linalg.logm(rho1)
        ret = np.trace(rho0 @ tmp0) - np.trace(rho0 @ tmp1)
    if abs(ret.imag.item())>zero_tol:
        assert kind in {'error', 'infinity', 'ignore'}
        if kind=='error':
            raise ValueError('quantum-relative-entropy be infinty')
        elif kind=='infinity':
            ret = torch.inf if is_torch else np.inf
    ret = ret.real
    return ret


def werner_state(d, alpha):
    # https://en.wikipedia.org/wiki/Werner_state
    # https://www.quantiki.org/wiki/werner-state
    # alpha = ((1-2*p)*d+1) / (1-2*p+d)
    # separable for alpha<1/d
    assert d>1
    assert (-1<=alpha) and (alpha<=1)
    pmat = np.eye(d**2).reshape(d,d,d,d).transpose(0,1,3,2).reshape(d**2,d**2)
    ret = (np.eye(d**2)-alpha*pmat) / (d**2-d*alpha)
    return ret


def werner_state_ree(d, alpha):
    # REE(relative entropy of entangement)
    if alpha<=1/d:
        ret = 0
    else:
        rho0 = werner_state(d, alpha)
        rho1 = werner_state(d, 1/d)
        ret = quantum_relative_entropy(rho0, rho1, kind='infinity')
    return ret


def isotropic_state(d, alpha):
    # https://www.quantiki.org/wiki/isotropic-state
    assert d>1
    assert ((-1/(d**2-1))<=alpha) and (alpha<=1) #beyond this range, the density matrix is not SDP
    tmp0 = np.eye(d).reshape(-1)
    ret = ((1-alpha)/d**2) * np.eye(d**2) + (alpha/d) * (tmp0[:,np.newaxis]*tmp0)
    # separable for alpha<1/(d+1)
    return ret


def isotropic_state_ree(d, alpha):
    if alpha<=1/(d+1):
        ret = 0
    else:
        rho0 = isotropic_state(d, alpha)
        rho1 = isotropic_state(d, 1/(d+1))
        ret = quantum_relative_entropy(rho0, rho1, kind='infinity')
    return ret


def copy_numpy_to_cp(np0, cp0):
    assert (np0.size==cp0.size) and (np0.itemsize==cp0.itemsize)
    cp0.data.copy_from_host(np0.__array_interface__['data'][0], np0.size*np0.itemsize)


def product_state_to_dm(ketA, ketB, probability):
    tmp0 = (ketA[:,:,np.newaxis]*ketB[:,np.newaxis]).reshape(ketA.shape[0],-1)
    ret = np.sum((tmp0[:,:,np.newaxis]*tmp0[:,np.newaxis].conj())*probability[:,np.newaxis,np.newaxis], axis=0)
    return ret


def density_matrix_fidelity(rho, sigma):
    is_torch = isinstance(rho, torch.Tensor)
    if is_torch:
        sqrt_rho = TorchMatrixSqrtm.apply(rho)
        ret = torch.trace(TorchMatrixSqrtm.apply(sqrt_rho @ sigma @ sqrt_rho))
    else:
        sqrt_rho = scipy.linalg.sqrtm(rho)
        ret = np.trace(scipy.linalg.sqrtm(sqrt_rho @ sigma @ sqrt_rho))
    return ret


def max_entangled_state(dim):
    ret = np.diag(np.ones(dim)*1/np.sqrt(dim))
    ret = ret.reshape(-1)
    return ret


def density_matrix_purity(rho):
    # ret = np.trace(rho @ rho).real
    ret = np.sum(rho*rho.T).real
    return ret


def state_to_dm(ket):
    ret = ket[:,np.newaxis] * ket.conj()
    return ret


def partial_trace(rho, dim, keep_index):
    if not isinstance(keep_index, collections.Iterable):
        keep_index = [keep_index]
    N0 = len(dim)
    keep_index = sorted(set(keep_index))
    rho = rho.reshape(*dim, *dim)
    assert all(0<=x<N0 for x in keep_index)
    tmp0 = list(range(N0))
    tmp1 = list(range(N0,2*N0))
    tmp2 = set(range(N0))-set(keep_index)
    for x in tmp2:
        tmp1[x] = x
    tmp3 = list(keep_index) + [x+N0 for x in keep_index]
    N1 = np.prod([dim[x] for x in keep_index])
    ret = np.einsum(rho, tmp0+tmp1, tmp3, optimize=True).reshape(N1, N1)
    return ret
