import numpy as np
import scipy.optimize
import scipy.sparse.linalg

from .gellmann import dm_to_gellmann_distance
from ._misc import density_matrix_boundary
from ._matlab_utils import PPT_relative_entangled_entropy as relative_entangled_entropy


def get_boundary(dm_target, dimA, dimB, within_dm=True, return_tag=False):
    # within_dm=False TODO return_tag
    assert (dimA*dimB==dm_target.shape[0]) and (dm_target.shape[0]==dm_target.shape[1])
    rhoI = np.eye(dm_target.shape[0])/dm_target.shape[0]
    def hf0(alpha):
        alpha = max(alpha,0)
        tmp0 = rhoI*(1-alpha) + dm_target*alpha
        tmp1 = tmp0.reshape(dimA,dimB,dimA,dimB).transpose(0,3,2,1).reshape(dimA*dimB,dimA*dimB)
        ret = np.linalg.eigvalsh(tmp1)[0]
        return ret
    alpha = scipy.optimize.fsolve(hf0, 1/(dimA*dimB))[0]
    beta = dm_to_gellmann_distance(alpha*dm_target) #the missed identity doesn't matter here
    if within_dm:
        # PPT boundary might be outside of DM boundary
        alpha_dm,beta_dm,_ = density_matrix_boundary(dm_target, return_alpha=True)
        tag_out_dm = beta>=beta_dm
        if return_tag:
            ret = (alpha_dm,beta_dm,True) if tag_out_dm else (alpha,beta,False)
        else:
            ret = (alpha_dm,beta_dm) if tag_out_dm else (alpha,beta)
    else:
        ret = alpha,beta
    return ret


def is_positive_partial_transpose(rho, dim=None, eps=-1e-7, return_info=False):
    # Positive Partial Transpose (PPT)
    # https://en.wikipedia.org/wiki/Entanglement_witness
    # https://en.wikipedia.org/wiki/Peres%E2%80%93Horodecki_criterion
    N0 = rho.shape[0]
    if dim is None:
        tmp0 = int(np.sqrt(N0))
        assert tmp0*tmp0==N0
        dim = [tmp0,tmp0]
    assert (len(dim)>1) and (rho.shape[1]==N0) and (np.prod(dim)==N0) and all(x>1 for x in dim)
    def hf0(i):
        tmp0 = np.prod(dim[:i]) if i>0 else 1
        tmp1 = np.prod(dim[(i+1):]) if (i+1)<len(dim) else 1
        rhoT = rho.reshape(tmp0,dim[i],tmp1,tmp0,dim[i],tmp1).transpose(0,4,2,3,1,5).reshape(N0,N0)
        EVL = scipy.sparse.linalg.eigsh(rhoT, k=1, sigma=None, which='SA', return_eigenvectors=False)[0]
        return EVL
    tmp0 = (hf0(i) for i in range(len(dim)))
    if return_info:
        tmp1 = list(tmp0)
        ret = all(x>eps for x in tmp1), tmp1
    else:
        ret = all(x>eps for x in tmp0)
    return ret
