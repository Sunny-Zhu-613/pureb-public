import numpy as np
import torch

def gellmann_matrix(i,j,d):
    """
    Generate Gellmann matrix for d-dimensional space
    input: i,j,d
    output: Gellmann matrix G_ij 
    """
    assert (d>0) and (0<=i<d) and (0<=j<d)
    if i > j:
        ind0 = [i, j]
        ind1 = [j, i]
        data = [1j, -1j]
    elif j > i:
        ind0 = [i, j]
        ind1 = [j, i]
        data = [1, 1]
    elif i==0 and j==0:
        ind0 = np.arange(d)
        ind1 = np.arange(d)
        data = np.ones(d)
    else:
        ind0 = np.arange(i+1)
        ind1 = np.arange(i+1)
        data = np.sqrt(2/(i*(i+1)))*np.array([1]*i + [-i])
    ret = np.zeros((d, d), dtype=np.complex128)
    ret[ind0, ind1] = data
    return ret

_all_gellmann_matrix_cache = dict()
def all_gellmann_matrix(d, with_I=True):
    """
    Generate all Gellmann matrices for d-dimensional space
    input: d
    output: a list of Gellmann matrices (with I if with_I=True)
    """
    d = int(d)
    assert d>=2
    if d in _all_gellmann_matrix_cache:
        ret = _all_gellmann_matrix_cache[d]
    else:
            sym_mat = [gellmann_matrix(i,j,d) for i in range(d) for j in range(i+1,d)]
            antisym_mat = [gellmann_matrix(j,i,d) for i in range(d) for j in range(i+1,d)]
            diag_mat = [gellmann_matrix(i,i,d) for i in range(1,d)]
            tmp0 = [gellmann_matrix(0,0,d)]
            ret = np.stack(sym_mat+antisym_mat+diag_mat+tmp0, axis=0)
            _all_gellmann_matrix_cache[d] = ret
    if not with_I:
            ret = ret[:-1]
    return ret

def matrix_to_gellmann_basis(A):
    """
    Convert a matrix to Gellmann basis
    input: a list of matrices A or a matrix A
    output: (a list of) vec = [aS, aA, aD, aI] where aS, aA, aD, aI are the coefficients for symmetric, antisymmetric, diagonal matrices, and identity matrix 
    """
    shape0 = A.shape
    N0 = shape0[-1]
    assert len(shape0)>=2 and shape0[-2]==N0
    A = A.reshape(-1,N0,N0)
    if isinstance(A, torch.Tensor):
        indU0,indU1 = torch.triu_indices(N0, N0, offset=1)
        aS = (A + A.transpose(1,2))[:,indU0,indU1]/2
        aA = (A - A.transpose(1,2))[:,indU0,indU1] * (0.5j)
        tmp0 = torch.diagonal(A, dim1=1, dim2=2)
        tmp1 = torch.sqrt(2*torch.arange(1,N0,dtype=torch.float64)*torch.arange(2,N0+1))
        aD = (torch.cumsum(tmp0,dim=1)[:,:-1] - torch.arange(1,N0)*tmp0[:,1:])/tmp1
        aI = torch.einsum(A, [0,1,1], [0]) / N0
        ret = torch.concat([aS,aA,aD,aI.view(-1,1)], dim=1)
    else:
        indU0,indU1 = np.triu_indices(N0,1)
        aS = (A + A.transpose(0,2,1))[:,indU0,indU1]/2
        aA = (A - A.transpose(0,2,1))[:,indU0,indU1] * (0.5j)
        tmp0 = np.diagonal(A, axis1=1, axis2=2)
        tmp1 = np.sqrt(2*np.arange(1,N0)*np.arange(2,N0+1))
        aD = (np.cumsum(tmp0,axis=1)[:,:-1] - np.arange(1,N0)*tmp0[:,1:])/tmp1
        aI = np.trace(A, axis1=1, axis2=2) / N0
        ret = np.concatenate([aS,aA,aD,aI[:,np.newaxis]], axis=1)
    if len(shape0)==2:
        ret = ret[0]
    else:
        ret = ret.reshape(*shape0[:-2], -1)
    return ret

def gellmann_basis_to_matrix(vec):
    """
    Convert a list of vectors or a single vector in Gellmann basis to a list of matrices or a single matrix
    input: (a list of) vec = [aS, aA, aD, aI] where aS, aA, aD, aI are the coefficients for symmetric, antisymmetric, diagonal matrices, and identity matrix (with norm_I='sqrt(2/d)' or '1/d')
    output: a list of matrices A or a matrix A
    """
    shape = vec.shape
    vec = vec.reshape(-1, shape[-1])
    N0 = vec.shape[0]
    N1 = int(np.sqrt(vec.shape[1]))
    vec0 = vec[:,:(N1*(N1-1)//2)]
    vec1 = vec[:,(N1*(N1-1)//2):(N1*(N1-1))]
    vec2 = vec[:,(N1*(N1-1)):-1]
    vec3 = vec[:,-1:]
    assert vec.shape[1]==N1*N1
    is_torch = isinstance(vec, torch.Tensor)
    if is_torch:
        indU0,indU1 = torch.triu_indices(N1,N1,1)
        indU01 = torch.arange(N1*N1).reshape(N1,N1)[indU0,indU1]
        ind0 = torch.arange(N1)
        indU012 = (((N1*N1)*torch.arange(N0).view(-1,1)) + indU01).view(-1)
        zero0 = torch.zeros(N0*N1*N1, dtype=torch.complex128)
        ret0 = torch.scatter(zero0, 0, indU012, (vec0 - 1j*vec1).view(-1)).reshape(N0, N1, N1)
        ret1 = torch.scatter(zero0, 0, indU012, (vec0 + 1j*vec1).view(-1)).reshape(N0, N1, N1).transpose(1,2)
        tmp0 = torch.sqrt(torch.tensor(2,dtype=torch.float64)/(ind0[1:]*(ind0[1:]+1)))
        tmp1 = torch.concat([tmp0*vec2, vec3], axis=1)
        ret2 = torch.diag_embed(tmp1 @ ((ind0.view(-1,1)>=ind0) + torch.diag(-ind0[1:], diagonal=1)).to(tmp1.dtype))
        ret = ret0 + ret1 + ret2
    else:
        ret = np.zeros((N0,N1,N1), dtype=np.complex128)
        indU0,indU1 = np.triu_indices(N1,1)
        # indL0,indL1 = np.tril_indices(N1,-1)
        ind0 = np.arange(N1, dtype=np.int64)
        ret[:,indU0,indU1] = vec0 - 1j*vec1
        tmp0 = np.zeros_like(ret)
        tmp0[:,indU0,indU1] = vec0 + 1j*vec1
        ret += tmp0.transpose(0,2,1)
        tmp1 = np.concatenate([np.sqrt(2/(ind0[1:]*(ind0[1:]+1)))*vec2, vec3], axis=1)
        ret[:,ind0,ind0] = tmp1 @ ((ind0[:,np.newaxis]>=ind0) + np.diag(-ind0[1:], k=1))
    ret = ret[0] if (len(shape)==1) else ret.reshape(*shape[:-1], N1, N1)
    return ret

def dm_to_gellmann_basis(dm, with_rho0=False):
    """
    Convert a density matrix to Gellmann basis
    Input: a density matrix dm
    Output: (a list of) vec = [aS, aA, aD, aI] where aS, aA, aD, aI are the coefficients for symmetric, antisymmetric, diagonal matrices, and identity matrix (with_rh0=True)
    """
    ret = matrix_to_gellmann_basis(dm)
    if not with_rho0:
        shape = ret.shape
        tmp0 = ret.reshape(-1,shape[-1])[:,:-1].real
        ret = tmp0[0] if (len(shape)==1) else tmp0.reshape(*shape[:-1],-1)
    return ret

def gellmann_basis_to_dm(vec):
    """
    Convert (a list of) vector in Gellmann basis to density matrix(matrices)
    Input: a list of vectors vec in Gellmann basis
    Output: a density matrix (a list of density matrices)
    """
    shape = vec.shape
    d = int(round(np.sqrt(shape[-1]+1).item()))
    assert shape[-1]==d**2-1
    vec = vec.reshape(-1,shape[-1])
    tmp0 = np.concatenate([vec, np.ones((vec.shape[0],1), dtype=np.float64)/d], axis=1)
    ret = gellmann_basis_to_matrix(tmp0)
    ret = ret[0] if (len(shape)==1) else ret.reshape(*shape[:-1], *ret.shape[-2:])
    return ret

def dm_to_gellmann_distance(dm):
    """
    Calculate the Euclidean distance of a density matrix from maximally mixed state in the Gellmann basis
    Input: a density matrix dm
    Output: the Euclidean distance
    """
    tmp0 = dm_to_gellmann_basis(dm)
    shape = tmp0.shape
    tmp1 = tmp0.reshape(-1,shape[-1])
    ret = np.linalg.norm(tmp1, axis=1)
    ret = ret.item() if (len(shape)==1) else ret.reshape(*shape[:-1])
    return ret






    
    
    
    
  


  
     


        


