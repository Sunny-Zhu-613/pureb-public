import itertools
import numpy as np
import scipy.special
import torch

def dicke_state(num_qubit, num_excited, return_dm=False):
    """
    Generate dicke state (qubit case)
    input: num_qubit, num_excited
    output: pure dicke state (or density matrix)
    """
    # number of states
    num_term = int(scipy.special.comb(num_qubit, num_excited))
    d_base_excited_pos = list(itertools.combinations(range(num_qubit), num_excited))
    hf = lambda x: sum([2**i for i in x])
    index_excited_pos = [hf(pos) for pos in d_base_excited_pos]
    dicke_state = np.zeros(2**num_qubit, dtype=np.float64)

    for pos in index_excited_pos:
        dicke_state[pos] = 1
    dicke_state /= np.sqrt(num_term)

    if return_dm:
        return dicke_state[:, np.newaxis] @ dicke_state[np.newaxis, :]
    else:
        return dicke_state
    
def dicke_state_partial_trace(num_qubit):
    """
    partial trace of dicke state e1, e2 where e1 is the excited number of the ket and e2 is the excited number of the bra, i.e. partial_trace(ket e1 * bra e2)
    input: num_qubit
    output: a00, a01, a10, a11 reduced matrix elements on the 0,1 basis 
    a00: (e1,e2) from (0,0) to (n,n)
    a01: (e1,e2) from (0,1) to (n-1,n)
    a10: (e1,e2) from (1,0) to (n,n-1)
    a11: (e1,e2) from (1,1) to (n-1,n-1)
    """
    tmp0 = np.arange(num_qubit+1)
    a00 = (num_qubit - tmp0) / num_qubit # from (0,0) to (n,n)
    a11 = tmp0 / num_qubit # from (0,0) to (n,n)
    a01 = np.sqrt(tmp0[:0:-1]*tmp0[1:]) / num_qubit # from (0,1) to (n-1,n)
    a10 = a01 # from (1,0) to (n,n-1)
    return a00, a01, a10, a11

def partial_trace_AC_to_AB(state, dicke_a_vec=None):
    """
    partial trace of a state AC to AB, C=B^n, B is the qubit system
    input: state AC, stateAC.shape = (dimA, dimC), dimC is the dimension of the symmetric space
    output: state AB, stateAB.shape = (dimA*2, dimA*2)
    """
    is_torch = isinstance(state, torch.Tensor)
    assert state.ndim==2
    dimA,dimC = state.shape
    if dicke_a_vec is None:
        a00,a01,a10,a11 = dicke_state_partial_trace(dimC-1)
        if is_torch:
            a00,a01,a10,a11 = [torch.tensor(x,dtype=torch.complex128) for x in [a00,a01,a10,a11]]
    else:
        a00,a01,a10,a11 = dicke_a_vec
    state_conj = state.conj()
    rho00 = (state * a00) @ state_conj.T
    rho11 = (state * a11) @ state_conj.T
    rho01 = (state[:,:-1] * a01) @ state_conj[:,1:].T
    rho10 = (state[:,1:] * a10) @ state_conj[:,:-1].T
    if is_torch:
        ret = torch.stack([rho00,rho01,rho10,rho11], dim=2).reshape(dimA,dimA,2,2).transpose(1,2).reshape(dimA*2,dimA*2)
    else:
        ret = np.stack([rho00,rho01,rho10,rho11], axis=2).reshape(dimA,dimA,2,2).transpose(0,2,1,3).reshape(dimA*2,dimA*2)
    return ret
    
def qudit_dicke_state(*seq):
    """
    Generate generalized dicke state for qudit
    input: seq, a list of integers (w0,w1,...,wd-1), where wi is the number of qudits in the i-th energy level
    output: qudit dicke state, shape = (d**num_qudit,)
    """
    seq = np.asarray([int(x) for x in seq], dtype=np.int64)
    d = len(seq)
    num_qudit = seq.sum()
    tmp0 = [int(x) for x,y in enumerate(seq) for _ in range(y)]
    tmp1 = np.unique(np.array(list(itertools.permutations(tmp0))), axis=0)
    tmp2 = d**(np.arange(num_qudit)[::-1])
    ret = np.zeros(d**num_qudit, dtype=np.float64)
    ret[tmp1@tmp2] = 1/np.sqrt(len(tmp1))
    return ret

def qudit_dicke_state_partial_trace(d, num_qudit):
    """
    partial trace of qudit dicke state w1, w2 where w1 is the weight tuple of the ket and e2 is the weight tuple of the bra, i.e. partial_trace(ket w1 * bra w2)
    input: d, num_qudit
    output: 
    ret: [(list of index of w1, list of index of w2, list of matrix elements), ...], this list has length d*d and each element is a tuple of three lists
    weight_list: list of weight tuples, i.e. [(w0,w1,...,wd-1), ...]
    pos_dict: dictionary of the position of matrix elements corresponding to the index(w1), index(w2) and int(i,j)
    """
    assert (d>1) and (num_qudit>=1)
    hf0 = lambda d, n: [(n,)] if (d<=1) else [(x,)+y for x in range(n+1) for y in hf0(d-1,n-x)]
    weight_list = hf0(d, num_qudit)
    len_weight_list = len(weight_list) # dimension of the qudit dicke states
    weight_list_to_index = {y:x for x,y in enumerate(weight_list)}
    weight_list_np = np.array(weight_list, dtype=np.int64)
    pos_dict = dict()
    ret = []
    for i in range(d):
        for j in range(d):
            if i==j:
                tmp0 = np.arange(len_weight_list)
                tmp1 = np.arange(len_weight_list)
                tmp2 = weight_list_np[:,i]/ num_qudit
                ret.append((tmp0,tmp1,tmp2))
                for x,y in enumerate(weight_list):
                    pos_dict[y,y,d*i+j] = x
            else:
                tmp0 = []
                for x,y in enumerate(weight_list):
                    tmp1 = list(y)
                    tmp1[i] -= 1
                    tmp1[j] += 1
                    tmp1 = tuple(tmp1)
                    if tmp1 in weight_list_to_index:
                        tmp0.append((x, weight_list_to_index[tmp1]))
                for x,(y0,y1) in enumerate(tmp0):
                    pos_dict[weight_list[y0],weight_list[y1],d*i+j] = x
                tmp1 = np.array([x[0] for x in tmp0], dtype=np.int64)
                tmp2 = np.array([x[1] for x in tmp0], dtype=np.int64)
                tmp3 = np.sqrt(weight_list_np[tmp1,i]*weight_list_np[tmp2,j])/num_qudit
                ret.append((tmp1,tmp2,tmp3))
    return ret, weight_list, pos_dict
            
def qudit_partial_trace_AC_to_AB(state, dicke_Bij):
    """
    partial trace of a state AC to AB, C=B^n, B is the qudit system
    input: state AC, stateAC.shape = (dimA, dimC), dimC is the dimension of the symmetric space
    output: state AB, stateAB.shape = (dimA*d, dimA*d)
    """
    is_torch = isinstance(state, torch.Tensor)
    assert state.ndim == 2
    dimA, _ = state.shape
    dimB = int(np.sqrt(len(dicke_Bij)))
    ret = []
    state_conj = state.conj()
    for ind0, ind1, value in dicke_Bij:
        ret.append((state[:,ind0]*value)@state_conj[:,ind1].T)
    if is_torch:
        ret = torch.stack(ret, dim=2).reshape(dimA,dimA,dimB,dimB).transpose(1,2).reshape(dimA*dimB,dimA*dimB)
    else:
        ret = np.stack(ret, axis=2).reshape(dimA,dimA,dimB,dimB).transpose(0,2,1,3).reshape(dimA*dimB,dimA*dimB)

    return ret
