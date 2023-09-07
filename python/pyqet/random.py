import numpy as np

from .gellmann import gellmann_basis_to_dm, dm_to_gellmann_basis

def get_numpy_rng(np_rng_or_seed_or_none):
    if np_rng_or_seed_or_none is None:
        ret = np.random.default_rng()
    elif isinstance(np_rng_or_seed_or_none, np.random.Generator):
        ret = np_rng_or_seed_or_none
    else:
        seed = int(np_rng_or_seed_or_none)
        ret = np.random.default_rng(seed)
    return ret


# not a public api
def _random_complex(*size, seed=None):
    np_rng = get_numpy_rng(seed)
    ret = np_rng.normal(size=size + (2,)).astype(np.float64, copy=False).view(np.complex128).reshape(size)
    return ret


def rand_haar_state(N0, seed=None):
    # http://www.qetlab.com/RandomStateVector
    ret = _random_complex(N0, seed=seed)
    ret /= np.linalg.norm(ret)
    return ret


def rand_haar_unitary(N0, seed=None):
    # http://www.qetlab.com/RandomUnitary
    # https://pennylane.ai/qml/demos/tutorial_haar_measure.html
    ginibre_ensemble = _random_complex(N0, N0, seed=seed)
    Q,R = np.linalg.qr(ginibre_ensemble)
    tmp0 = np.sign(np.diag(R).real)
    tmp0[tmp0==0] = 1
    ret = Q * tmp0
    return ret


def rand_bipartitle_state(N0, N1=None, k=None, seed=None, return_dm=False):
    # http://www.qetlab.com/RandomStateVector
    np_rng = get_numpy_rng(seed)
    if N1 is None:
        N1 = N0
    if k is None:
        ret = rand_haar_state(N0, np_rng)
    else:
        assert (0<k) and (k<=N0) and (k<=N1)
        tmp0 = np.linalg.qr(_random_complex(N0, N0, seed=np_rng), mode='complete')[0][:,:k]
        tmp1 = np.linalg.qr(_random_complex(N1, N1, seed=np_rng), mode='complete')[0][:,:k]
        tmp2 = _random_complex(k, seed=np_rng)
        tmp2 /= np.linalg.norm(tmp2)
        ret = ((tmp0*tmp2) @ tmp1.T).reshape(-1)
    if return_dm:
        ret = ret[:,np.newaxis] * ret.conj()
    return ret


def rand_density_matrix(N0, k=None, kind='haar', seed=None):
    # http://www.qetlab.com/RandomDensityMatrix
    np_rng = get_numpy_rng(seed)
    assert kind in {'haar','bures'}
    if k is None:
        k = N0
    if kind=='haar':
        ginibre_ensemble = _random_complex(N0, k, seed=np_rng)
    else:
        tmp0 = _random_complex(N0, k, seed=np_rng)
        ginibre_ensemble = (rand_haar_unitary(N0, seed=np_rng) + np.eye(N0)) @ tmp0
    ret = ginibre_ensemble @ ginibre_ensemble.T.conj()
    ret /= np.trace(ret)
    return ret


def rand_separable_dm(N0, N1=None, k=2, seed=None):
    np_rng = get_numpy_rng(seed)
    probability = np_rng.uniform(0, 1, size=k)
    probability /= probability.sum()
    ret = 0
    for ind0 in range(k):
        tmp0 = rand_density_matrix(N0, kind='haar', seed=np_rng)
        tmp1 = rand_density_matrix(N1, kind='haar', seed=np_rng)
        ret = ret + probability[ind0] * np.kron(tmp0, tmp1)
    return ret


def random_near_dm_direction(dm, theta=0.05, seed=None):
    np_rng = get_numpy_rng(seed)
    tmp0 = dm_to_gellmann_basis(dm)
    tmp1 = tmp0 / np.linalg.norm(tmp0)
    tmp2 = tmp1 + np_rng.uniform(-theta, theta, size=tmp1.shape)
    tmp3 = tmp2 / np.linalg.norm(tmp2)
    ret = gellmann_basis_to_dm(tmp3)
    return ret


def random_hermite_matrix(dim, seed=None):
    np_rng = get_numpy_rng(seed)
    tmp0 = np_rng.normal(size=(dim,dim)) + 1j*np_rng.normal(size=(dim,dim))
    ret = (tmp0 + tmp0.T.conj())/2
    return ret
