import torch
import numpy as np
import scipy.optimize
from tqdm import tqdm

import torch_wrapper

from . import dicke
from .random import get_numpy_rng
from ._misc import quantum_relative_entropy, density_matrix_boundary, hf_intertropalte_dm, alpha_to_beta
from .gellmann import dm_to_gellmann_basis
from .cha import get_cha_boundary
from ._purebQ import QuantumPureBosonicExt

class PureBosonicExt(torch.nn.Module):
    def __init__(self, dimA, dimB, kext):
        super().__init__()
        Bij,klist,klist_to_ij = dicke.qudit_dicke_state_partial_trace(dimB, kext)
        self.klist = klist
        tmp0 = [torch.int64,torch.int64,torch.complex128]
        self.Bij = [[torch.tensor(y0,dtype=y1) for y0,y1 in zip(x,tmp0)] for x in Bij]
        tmp0 = np.random.randn(dimA,len(klist)) + 1j*np.random.randn(dimA,len(klist))
        tmp0 /= np.linalg.norm(tmp0.reshape(-1))
        self.pj_real = torch.nn.Parameter(torch.tensor(tmp0.real, dtype=torch.float64, requires_grad=True))
        self.pj_imag = torch.nn.Parameter(torch.tensor(tmp0.imag, dtype=torch.float64, requires_grad=True))
        self.dimA = dimA
        self.dimB = dimB

        # special tag, use it at your own risk
        self.dm_torch = None
        self.dm_target = None
        self.expect_op_T_vec = None
        self.use_gellman_distance = False
        self.logm_image_kind = 'error'
        self.no_imag = False
        self.tag_bosonizeAB = False

    def randomize_parameter(self, seed=None):
        np_rng = get_numpy_rng(seed)
        dimA,dimC = self.pj_real.shape
        tmp0 = np_rng.normal(size=(dimA,dimC)) + 1j*np_rng.normal(size=(dimA,dimC))
        tmp0 /= np.linalg.norm(tmp0.reshape(-1))
        self.pj_real.data[:] = torch.tensor(tmp0.real, dtype=torch.float64, requires_grad=True)
        self.pj_imag.data[:] = torch.tensor(tmp0.imag, dtype=torch.float64, requires_grad=True)
        ret = torch_wrapper.get_model_flat_parameter(self)
        return ret

    def set_dm_target(self, target):
        assert target.ndim in {1,2}
        if target.ndim==1:
            target = target[:,np.newaxis] * target.conj()
        assert (target.shape[0]==target.shape[1])
        self.dm_target = torch.tensor(target, dtype=torch.complex128)

    def set_expectation_op(self, op):
        self.dm_target = None
        self.expect_op_T_vec = torch.tensor(op.T.reshape(-1), dtype=torch.complex128)

    def forward(self):
        if self.no_imag:
            tmp0 = self.pj_real + 0j*self.pj_imag
            # tmp0 = tmp0 * torch.tensor([[1,0,1,0,1],[0,1,0,1,0]])
        else:
            tmp0 = self.pj_real + 1j*self.pj_imag
        if self.tag_bosonizeAB:
            assert (self.dimB==2) and (self.dimA==2)
            tmp0 = bosonizeAB_qubit_dicke_p(tmp0)
        tmp1 = tmp0 / torch.linalg.norm(tmp0.view(-1))
        self.dm_torch = dicke.qudit_partial_trace_AC_to_AB(tmp1, self.Bij)
        if self.dm_target is not None:
            if self.use_gellman_distance:
                tmp0 = dm_to_gellmann_basis(self.dm_target)
                tmp1 = dm_to_gellmann_basis(self.dm_torch)
                loss = torch.sum((tmp0-tmp1)**2)
            else:
                loss = quantum_relative_entropy(self.dm_target, self.dm_torch, kind=self.logm_image_kind)
        else:
            loss = torch.dot(self.dm_torch.view(-1), self.expect_op_T_vec).real
        return loss

    def minimize_loss(self, num_repeat, print_freq=0, tol=1e-10, return_info=False, seed=None):
        hf_model = torch_wrapper.hf_model_wrapper(self)
        ret = []
        np_rng = get_numpy_rng(seed)
        for _ in range(num_repeat):
            hf_callback = torch_wrapper.hf_callback_wrapper(hf_model, print_freq=print_freq)
            theta0 = self.randomize_parameter(np_rng)
            optimize_bound = np.array([[-1,1]])*np.ones((len(theta0),1))
            theta_optim = scipy.optimize.minimize(hf_model, theta0, method='L-BFGS-B',
                    bounds=optimize_bound, tol=tol, jac=True, callback=hf_callback)
            ret.append(theta_optim)
        ret = min(ret, key=lambda x: x.fun)
        if not return_info:
            ret = ret.fun
        return ret

    def minimize_distance(self, dm_target, num_repeat=1, distance_kind='REE', print_freq=0, tol=1e-10, return_info=False, seed=None):
        assert distance_kind in {'REE','Euclidean'}
        if distance_kind=='Euclidean':
            self.use_gellman_distance = True #experiments find this doesn't help too much
        self.set_dm_target(dm_target)
        ret = self.minimize_loss(num_repeat, print_freq, tol, return_info, seed)

        # self.dm_target = torch.tensor(dm_target, dtype=torch.complex128)
        # hf_model = torch_wrapper.hf_model_wrapper(self)
        # history_info = dict()
        # hf_callback = torch_wrapper.hf_callback_wrapper(hf_model, history_info, print_freq=print_freq)
        # ret = []
        # for _ in range(num_repeat):
        #     theta0 = self.randomize_parameter()
        #     optimize_bound = np.array([[-1,1]])*np.ones((len(theta0),1))
        #     theta_optim = scipy.optimize.minimize(hf_model, theta0, method='L-BFGS-B',
        #             bounds=optimize_bound, tol=tol, jac=True, callback=hf_callback)
        #     ret.append(theta_optim)
        # ret = min(ret, key=lambda x: x.fun)
        # if not return_info:
        #     ret = ret.fun

        self.use_gellman_distance = False
        return ret

    def solve_boundary(self, dm_target, alpha_cha=None, xtol=1e-4, threshold=1e-7, num_repeat=1, print_freq=0, use_tqdm=True, seed=None):
        # TODO replace with alpha_lower_bound
        # TODO replace alpha with beta
        # TODO replace alpha_lower_bound with 0
        if alpha_cha is None:
            alpha_cha = get_cha_boundary(dm_target, self.dimA, num_repeat=3)[0]
        alpha0 = alpha_cha
        alpha1 = density_matrix_boundary(dm_target, return_alpha=True)[0]
        np_rng = get_numpy_rng(seed)
        history_alpha_list = []
        maxiter = int(np.ceil(np.log2(max(2, (alpha1-alpha0)/xtol))))
        tmp0 = tqdm(range(maxiter)) if use_tqdm else range(maxiter)
        for _ in tmp0:
            alpha_i = (alpha0+alpha1)/2
            distance_i = self.minimize_distance(hf_intertropalte_dm(dm_target, alpha_i),
                        num_repeat=num_repeat, distance_kind='REE', print_freq=print_freq, seed=np_rng)
            history_alpha_list.append((alpha_i,distance_i))
            if distance_i>=threshold:
                alpha1 = alpha_i
            else:
                alpha0 = alpha_i
        history_alpha_list = np.array(sorted(history_alpha_list, key=lambda x: x[0]))
        ret_alpha = alpha_i
        ret_beta = alpha_to_beta(dm_target, ret_alpha)
        return ret_alpha,ret_beta,history_alpha_list



