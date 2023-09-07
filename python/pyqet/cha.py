import torch
import numpy as np
import cvxpy
from tqdm import tqdm
import contextlib
import scipy.linalg
import concurrent.futures
import multiprocessing

import torch_wrapper

_num_cpu = multiprocessing.cpu_count()


from .random import get_numpy_rng, rand_haar_state, _random_complex
from .gellmann import dm_to_gellmann_basis, dm_to_gellmann_distance
from . import ppt
from ._misc import quantum_relative_entropy, state_to_dm


def _set_num_cpu(x):
    global _num_cpu
    _num_cpu = int(x)

def rand_norm_bounded_hermitian(N0, norm2_bound, seed=None):
    # |A|_2 <= |A|_F
    # https://math.stackexchange.com/a/252831
    tmp0 = _random_complex(N0, N0, seed=seed)
    ret = tmp0 + tmp0.T.conj()
    norm = np.linalg.norm(ret, ord='fro')
    ret *= (norm2_bound/norm)
    return ret


def convex_hull_approximation_iterative(rho, dimA, max_pure_state=2000, maxiter=100, norm2_init=1, decay_rate=0.97, seed=None, use_tqdm=True, zero_eps=1e-7):
    # CHA with bagging 10.1103/PhysRevA.98.012315
    np_rng = get_numpy_rng(seed)
    assert (rho.ndim==2) and (rho.shape[0]==rho.shape[1]) and (rho.shape[0]%dimA==0)
    dimB = rho.shape[0]//dimA
    rho_glm = dm_to_gellmann_basis(rho)
    # compared to original paper, we append the term eye (I) here
    ketA = np.zeros((max_pure_state,dimA), dtype=np.complex128)
    ketB = np.zeros((max_pure_state,dimB), dtype=np.complex128)
    for ind0 in range(max_pure_state):
        ketA[ind0] = rand_haar_state(dimA, seed=np_rng)
        ketB[ind0] = rand_haar_state(dimB, seed=np_rng)

    cvx_alpha = cvxpy.Variable(name='alpha')
    cvx_lambda = cvxpy.Variable(max_pure_state, name='lambda')
    cvx_A = cvxpy.Parameter((max_pure_state,rho_glm.shape[0]))
    cvx_obj = cvxpy.Maximize(cvx_alpha)
    cvx_constrants = [
        cvx_alpha*rho_glm==cvx_A.T@cvx_lambda,
        cvx_lambda>=0,
        cvxpy.sum(cvx_lambda)==1,
    ]
    cvx_problem = cvxpy.Problem(cvx_obj, cvx_constrants)
    def hf_cvxpy_solve():
        cvx_A.value = np.stack([dm_to_gellmann_basis(state_to_dm(np.kron(x,y))) for x,y in zip(ketA,ketB)])
        ret = cvx_problem.solve(ignore_dpp=True) #ECOS solver
        assert ret is not None, 'may due to that max_pure_state is not large enough'
        assert cvx_lambda.value is not None, 'may due to that max_pure_state is not large enough'
        return ret
    alpha_history = [hf_cvxpy_solve()]

    norm2_bound = norm2_init
    with (tqdm(range(maxiter)) if use_tqdm else contextlib.nullcontext()) as pbar:
        for _ in (pbar if use_tqdm else range(maxiter)):
            if use_tqdm:
                pbar.postfix = f'alpha={alpha_history[-1]:.5f}, eps={norm2_bound:.4f}'
            ind0 = cvx_lambda.value > zero_eps
            num_keep = np.sum(ind0)
            ketA[:num_keep] = ketA[ind0]
            ketB[:num_keep] = ketB[ind0]
            tmp0 = np_rng.integers(num_keep, size=(max_pure_state-num_keep))
            for ind1 in range(num_keep,max_pure_state):
                tmp1 = scipy.linalg.expm(1j*rand_norm_bounded_hermitian(dimA, norm2_bound, np_rng))
                ketA[ind1] = tmp1 @ ketA[tmp0[ind1-num_keep]]
                tmp1 = scipy.linalg.expm(1j*rand_norm_bounded_hermitian(dimB, norm2_bound, np_rng))
                ketB[ind1] = tmp1 @ ketB[tmp0[ind1-num_keep]]
            norm2_bound *= decay_rate
            alpha_history.append(hf_cvxpy_solve())
    ind0 = cvx_lambda.value > 0
    return alpha_history,ketA[ind0],ketB[ind0],cvx_lambda.value[ind0]


# TODO remove _hf_cha_boundary_one, call convex_hull_approximation_iterative directly
def _hf_cha_boundary_one(dm_target, dimA, use_tqdm, zero_eps):
    alpha_cha = convex_hull_approximation_iterative(dm_target, dimA, max_pure_state=2000, maxiter=100,
            norm2_init=1, decay_rate=0.97, use_tqdm=use_tqdm, zero_eps=zero_eps)[0][-1]
    beta_cha = dm_to_gellmann_distance(alpha_cha*dm_target)
    return alpha_cha,beta_cha
    # ret_list = []
    # for _ in range(num_repeat):
    #     ret_list.append((alpha_cha,beta_cha))
    # alpha_cha,beta_cha = sorted(ret_list, key=lambda x: x[1])[-1] #max
    # return alpha_cha,beta_cha


def get_cha_boundary(dm_target, dimA, num_repeat=1, zero_eps=1e-7):
    is_np = isinstance(dm_target, np.ndarray) and dm_target.ndim==2
    if is_np:
        dm_target_list = [dm_target]
    else:
        dm_target_list = dm_target
    num_worker = max(1, min(_num_cpu, len(dm_target_list)*num_repeat))
    if num_worker==1:
        ret = _hf_cha_boundary_one(dm_target_list[0], dimA, use_tqdm=True, zero_eps=zero_eps)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_worker) as executor:
            job_list = [executor.submit(_hf_cha_boundary_one, x, dimA, use_tqdm=False, zero_eps=zero_eps) for x in dm_target for _ in range(num_repeat)]
            ret = np.array([x.result() for x in tqdm(job_list)]).reshape(len(dm_target_list), num_repeat, 2)
            if num_repeat==1:
                ret = ret[:,0].T
            else:
                ind0 = np.argmax(ret[:,:,1], axis=1)
                ret = ret[np.arange(ret.shape[0]), ind0].T
        if is_np:
            ret = [x[0] for x in ret]
    #alpha_cha,beta_cha
    return ret


def get_ppt_cha_boundary(dm_target, dimA, dimB, num_repeat=1, zero_eps=1e-7):
    is_np = isinstance(dm_target, np.ndarray)
    if is_np:
        dm_target_list = [dm_target]
    else:
        dm_target_list = dm_target
    alpha_ppt,beta_ppt = np.array([ppt.get_boundary(x, dimA, dimB) for x in dm_target_list]).T
    if is_np:
        alpha_ppt = alpha_ppt[0]
        beta_ppt = beta_ppt[0]
    alpha_cha,beta_cha = get_cha_boundary(dm_target_list, dimA, num_repeat=num_repeat, zero_eps=zero_eps)
    return alpha_ppt,beta_ppt,alpha_cha,beta_cha



class AutodiffREE(torch.nn.Module):
    def __init__(self, num_state, dim0, dim1):
        super().__init__()
        self.num_state = num_state
        self.dim0 = dim0
        self.dim1 = dim1
        self.set_random_initial_state()
        self.dm_target = None
        self.dm_torch = None
        self.expect_op_T_vec = None
        self.use_gellman_distance = False

    def set_dm_target(self, dm):
        assert dm.shape[0]==(self.state0_real.shape[1]*self.state1_real.shape[1])
        assert dm.shape[0]==dm.shape[1]
        self.dm_target = torch.tensor(dm, dtype=torch.complex128)

    def set_expectation_op(self, op):
        self.dm_target = None
        self.expect_op_T_vec = torch.tensor(op.T.reshape(-1), dtype=torch.complex128)

    def set_random_initial_state(self, seed=None):
        np_rng = get_numpy_rng(seed)
        tmp0 = np.stack([rand_haar_state(self.dim0, np_rng) for _ in range(self.num_state)])
        tmp1 = np.stack([rand_haar_state(self.dim1, np_rng) for _ in range(self.num_state)])
        tmp2 = np_rng.uniform(0, 1, size=self.num_state)
        tmp2 = tmp2 / tmp2.sum()
        if not hasattr(self, 'state0_real'):
            self.state0_real = torch.nn.Parameter(torch.tensor(tmp0.real, dtype=torch.float64, requires_grad=True))
            self.state0_imag = torch.nn.Parameter(torch.tensor(tmp0.imag, dtype=torch.float64, requires_grad=True))
            self.state1_real = torch.nn.Parameter(torch.tensor(tmp1.real, dtype=torch.float64, requires_grad=True))
            self.state1_imag = torch.nn.Parameter(torch.tensor(tmp1.imag, dtype=torch.float64, requires_grad=True))
            self.probability = torch.nn.Parameter(torch.tensor(tmp2/tmp2.sum(), dtype=torch.float64, requires_grad=True))
        else:
            self.state0_real.data[:] = torch.from_numpy(tmp0.real)
            self.state0_imag.data[:] = torch.from_numpy(tmp0.imag)
            self.state1_real.data[:] = torch.from_numpy(tmp1.real)
            self.state1_imag.data[:] = torch.from_numpy(tmp1.imag)
            self.probability.data[:] = torch.from_numpy(tmp2)
        # ret = np.concatenate([tmp2, tmp0.imag.reshape(-1), tmp0.real.reshape(-1), tmp1.imag.reshape(-1), tmp1.real.reshape(-1)])
        # return ret

    def forward(self):
        tmp0 = self.state0_real + 1j*self.state0_imag
        state0 = tmp0 / torch.linalg.norm(tmp0, ord=2, dim=1, keepdim=True)
        tmp0 = self.state1_real + 1j*self.state1_imag
        state1 = tmp0 / torch.linalg.norm(tmp0, ord=2, dim=1, keepdim=True)
        probability = torch.maximum(torch.tensor(0, dtype=torch.float64), self.probability)
        tmp0 = probability / probability.sum()
        tmp1 = (state0.view(state0.shape[0],-1,1) * state1.view(state0.shape[0],1,-1)).view(self.num_state,-1,1)
        rho = torch.sum((tmp1 * tmp1.conj().view(self.num_state,1,-1))*tmp0.view(-1,1,1), dim=0)
        self.dm_torch = rho
        if self.dm_target is not None:
            if self.use_gellman_distance:
                tmp0 = dm_to_gellmann_basis(self.dm_target)
                tmp1 = dm_to_gellmann_basis(rho)
                loss = torch.sum((tmp0-tmp1)**2)
            else:
                loss = quantum_relative_entropy(self.dm_target, rho)
        else:
            loss = torch.dot(rho.view(-1), self.expect_op_T_vec).real
        return loss

    def get_parameter_bounds(self):
        parameter_sorted = sorted(list(self.named_parameters()), key=lambda x:x[0])
        tmp0 = np.cumsum(np.array([0] + [y.numel() for x,y in parameter_sorted])).tolist()
        tmp1 = [i for i,(k,v) in enumerate(parameter_sorted) if k=='probability'][0]
        index01 = list(zip(tmp0[:-1],tmp0[1:]))[tmp1]
        num_parameter = tmp0[-1]
        bounds = np.zeros((num_parameter,2), dtype=np.float64)
        bounds[:, 0] = -1
        bounds[:, 1] = 1
        bounds[index01[0]:index01[1],0] = 0
        return bounds

    def _minimize_loss_i(self, maxiter=100, zero_tol=1e-5, seed=None, use_tqdm=False):
        optimizer = torch.optim.Adam(self.parameters())
        np_rng = np.random.default_rng(seed)
        best_theta = None
        best_loss = None
        loss_history = []
        with (tqdm(range(maxiter)) if use_tqdm else contextlib.nullcontext()) as pbar:
            for ind_step in (pbar if use_tqdm else range(maxiter)):
                optimizer.zero_grad()
                loss = self()
                loss.backward()
                optimizer.step()
                ind0 = self.probability < zero_tol
                num0 = ind0.sum().item()
                if num0>0: # remove those useless state_vector
                    tmp0 = np.stack([rand_haar_state(self.dim0, np_rng) for _ in range(num0)])
                    tmp1 = np.stack([rand_haar_state(self.dim1, np_rng) for _ in range(num0)])
                    self.state0_real.data[ind0,:] = torch.tensor(tmp0.real)
                    self.state0_imag.data[ind0,:] = torch.tensor(tmp0.imag)
                    self.state1_real.data[ind0,:] = torch.tensor(tmp1.real)
                    self.state1_imag.data[ind0,:] = torch.tensor(tmp1.imag)
                    # TODO reset probability[ind0] to 1 (or 10*zero_tol)
                loss_history.append(loss.item())
                if use_tqdm and (ind_step%10==0):
                    pbar.set_postfix(loss=f'{loss_history[-1]:.5f}')
                if (best_loss is None) or (best_loss > loss_history[-1]):
                    best_theta = torch_wrapper.get_model_flat_parameter(self)
                    best_loss = loss_history[-1]
        return best_loss,best_theta,loss_history

    def minimize_loss(self, maxiter=100, num_repeat=1, zero_tol=1e-5, seed=None, use_tqdm=False):
        np_rng = get_numpy_rng(seed)
        ret = []
        for _ in range(num_repeat):
            self.set_random_initial_state(seed)
            ret.append(self._minimize_loss_i(maxiter, zero_tol, np_rng, use_tqdm))
        ret = sorted(ret, key=lambda x:x[0])[0]
        return ret

    def minimize_distance(self, dm_target, maxiter=100, num_repeat=1, zero_tol=1e-5, use_tqdm=True, distance_kind='REE', seed=None):
        assert distance_kind in {'REE', 'Euclidean'}
        self.use_gellman_distance = (distance_kind=='Euclidean')
        self.dm_target = torch.tensor(dm_target, dtype=torch.complex128)
        best_loss,best_theta,loss_history = self.minimize_loss(maxiter, num_repeat, zero_tol, seed=seed, use_tqdm=use_tqdm)
        self.use_gellman_distance = False
        return best_loss,best_theta,loss_history
        # BFGS / LBFGS is not quite suitable for CHA kind optimization problem, most of paramters are wasted
        # hf_model = torch_wrapper.hf_model_wrapper(self)
        # history_info = dict()
        # hf_callback = torch_wrapper.hf_callback_wrapper(hf_model, history_info, print_freq=0)
        # ret = []
        # for _ in range(num_repeat):
        #     theta0 = self.set_random_initial_state()
        #     optimize_bound = np.array([[-1,1]])*np.ones((len(theta0),1))
        #     theta_optim = scipy.optimize.minimize(hf_model, theta0, method='L-BFGS-B',
        #             bounds=optimize_bound, tol=1e-10, jac=True, callback=hf_callback)
        #     ret.append(theta_optim.fun)

    # bad idea, no sharp boundary
    def solve_boundary(self, dm_target, alpha_ppt=None, xtol=1e-4, threshold=1e-7, num_repeat=1):
        if alpha_ppt is None:
            alpha_ppt = ppt.get_boundary(dm_target, self.dim0, self.dim1)[0]
        alpha0 = 0
        alpha1 = alpha_ppt
        maxiter = int(np.ceil(np.log2((alpha1 - alpha0)/xtol)))
        distance_list = []
        for _ in range(maxiter):
            alpha_i = (alpha0 + alpha1) / 2
            distance_i = self.minimize_distance(dm_target, num_repeat=num_repeat)[0]
            distance_list.append((alpha_i,distance_i))
            if distance_i>=threshold:
                alpha1 = alpha_i
            else:
                alpha0 = alpha_i
        distance_list = np.array(sorted(distance_list,key=lambda x: x[0]))
        ret_alpha = alpha_i
        ret_beta = dm_to_gellmann_distance(ret_alpha*dm_target)
        return ret_alpha, ret_beta, distance_list


class AutodiffCHAAlpha(torch.nn.Module):
    def __init__(self, num_state, dimA, dimB):
        super().__init__()
        self.num_state = num_state
        self.dimA = dimA
        self.dimB = dimB
        self.set_random_initial_state()
        self.dm_target = None
        self.dm_target_glm = None
        self.dm_torch = None
        self.loss_L1 = None

        import cvxpylayers.torch
        dim_glm = dimA*dimA*dimB*dimB - 1
        cvx_alpha = cvxpy.Variable(name='alpha')
        cvx_lambda = cvxpy.Variable(num_state, name='lambda')
        cvx_rho = cvxpy.Parameter((num_state,dim_glm))
        cvx_rho_target = cvxpy.Parameter(dim_glm)
        cvx_obj = cvxpy.Maximize(cvx_alpha)
        cvx_constrants = [
            cvx_alpha*cvx_rho_target==cvx_rho.T@cvx_lambda,
            cvx_lambda>=0,
            cvxpy.sum(cvx_lambda)==1,
        ]
        cvx_problem = cvxpy.Problem(cvx_obj, cvx_constrants)
        self.cvxpylayer = cvxpylayers.torch.CvxpyLayer(cvx_problem, parameters=[cvx_rho,cvx_rho_target], variables=[cvx_alpha,cvx_lambda])
        self.solver_args = None

    def set_random_initial_state(self, seed=None):
        np_rng = get_numpy_rng(seed)
        tmp0 = np.stack([rand_haar_state(self.dimA, np_rng) for _ in range(self.num_state)])
        tmp1 = np.stack([rand_haar_state(self.dimB, np_rng) for _ in range(self.num_state)])
        tmp2 = np_rng.uniform(0, 1, size=self.num_state)
        tmp2 = tmp2 / tmp2.sum()
        if not hasattr(self, 'state0_real'):
            self.state0_real = torch.nn.Parameter(torch.tensor(tmp0.real, dtype=torch.float64))
            self.state0_imag = torch.nn.Parameter(torch.tensor(tmp0.imag, dtype=torch.float64))
            self.state1_real = torch.nn.Parameter(torch.tensor(tmp1.real, dtype=torch.float64))
            self.state1_imag = torch.nn.Parameter(torch.tensor(tmp1.imag, dtype=torch.float64))
        else:
            self.state0_real.data[:] = torch.from_numpy(tmp0.real)
            self.state0_imag.data[:] = torch.from_numpy(tmp0.imag)
            self.state1_real.data[:] = torch.from_numpy(tmp1.real)
            self.state1_imag.data[:] = torch.from_numpy(tmp1.imag)

    def set_dm_target(self, dm):
        self.dm_target = torch.tensor(dm, dtype=torch.complex128)
        tmp0 = np.linalg.eigvalsh(self.dm_target.numpy())[0]
        tmp1 = 1-self.dm_target.shape[0]*tmp0
        assert tmp1>0
        alpha_upper_bound = 1/tmp1

    def solve(self, dm, maxiter=100, zero_tol=1e-5, seed=None, solver_eps=1e-5):
        # not recommand to use scipy.optimize.minimize, most of state-vectors make no contribution
        self.solver_args = dict(eps=solver_eps) #solver=ECS
        assert dm.shape[0]==(self.state0_real.shape[1]*self.state1_real.shape[1])
        assert dm.shape[0]==dm.shape[1]
        self.dm_target = torch.tensor(dm, dtype=torch.complex128)
        self.dm_target_glm = dm_to_gellmann_basis(self.dm_target)
        optimizer = torch.optim.Adam(self.parameters())
        np_rng = np.random.default_rng(seed=seed)
        best_alpha = None
        best_state = None
        best_lambda_ = None
        with tqdm(range(maxiter)) as pbar:
            for _ in pbar:
                optimizer.zero_grad()
                loss = self()
                loss.backward()
                optimizer.step()
                ind0 = self.lambda_<zero_tol
                num0 = ind0.sum().item()
                if num0>0:
                    tmp0 = np.stack([rand_haar_state(self.dimA, np_rng) for _ in range(num0)])
                    tmp1 = np.stack([rand_haar_state(self.dimB, np_rng) for _ in range(num0)])
                    self.state0_real.data[ind0,:] = torch.tensor(tmp0.real)
                    self.state0_imag.data[ind0,:] = torch.tensor(tmp0.imag)
                    self.state1_real.data[ind0,:] = torch.tensor(tmp1.real)
                    self.state1_imag.data[ind0,:] = torch.tensor(tmp1.imag)
                pbar.set_postfix(alpha=f'{loss.item():.5f}')
                if (best_alpha is None) or (-loss.item()>best_alpha):
                    best_alpha = -loss.item()
                    tmp0 = [x.detach().numpy() for x in [self.state0_real,self.state0_imag,self.state1_real,self.state1_imag]]
                    best_state = tmp0[0]+1j*tmp0[1], tmp0[2]+1j*tmp0[3]
                    best_lambda_ = self.lambda_.copy()
        return best_alpha,best_state,best_lambda_

    def forward(self):
        # TODO torch.nan_to_num
        tmp0 = self.state0_real + 1j*self.state0_imag
        state0 = tmp0 / torch.linalg.norm(tmp0, ord=2, dim=1, keepdim=True)
        tmp0 = self.state1_real + 1j*self.state1_imag
        state1 = tmp0 / torch.linalg.norm(tmp0, ord=2, dim=1, keepdim=True)

        tmp1 = (state0.view(state0.shape[0],-1,1) * state1.view(state0.shape[0],1,-1)).view(self.num_state,-1,1)
        rho = (tmp1 * tmp1.conj().view(self.num_state,1,-1))
        tmp0 = dm_to_gellmann_basis(rho)
        tmp1 = dm_to_gellmann_basis(self.dm_target)
        alpha_,lambda_ = self.cvxpylayer(tmp0, tmp1, solver_args=self.solver_args)
        loss = -alpha_
        self.lambda_ = lambda_.detach().numpy().copy()
        return loss

    def get_parameter_bounds(self):
        parameter_sorted = sorted(list(self.named_parameters()), key=lambda x:x[0])
        tmp0 = np.cumsum(np.array([0] + [y.numel() for x,y in parameter_sorted])).tolist()
        index01 = list(zip(tmp0[:-1],tmp0[1:]))
        num_parameter = tmp0[-1]
        ret = [(None,None) for _ in range(num_parameter)]
        tmp1 = [('state0_real',None,None), ('state0_imag',None,None), ('state1_real',None,None), ('state1_imag',None,None)]
        for key,lower,upper in tmp1:
            tmp2 = [i for i,(k,v) in enumerate(parameter_sorted) if k==key][0]
            for x in range(*index01[tmp2]):
                ret[x] = lower,upper
        return ret
