import time
import pickle
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
tableau = ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1', '#C85200', '#898989', '#A2C8EC', '#FFBC79', '#CFCFCF']

import pyqet

torch.set_num_threads(1)

np_rng = np.random.default_rng()


def demo_werner2_kext_boundary():
    dim = 2
    dm0 = pyqet.entangle.get_werner_state(dim, alpha=1)
    dm_norm = pyqet.gellmann.dm_to_gellmann_norm(dm0)
    kext_list = [4,5,6,7,8,9,10,11,12,16,512,8192,65536]
    beta_list = []
    werner_alpha_list = []
    kwargs = dict(xtol=1e-5, num_repeat=3, threshold=1e-7, converge_tol=1e-10)
    for kext in kext_list:
        model = pyqet.entangle.PureBosonicExt(dim, dim, kext=kext, distance_kind='ree')
        beta_i = model.get_boundary(dm0, **kwargs)
        werner_alpha_i = (dim*dim*beta_i)/(2*dm_norm+dim*beta_i)
        beta_list.append(beta_i)
        werner_alpha_list.append(werner_alpha_i)
        print(kext, werner_alpha_i)
    beta_list = np.array(beta_list)
    werner_alpha_list = np.array(werner_alpha_list)

    ## matlab code here
    # use_PPT = 0;
    # na = 2;
    # hfBOS = @(alpha,kext) SymmetricExtension(full(WernerState(na, alpha)), kext, [na,na], use_PPT, 1);
    # hfnoBOS = @(alpha,kext) SymmetricExtension(full(WernerState(na, alpha)), kext, [na,na], use_PPT, 0);

    # Werner(d=2) boundary, alpha(SVQC//qetlab-BOS/qetlab)
    # k     PureB  QETLAB-BOS  QETLAB-Sym
    # 4     0.64585  0.6668   0.6668
    # 5     0.63661  0.63650  0.63650
    # 6     0.61206  0.61556  0.61554
    # 7     0.60019  0.60016  0.60017
    # 8     0.58846  0.58840  NA
    # 9     0.57918  0.57919  NA
    # 10    0.57165  0.57168  NA
    # 11    0.56547  0.56548  NA
    # 12    0.56024  NA       NA
    # 16    0.54575  NA       NA
    # 51 2  0.50175  NA       NA
    # 8192  0.50040  NA       NA
    # 65536 0.50034  NA       NA


def _ax_plot(ax, data_dict, alpha_boundary, axin_xticks, title=None):
    alpha_list = data_dict['alpha_list']
    kext_list = data_dict['kext_list']
    ree_pureb = data_dict['ree_pureb']
    ree_analytical = data_dict['ree_analytical']
    ree_ppt = data_dict['ree_ppt']
    ree_cha = data_dict['ree_cha']

    for ind0 in range(len(kext_list)):
        ax.plot(alpha_list, ree_pureb[ind0], color=tableau[ind0+3], label=f'PureB({kext_list[ind0]})')
    ax.plot(alpha_list, ree_analytical, ':', color=tableau[0], markersize=3, label='analytical', markerfacecolor='none')
    ax.plot(alpha_list, ree_ppt, '+', color=tableau[1], label='PPT')
    ax.plot(alpha_list, ree_cha, 'x', color=tableau[2], label='CHA')
    ax.legend(ncol=2, fontsize=12, loc='upper left')
    ax.set_xlim(0, 1)
    ax.set_xlabel(r'$\alpha$', fontsize=12)
    ax.set_ylabel('REE', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=11)
    if title is not None:
        ax.set_title(title)

    alpha_fine_list = data_dict['alpha_fine_list']
    ree_pureb_fine = data_dict['ree_pureb_fine']
    ree_fine_ppt = data_dict['ree_fine_ppt']
    ree_fine_analytical = data_dict['ree_fine_analytical']
    ree_fine_cha = data_dict['ree_fine_cha']
    axin = ax.inset_axes([0.1, 0.24, 0.47, 0.47])

    for ind0 in range(len(kext_list)):
        axin.plot(alpha_fine_list, ree_pureb_fine[ind0], color=tableau[ind0+3])
    axin.plot(alpha_fine_list, ree_fine_analytical, ':', color=tableau[0], markersize=3, markerfacecolor='none')
    tmp0 = ree_fine_ppt.copy()
    tmp0[alpha_fine_list<alpha_boundary] = np.nan
    axin.plot(alpha_fine_list, tmp0, '+', color=tableau[1])
    axin.plot(alpha_fine_list, ree_fine_cha, 'x', color=tableau[2])
    axin.set_xlim(alpha_fine_list[0], alpha_fine_list[-1])
    axin.set_yscale('log')
    axin.tick_params(axis='both', which='major', labelsize=11)
    axin.set_xticks(axin_xticks)
    hrect,hpatch = ax.indicate_inset_zoom(axin, edgecolor="red")
    hrect.set_xy((hrect.get_xy()[0], -0.02))
    hrect.set_height(0.05)

def demo_werner3_ree():
    dim = 3

    alpha_boundary = 1/dim
    alpha_list = np.linspace(0, 1, 50, endpoint=True) #alpha=1 is unstable for analytical
    if dim==2:
        alpha_fine_list = np.linspace(alpha_boundary*0.95, min(1,alpha_boundary*1.2), 50)#dim=2
    else:
        assert dim==3
        alpha_fine_list = np.linspace(alpha_boundary*0.95, min(1,alpha_boundary*1.4), 50)#dim=3

    kext_list = [8, 16, 32, 64]
    kwargs = dict(num_repeat=3, tol=1e-10, print_every_round=0)
    ree_pureb = []
    ree_pureb_fine = []
    for kext in kext_list:
        model = pyqet.entangle.PureBosonicExt(dim, dim, kext=kext, distance_kind='ree')
        for alpha_i in tqdm(alpha_list):
            model.set_dm_target(pyqet.entangle.get_werner_state(dim, alpha_i))
            ree_pureb.append(pyqet.optimize.minimize(model, **kwargs).fun)
        for alpha_i in tqdm(alpha_fine_list):
            model.set_dm_target(pyqet.entangle.get_werner_state(dim, alpha_i))
            ree_pureb_fine.append(pyqet.optimize.minimize(model, **kwargs).fun)
    ree_pureb = np.array(ree_pureb).reshape(len(kext_list), len(alpha_list))
    ree_pureb_fine = np.array(ree_pureb_fine).reshape(len(kext_list), len(alpha_fine_list))

    model = pyqet.entangle.AutodiffCHAREE(dim, dim, distance_kind='ree')
    kwargs = dict(num_repeat=3, tol=1e-10, print_every_round=0)
    ree_cha = []
    ree_fine_cha = []
    for alpha_i in tqdm(alpha_list):
        model.set_dm_target(pyqet.entangle.get_werner_state(dim, alpha_i))
        ree_cha.append(pyqet.optimize.minimize(model, **kwargs).fun)
    for alpha_i in tqdm(alpha_fine_list):
        model.set_dm_target(pyqet.entangle.get_werner_state(dim, alpha_i))
        ree_fine_cha.append(pyqet.optimize.minimize(model, **kwargs).fun)
    ree_cha = np.array(ree_cha)
    ree_fine_cha = np.array(ree_fine_cha)

    ree_analytical = np.array([pyqet.entangle.get_werner_state_ree(dim, x) for x in alpha_list])
    ree_fine_analytical = np.array([pyqet.entangle.get_werner_state_ree(dim, x) for x in alpha_fine_list])

    tmp0 = np.stack([pyqet.entangle.get_werner_state(dim,x) for x in alpha_list])
    ree_ppt = pyqet.entangle.get_ppt_ree(tmp0, dim, dim, use_tqdm=True)
    tmp0 = np.stack([pyqet.entangle.get_werner_state(dim,x) for x in alpha_fine_list])
    ree_fine_ppt = pyqet.entangle.get_ppt_ree(tmp0, dim, dim, use_tqdm=True)

    # with open('data/werner3_ree.pkl', 'wb') as fid:
    #     tmp0 = dict(alpha_list=alpha_list, kext_list=kext_list, ree_pureb=ree_pureb, ree_analytical=ree_analytical,
    #             ree_ppt=ree_ppt, ree_cha=ree_cha, alpha_fine_list=alpha_fine_list, ree_pureb_fine=ree_pureb_fine,
    #             ree_fine_analytical=ree_fine_analytical, ree_fine_ppt=ree_fine_ppt, ree_fine_cha=ree_fine_cha)
    #     pickle.dump(tmp0, fid)

    with open('data/werner3_ree.pkl', 'rb') as fid:
        data_dict = pickle.load(fid)
    dim = 3
    alpha_boundary = 1/dim
    fig,ax = plt.subplots(figsize=(6.4, 4.8))
    axin_xticks = [0.33, 0.37, 0.41, 0.45] if (dim==3) else [0.5, 0.54, 0.58]
    _ax_plot(ax, data_dict, alpha_boundary, axin_xticks)
    fig.tight_layout()
    fig.savefig('tbd00.png', dpi=200)
    # fig.savefig('data/werner3_ree.png', dpi=200)
    # fig.savefig('data/werner3_ree.pdf')


def demo_isotropic3_ree():
    dim = 3

    alpha_boundary = 1/(dim+1)
    alpha_list = np.linspace(0, 1, 50, endpoint=True) #alpha=1 is unstable for analytical
    alpha_fine_list = np.linspace(alpha_boundary*0.95, min(1,alpha_boundary*1.44), 50)

    kext_list = [8, 16, 32, 64]
    kwargs = dict(num_repeat=3, tol=1e-10, print_every_round=0)
    ree_pureb = []
    ree_pureb_fine = []
    for kext in kext_list:
        model = pyqet.entangle.PureBosonicExt(dim, dim, kext=kext, distance_kind='ree')
        for alpha_i in tqdm(alpha_list):
            model.set_dm_target(pyqet.entangle.get_isotropic_state(dim, alpha_i))
            ree_pureb.append(pyqet.optimize.minimize(model, **kwargs).fun)
        for alpha_i in tqdm(alpha_fine_list):
            model.set_dm_target(pyqet.entangle.get_isotropic_state(dim, alpha_i))
            ree_pureb_fine.append(pyqet.optimize.minimize(model, **kwargs).fun)
    ree_pureb = np.array(ree_pureb).reshape(len(kext_list), len(alpha_list))
    ree_pureb_fine = np.array(ree_pureb_fine).reshape(len(kext_list), len(alpha_fine_list))

    model = pyqet.entangle.AutodiffCHAREE(dim, dim, distance_kind='ree')
    kwargs = dict(num_repeat=3, tol=1e-10, print_every_round=0)
    ree_cha = []
    ree_fine_cha = []
    for alpha_i in tqdm(alpha_list):
        model.set_dm_target(pyqet.entangle.get_isotropic_state(dim, alpha_i))
        ree_cha.append(pyqet.optimize.minimize(model, **kwargs).fun)
    for alpha_i in tqdm(alpha_fine_list):
        model.set_dm_target(pyqet.entangle.get_isotropic_state(dim, alpha_i))
        ree_fine_cha.append(pyqet.optimize.minimize(model, **kwargs).fun)
    ree_cha = np.array(ree_cha)
    ree_fine_cha = np.array(ree_fine_cha)

    ree_analytical = np.array([pyqet.entangle.get_isotropic_state_ree(dim, x) for x in alpha_list])
    ree_fine_analytical = np.array([pyqet.entangle.get_isotropic_state_ree(dim, x) for x in alpha_fine_list])

    tmp0 = np.stack([pyqet.entangle.get_isotropic_state(dim,x) for x in alpha_list])
    ree_ppt = pyqet.entangle.get_ppt_ree(tmp0, dim, dim, use_tqdm=True)
    tmp0 = np.stack([pyqet.entangle.get_isotropic_state(dim,x) for x in alpha_fine_list])
    ree_fine_ppt = pyqet.entangle.get_ppt_ree(tmp0, dim, dim, use_tqdm=True)
    # with open('data/isotropic3_ree.pkl', 'wb') as fid:
    #     tmp0 = dict(alpha_list=alpha_list, kext_list=kext_list, ree_pureb=ree_pureb, ree_analytical=ree_analytical,
    #             ree_ppt=ree_ppt, ree_cha=ree_cha, alpha_fine_list=alpha_fine_list, ree_pureb_fine=ree_pureb_fine,
    #             ree_fine_analytical=ree_fine_analytical, ree_fine_ppt=ree_fine_ppt, ree_fine_cha=ree_fine_cha)
    #     pickle.dump(tmp0, fid)

    with open('data/isotropic3_ree.pkl', 'rb') as fid:
        data_dict = pickle.load(fid)
    dim = 3
    alpha_boundary = 1/(dim+1)
    fig,ax = plt.subplots(figsize=(6.4, 4.8))
    axin_xticks = [0.25, 0.3, 0.35]
    _ax_plot(ax, data_dict, alpha_boundary, axin_xticks)
    fig.tight_layout()
    fig.savefig('tbd00.png', dpi=200)
    # fig.savefig('data/isotropic3_ree.png', dpi=200)
    # fig.savefig('data/isotropic3_ree.pdf')


def plot_werner3_isotropic3_ree():
    with open('data/werner3_ree.pkl', 'rb') as fid:
        data_figa = pickle.load(fid)
    with open('data/isotropic3_ree.pkl', 'rb') as fid:
        data_figb = pickle.load(fid)

    fig,(ax0,ax1) = plt.subplots(2,1,figsize=(6.4, 9.6))

    dim = 3
    alpha_boundary = 1/dim
    axin_xticks = [0.33, 0.37, 0.41, 0.45] if (dim==3) else [0.5, 0.54, 0.58]
    title = r'$3\otimes 3$ Werner state'
    _ax_plot(ax0, data_figa, alpha_boundary, axin_xticks, title)

    dim = 3
    alpha_boundary = 1/(dim+1)
    axin_xticks = [0.25, 0.3, 0.35]
    title = r'$3\otimes 3$ isotropic state'
    _ax_plot(ax1, data_figb, alpha_boundary, axin_xticks, title)
    fig.tight_layout()
    fig.savefig('tbd00.png', dpi=200)
    # fig.savefig('data/werner3_isotropic3_ree.png', dpi=200)
    # fig.savefig('data/werner3_isotropic3_ree.pdf')


def demo_purebQ_werner2_ree():
    dim = 2

    alpha_boundary = 1/dim
    alpha_list = np.linspace(0, 1, 50)
    alpha_fine_list = np.linspace(alpha_boundary*0.95, min(1,alpha_boundary*1.34), 50)

    model = pyqet.entangle.AutodiffCHAREE(dim, dim, distance_kind='ree')
    kwargs = dict(num_repeat=3, tol=1e-10, print_every_round=0)
    ree_cha = []
    ree_fine_cha = []
    for alpha_i in tqdm(alpha_list):
        model.set_dm_target(pyqet.entangle.get_werner_state(dim, alpha_i))
        ree_cha.append(pyqet.optimize.minimize(model, **kwargs).fun)
    for alpha_i in tqdm(alpha_fine_list):
        model.set_dm_target(pyqet.entangle.get_werner_state(dim, alpha_i))
        ree_fine_cha.append(pyqet.optimize.minimize(model, **kwargs).fun)
    ree_cha = np.array(ree_cha)
    ree_fine_cha = np.array(ree_fine_cha)

    kext_layer_list = [(4,5),(8,9),(12,13)]
    kwargs = dict(num_repeat=3, tol=1e-10, print_every_round=0)
    ree_purebQ = []
    ree_purebQ_fine = []
    for kext,num_layer in kext_layer_list:
        model = pyqet.entangle.QuantumPureBosonicExt(dim, dim, kext, num_layer)
        for alpha_i in tqdm(alpha_list):
            model.set_dm_target(pyqet.entangle.get_werner_state(dim, alpha_i))
            ree_purebQ.append(pyqet.optimize.minimize(model, **kwargs).fun)
        for alpha_i in tqdm(alpha_fine_list):
            model.set_dm_target(pyqet.entangle.get_werner_state(dim, alpha_i))
            ree_purebQ_fine.append(pyqet.optimize.minimize(model, **kwargs).fun)
    ree_purebQ = np.array(ree_purebQ).reshape(len(kext_layer_list), len(alpha_list))
    ree_purebQ_fine = np.array(ree_purebQ_fine).reshape(len(kext_layer_list), len(alpha_fine_list))

    ree_analytical = np.array([pyqet.entangle.get_werner_state_ree(dim, x) for x in alpha_list])
    ree_fine_analytical = np.array([pyqet.entangle.get_werner_state_ree(dim, x) for x in alpha_fine_list])

    tmp0 = np.stack([pyqet.entangle.get_werner_state(dim,x) for x in alpha_list])
    ree_ppt = pyqet.entangle.get_ppt_ree(tmp0, dim, dim, use_tqdm=True)
    tmp0 = np.stack([pyqet.entangle.get_werner_state(dim,x) for x in alpha_fine_list])
    ree_fine_ppt = pyqet.entangle.get_ppt_ree(tmp0, dim, dim, use_tqdm=True)

    # with open('data/purebQ_werner2_ree.pkl', 'wb') as fid:
    #     tmp0 = dict(alpha_list=alpha_list, kext_layer_list=kext_layer_list, ree_purebQ=ree_purebQ, ree_analytical=ree_analytical,
    #             ree_ppt=ree_ppt, ree_cha=ree_cha, alpha_fine_list=alpha_fine_list, ree_purebQ_fine=ree_purebQ_fine,
    #             ree_fine_analytical=ree_fine_analytical, ree_fine_ppt=ree_fine_ppt, ree_fine_cha=ree_fine_cha)
    #     pickle.dump(tmp0, fid)

    fig,ax = plt.subplots(figsize=(6.4, 4.8))
    for ind0 in range(len(kext_layer_list)):
        tmp0 = f'PureB({kext_layer_list[ind0][0]}), #layer={kext_layer_list[ind0][1]}'
        ax.plot(alpha_list, ree_purebQ[ind0], color=tableau[ind0+3], label=tmp0)
    ax.plot(alpha_list, ree_analytical, ':', color=tableau[0], markersize=3, label='analytical', markerfacecolor='none')
    ax.plot(alpha_list, ree_ppt, '+', color=tableau[1], label='PPT')
    ax.plot(alpha_list, ree_cha, 'x', color=tableau[2], label='CHA')
    ax.legend(ncol=2, fontsize=12, loc='upper left')
    ax.set_xlim(0, 1)
    ax.set_xlabel(r'$\alpha$', fontsize=12)
    ax.set_ylabel('relative entropy of entanglement', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=11)

    axin = ax.inset_axes([0.15, 0.24, 0.53, 0.47])
    for ind0 in range(len(kext_layer_list)):
        axin.plot(alpha_fine_list, ree_purebQ_fine[ind0], color=tableau[ind0+3])
    axin.plot(alpha_fine_list, ree_fine_analytical, ':', color=tableau[0], markersize=3, markerfacecolor='none')
    tmp0 = ree_fine_ppt.copy()
    tmp0[alpha_fine_list<alpha_boundary] = np.nan
    axin.plot(alpha_fine_list, tmp0, '+', color=tableau[1])
    axin.plot(alpha_fine_list, ree_fine_cha, 'x', color=tableau[2])
    axin.set_xlim(alpha_fine_list[0], alpha_fine_list[-1])
    # axin.set_ylim(1e-12, None)
    axin.set_yscale('log')
    axin.tick_params(axis='both', which='major', labelsize=11)
    axin.set_xticks([0.5, 0.55, 0.6, 0.65])
    hrect,hpatch = ax.indicate_inset_zoom(axin, edgecolor="red")
    hrect.set_xy((hrect.get_xy()[0], -0.02))
    hrect.set_height(0.05)
    fig.tight_layout()
    fig.savefig('tbd00.png', dpi=200)
    # fig.savefig('data/purebQ_werner2_ree.png', dpi=200)
    # fig.savefig('data/purebQ_werner2_ree.pdf')


def demo_cha_critical_point_N():
    # 2x2: 3/4,5
    # 2x3: 5/6,7
    # 3x3: 8/9,10
    dimA = 2
    dimB = 2
    num_state_list = [3,4,5]
    num_sample = 1000
    kwargs = dict(tol=1e-12, num_repeat=3, print_every_round=0)

    z0 = []
    for num_state in num_state_list:
        model = pyqet.entangle.AutodiffCHAREE(dimA, dimB, num_state, distance_kind='gellmann')
        for _ in tqdm(range(num_sample), desc=f'[k={num_state}]'):
            dm0 = pyqet.random.rand_separable_dm(dimA, dimB, k=2*(dimA*dimB)**2)
            model.set_dm_target(dm0)
            z0.append(pyqet.optimize.minimize(model, **kwargs).fun)
    z0 = np.array(z0).reshape(-1, num_sample)

    fig,ax = plt.subplots()
    ax.errorbar(num_state_list, z0.mean(axis=1), yerr=z0.std(axis=1), fmt='o')
    ax.fill_between(num_state_list, z0.min(axis=1), z0.max(axis=1), alpha=0.2)
    ax.set_xlabel('#cha')
    ax.set_ylabel('CHA loss(gellmann)')
    ax.set_yscale('log')
    ax.set_title(f'{dimA}x{dimB}, #sample={num_sample}')
    fig.tight_layout()
    fig.savefig('tbd00.png', dpi=200)



def _pureb_critical_point_k_00_hf0(dimA, dimB, kext, num_sample):
    model = pyqet.entangle.PureBosonicExt(dimA, dimB, kext, distance_kind='ree')
    num_fail = 0
    for _ in range(num_sample):
        dm0 = pyqet.random.rand_density_matrix(dimA*dimB)
        model.set_dm_target(dm0)
        try:
            # kstar is tol-dependent
            _ = pyqet.optimize.minimize(model, theta0='uniform', tol=1e-12, num_repeat=1, print_every_round=0)
        except Exception:
            num_fail += 1
    return num_fail


def demo_pureb_critical_point_k_00():
    num_sample = 100

    dimA_dimB_to_kstar = {(x,2):(2*x) for x in range(2,8)}
    tmp0 = {2:3, 3:4, 4:5, 5:5, 6:6, 7:6, 8:7} #dimA>=9, the numerical stability is not a issue maybe from pade approximation
    dimA_dimB_to_kstar.update({(x,3):y for x,y in tmp0.items()})

    dimA = 3
    dimB = 4
    kstar = dimA_dimB_to_kstar[(dimA,dimB)]
    kext_list = [kstar-1,kstar]
    num_fail_list = []
    for kext in kext_list:
        t0 = time.time()
        num_fail = _pureb_critical_point_k_00_hf0(dimA, dimB, kext, num_sample)
        tmp0 = time.time() - t0
        print(f'[{tmp0:.1f}s] ({dimA}x{dimB},k={kext}), num_fail={num_fail}/{num_sample}')
        num_fail_list.append(num_fail)


def demo_pure_critical_point_k_01():
    seed = np_rng.integers(0, 2**32-1, size=1)
    seed = 3034543436 #for reproducible
    np_rng_i = np.random.default_rng(seed)
    num_point = 200
    kwargs = dict(theta0='uniform', tol=1e-12, num_repeat=3, print_every_round=0)

    tmp0 = [(2,2,(3,4,5)), (3,2,(5,6,7)), (2,3,(2,3,4)), (3,3,(3,4,5))]
    para_list = [dict(dimA=x,dimB=y,kext_list=z) for x,y,z in tmp0]

    for para_i in para_list:
        dimA = para_i['dimA']
        dimB = para_i['dimB']
        kext_list = para_i['kext_list']

        model = pyqet.entangle.PureBosonicExt(dimA, dimB, kext=kext_list[-1], distance_kind='gellmann')
        while True:
            tmp0 = pyqet.random.rand_density_matrix(dimA*dimB, seed=np_rng_i)
            beta_u = pyqet.entangle.get_density_matrix_boundary(tmp0)[1]
            dm_target = pyqet.entangle.hf_interpolate_dm(tmp0, beta=beta_u)
            beta_list = np.linspace(0, beta_u, num_point)
            dm_target_list = [pyqet.entangle.hf_interpolate_dm(dm_target,beta=x) for x in beta_list]
            model.set_dm_target(dm_target)
            # find a direction where PureB boundary is not density matrix boundary
            if pyqet.optimize.minimize(model, **kwargs).fun>1e-3:
                break

        z0 = []
        for kext in kext_list:
            model = pyqet.entangle.PureBosonicExt(dimA, dimB, kext=kext, distance_kind='gellmann')
            for dm_target_i in tqdm(dm_target_list):
                model.set_dm_target(dm_target_i)
                z0.append(pyqet.optimize.minimize(model, **kwargs).fun)
        para_i['dm_target'] = dm_target
        para_i['beta_list'] = beta_list
        para_i['distance'] = np.array(z0).reshape(len(kext_list), len(beta_list))

        tmp0 = dict(rho=dm_target, dim=(dimA,dimB), xtol=1e-5, use_boson=True, use_tqdm=True)
        kext_boundary_sdp = [pyqet.entangle.get_ABk_symmetric_extension_boundary(kext=x, **tmp0) for x in kext_list]
        para_i['kext_boundary_sdp'] = kext_boundary_sdp

    fig,tmp0 = plt.subplots(2,2,figsize=(8,6))
    ax_list = [tmp0[0,0], tmp0[0,1], tmp0[1,0], tmp0[1,1]]
    for para_i,ax in zip(para_list,ax_list):
        dimA = para_i['dimA']
        dimB = para_i['dimB']
        kext_list = para_i['kext_list']
        beta_list = para_i['beta_list']
        kext_boundary_sdp = para_i['kext_boundary_sdp']
        z0 = para_i['distance']
        for ind0 in range(len(kext_list)):
            ax.plot(beta_list, z0[ind0], label=f"PureB({kext_list[ind0]})", color=tableau[ind0])
            ax.axvline(kext_boundary_sdp[ind0], color=tableau[ind0], linestyle='--')
        ax.set_xlim(min(beta_list), max(beta_list))
        ax.set_title(f'$d_A={dimA}, d_B={dimB}$')
        # ax.set_ylim(1e-13, 1)
        ax.set_yscale('log')
        ax.set_xlabel(r"$||\vec{\rho}||_2$")
        ax.set_ylabel(r'$D(\rho,\bar{\Theta}_k)^2$')
        ax.legend(loc='center left')
    tmp0 = dict(horizontalalignment='center', verticalalignment='center')#, fontsize=16
    ax_list[0].text(0.03, 5e-16, '(a)', **tmp0)
    ax_list[1].text(0.02, 5e-14, '(b)', **tmp0)
    ax_list[2].text(0.02, 1e-14, '(c)', **tmp0)
    ax_list[3].text(0.02, 2e-13, '(d)', **tmp0)
    for ax in ax_list[1::2]:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position('right')
    fig.tight_layout()
    fig.savefig('tbd00.png', dpi=200)

    # fig.savefig('data/pure_critical_point_k_01.png', dpi=200)
    # fig.savefig('data/pure_critical_point_k_01.pdf')
    # with open('data/pure_critical_point_k_01.pkl', 'wb') as fid:
    #     pickle.dump(dict(para_list=para_list), fid)



def demo_upb_bes_boundary():
    dm_tiles = pyqet.entangle.load_upb('tiles', return_bes=True)[1]
    dm_pyramid = pyqet.entangle.load_upb('pyramid', return_bes=True)[1]

    # CHA(18min) PureB8(6min) PureB32(12min)
    fig,ax,all_data = pyqet.entangle.plot_dm0_dm1_plane(dm0=dm_tiles, dm1=dm_pyramid, dimA=3, dimB=3, num_eig0=0,
                num_point=201, pureb_kext=[8,32], tag_cha=True, label0='Tiles', label1='Pyramid')
    ax.legend(fontsize='large', ncols=2, loc='lower right')
    fig.tight_layout()
    # fig.savefig('data/tiles_pyramid_boundary.png', dpi=200)
    # fig.savefig('data/tiles_pyramid_boundary.pdf')
    # with open('data/tiles_pyramid_boundary.pkl', 'wb') as fid:
    #     pickle.dump(all_data, fid)



def demo_kext_boundary_accuracy():
    dimA = 2
    dimB = 2
    kext_list = list(range(4,12))[::-1] #[4,5,6,7,8,9,10,11]
    dm_target_list = np.stack([pyqet.random.rand_density_matrix(dimA*dimB) for _ in range(100)])

    beta_sdp = []
    for kext in kext_list:
        for rho in tqdm(dm_target_list, desc=f'SDP(k={kext})'):
            beta_sdp.append(pyqet.entangle.get_ABk_symmetric_extension_boundary(rho, (dimA,dimB), kext, xtol=1e-5, use_boson=True, use_tqdm=False))
    beta_sdp = np.array(beta_sdp, dtype=np.float64).reshape(len(kext_list), -1)

    beta_pureb = []
    for kext in kext_list:
        model = pyqet.entangle.PureBosonicExt(dimA, dimB, kext=kext)
        for rho in tqdm(dm_target_list, desc=f'PureB(k={kext})'):
            beta_pureb.append(model.get_boundary(rho, xtol=1e-5, converge_tol=1e-10, threshold=1e-7, num_repeat=3, use_tqdm=False))
    beta_pureb = np.array(beta_pureb, dtype=np.float64).reshape(len(kext_list), -1)

    ydata = np.abs(beta_pureb - beta_sdp)/beta_sdp
    ydata_mean = ydata.mean(axis=1)
    ydata_max = ydata.max(axis=1)
    fig,ax = plt.subplots()
    ax.plot(kext_list, ydata_mean, '-x', label=f'average')
    ax.plot(kext_list, ydata_max, '--x', label=f'maximum')
    # ax.fill_between(kext_list, ydata.min(axis=1), ydata.max(axis=1), alpha=0.3)
    ax.set_xlabel('k-ext', fontsize=12)
    ax.set_ylabel(r'relative error of $\beta$', fontsize=12)
    ax.legend(fontsize=12)
    ax.set_ylim(5e-5, 0.5)
    ax.set_yscale('log')
    ax.tick_params(axis='both', which='major', labelsize=11)
    fig.tight_layout()
    fig.savefig('tbd00.png', dpi=200)

    # fig.savefig('data/kext_boundary_accuracy.png', dpi=200)
    # fig.savefig('data/kext_boundary_accuracy.pdf')
    # with open('data/kext_boundary_accuracy.pkl', 'wb') as fid:
    #     tmp0 = dict(beta_sdp=beta_sdp, beta_pureb=beta_pureb, dimA=dimA, dimB=dimB,
    #                 dm_target_list=dm_target_list, kext_list=kext_list)
    #     pickle.dump(tmp0, fid)
