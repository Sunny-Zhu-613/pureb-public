import os
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.ion()

import pyqet

hf_data = lambda *x: os.path.join('..', 'data', *x)
tableau = ['#4c72b0', '#dd8452', '#55a868', '#c44e52', '#8172b3', '#937860', '#da8bc3', '#8c8c8c', '#ccb974', '#64b5cd']

def demo_pure_critical_point_k_01():
    datapath = hf_data('pure_critical_point_k_01.pkl')
    if os.path.exists(datapath):
        with open(datapath, 'rb') as fid:
            para_list = pickle.load(fid)['para_list']
    else:
        # seed = np.random.default_rng().integers(0, 2**32-1, size=1)
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
        with open(datapath, 'wb') as fid:
            pickle.dump(dict(para_list=para_list), fid)


    FONTSIZE = 16
    fig,tmp0 = plt.subplots(2,2,figsize=(8,6))
    ax_list = [tmp0[0,0], tmp0[0,1], tmp0[1,0], tmp0[1,1]]
    for para_i,ax in zip(para_list,ax_list):
        dimA = para_i['dimA']
        dimB = para_i['dimB']
        kext_list = para_i['kext_list']
        beta_list = para_i['beta_list']
        kext_boundary_sdp = para_i['kext_boundary_sdp']
        z0 = para_i['distance']
        color_list = [tableau[x] for x in [0,1,7]]
        for ind0 in range(len(kext_list)):
            ax.plot(beta_list, z0[ind0], label=f"PureB({kext_list[ind0]})", color=color_list[ind0])
            ax.axvline(kext_boundary_sdp[ind0], color=color_list[ind0], linestyle='--')
        ax.set_xlim(min(beta_list), max(beta_list))
        ax.set_title(f'$d_A={dimA}, d_B={dimB}, k^*={kext_list[-2]}$', fontsize=FONTSIZE)
        # ax.set_ylim(1e-13, 1)
        ax.set_yscale('log')
        ax.set_xlabel(r"$||\vec{\rho}||_2$", fontsize=FONTSIZE)
        ax.set_ylabel(r'$D(\rho,\mathcal{S})^2$', fontsize=FONTSIZE)
        ax.legend(loc='upper left', fontsize=FONTSIZE-2)
        ax.tick_params(axis='both', which='major', labelsize=FONTSIZE-3)
    for ax in ax_list[:2]:
        ax.set_xlabel('')
    tmp0 = dict(horizontalalignment='center', verticalalignment='center')#, fontsize=16
    ax_list[0].text(0.03, 5e-16, '(a)', **tmp0, fontsize=FONTSIZE)
    ax_list[1].text(0.02, 5e-14, '(b)', **tmp0, fontsize=FONTSIZE)
    ax_list[2].text(0.02, 1e-14, '(c)', **tmp0, fontsize=FONTSIZE)
    ax_list[3].text(0.02, 1e-13, '(d)', **tmp0, fontsize=FONTSIZE)
    for ax in ax_list[1::2]:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position('right')
    fig.tight_layout()
    fig.savefig(hf_data('pure_critical_point_k_01.png'), dpi=200)
    fig.savefig(hf_data('pure_critical_point_k_01.pdf'))


def plot_werner3_isotropic3_ree():
    datapath_werner = hf_data('werner3_ree.pkl')
    datapath_isotropic = hf_data('isotropic3_ree.pkl')
    with open(datapath_werner, 'rb') as fid:
        data_figa = pickle.load(fid)
    with open(datapath_isotropic, 'rb') as fid:
        data_figb = pickle.load(fid)

    FONTSIZE = 16
    fig,(ax0,ax1) = plt.subplots(2,1,figsize=(6.4, 9.6))

    alpha_list = data_figa['alpha_list']
    kext_list = data_figa['kext_list']
    ree_pureb = data_figa['ree_pureb']
    ree_analytical = data_figa['ree_analytical']
    ree_ppt = data_figa['ree_ppt']
    ree_cha = data_figa['ree_cha']
    for ind0 in range(len(kext_list)):
        ax0.plot(alpha_list, ree_pureb[ind0], color=tableau[ind0+3], label=f'PureB({kext_list[ind0]})')
    ax0.plot(alpha_list, ree_analytical, ':', color=tableau[0], markersize=3, label='analytical', markerfacecolor='none')
    ax0.plot(alpha_list, ree_ppt, '+', color=tableau[1], label='PPT')
    ax0.plot(alpha_list, ree_cha, 'x', color=tableau[7], label='CHA')
    ax0.legend(ncol=3, fontsize=FONTSIZE-2, loc='upper left')
    ax0.set_xlim(0, 1)
    ax0.set_xlabel(r'$\alpha$', fontsize=FONTSIZE)
    ax0.set_ylabel('REE', fontsize=FONTSIZE)
    ax0.set_title(r'$3\otimes 3$ Werner state', fontsize=FONTSIZE)
    ax0.tick_params(axis='both', which='major', labelsize=FONTSIZE-2)

    dim = 3
    alpha_boundary = 1/(dim)
    alpha_fine_list = data_figa['alpha_fine_list']
    ree_pureb_fine = data_figa['ree_pureb_fine']
    ree_fine_ppt = data_figa['ree_fine_ppt']
    ree_fine_analytical = data_figa['ree_fine_analytical']
    ree_fine_cha = data_figa['ree_fine_cha']
    axin = ax0.inset_axes([0.15, 0.24, 0.47, 0.47])
    for ind0 in range(len(kext_list)):
        axin.plot(alpha_fine_list, ree_pureb_fine[ind0], color=tableau[ind0+3])
    axin.plot(alpha_fine_list, ree_fine_analytical, ':', color=tableau[0], markersize=3, markerfacecolor='none')
    tmp0 = ree_fine_ppt.copy()
    tmp0[alpha_fine_list<alpha_boundary] = np.nan
    axin.plot(alpha_fine_list, tmp0, '+', color=tableau[1])
    axin.plot(alpha_fine_list, ree_fine_cha, 'x', color=tableau[7])
    axin.set_xlim(alpha_fine_list[0], alpha_fine_list[-1])
    axin.set_yscale('log')
    axin.tick_params(axis='both', which='major', labelsize=FONTSIZE-3)
    if dim==3:
        axin.set_xticks([0.33, 0.37, 0.41, 0.45])
    elif dim==2:
        axin.set_xticks([0.5, 0.54, 0.58])
    hrect,hpatch = ax0.indicate_inset_zoom(axin, edgecolor="red")
    hrect.set_xy((hrect.get_xy()[0], -0.02))
    hrect.set_height(0.05)



    alpha_list = data_figb['alpha_list']
    kext_list = data_figb['kext_list']
    ree_pureb = data_figb['ree_pureb']
    ree_analytical = data_figb['ree_analytical']
    ree_ppt = data_figb['ree_ppt']
    ree_cha = data_figb['ree_cha']
    for ind0 in range(len(kext_list)):
        ax1.plot(alpha_list, ree_pureb[ind0], color=tableau[ind0+3], label=f'PureB({kext_list[ind0]})')
    ax1.plot(alpha_list, ree_analytical, ':', color=tableau[0], markersize=3, label='analytical', markerfacecolor='none')
    ax1.plot(alpha_list, ree_ppt, '+', color=tableau[1], label='PPT')
    ax1.plot(alpha_list, ree_cha, 'x', color=tableau[7], label='CHA')
    ax1.legend(ncol=3, fontsize=FONTSIZE-2, loc='upper left')
    ax1.set_xlim(0, 1)
    ax1.set_xlabel(r'$\alpha$', fontsize=FONTSIZE)
    ax1.set_ylabel('REE', fontsize=FONTSIZE)
    ax1.set_title(r'$3\otimes 3$ Isotropic state', fontsize=FONTSIZE)
    ax1.tick_params(axis='both', which='major', labelsize=FONTSIZE-2)

    dim = 3
    alpha_boundary = 1/(dim+1)
    alpha_fine_list = data_figb['alpha_fine_list']
    ree_pureb_fine = data_figb['ree_pureb_fine']
    ree_fine_ppt = data_figb['ree_fine_ppt']
    ree_fine_analytical = data_figb['ree_fine_analytical']
    ree_fine_cha = data_figb['ree_fine_cha']
    axin = ax1.inset_axes([0.1, 0.24, 0.47, 0.47])
    for ind0 in range(len(kext_list)):
        axin.plot(alpha_fine_list, ree_pureb_fine[ind0], color=tableau[ind0+3])
    axin.plot(alpha_fine_list, ree_fine_analytical, ':', color=tableau[0], markersize=3, markerfacecolor='none')
    tmp0 = ree_fine_ppt.copy()
    tmp0[alpha_fine_list<alpha_boundary] = np.nan
    axin.plot(alpha_fine_list, tmp0, '+', color=tableau[1])
    axin.plot(alpha_fine_list, ree_fine_cha, 'x', color=tableau[7])
    axin.set_xlim(alpha_fine_list[0], alpha_fine_list[-1])
    axin.set_yscale('log')
    axin.tick_params(axis='both', which='major', labelsize=FONTSIZE-3)
    axin.set_xticks([0.25, 0.3, 0.35])
    hrect,hpatch = ax1.indicate_inset_zoom(axin, edgecolor="red")
    hrect.set_xy((hrect.get_xy()[0], -0.02))
    hrect.set_height(0.05)

    fig.tight_layout()
    fig.savefig(hf_data('werner3_isotropic3_ree.png'), dpi=200)
    fig.savefig(hf_data('werner3_isotropic3_ree.pdf'))


def demo_upb_bes_boundary():
    datapath = hf_data('tiles_pyramid_boundary.pkl')
    num_point = 201
    pureb_kext = [8,32]
    dm_tiles = pyqet.upb.load_upb('tiles', return_bes=True)[1]
    dm_pyramid = pyqet.upb.load_upb('pyramid', return_bes=True)[1]
    if os.path.exists(datapath):
        with open(datapath, 'rb') as fid:
            all_data = pickle.load(fid)
    else:
        # CHA(18min) PureB8(6min) PureB32(12min)
        fig,ax,all_data = pyqet.bes.plot_dm0_dm1_plane(dm0=dm_tiles, dm1=dm_pyramid, dimA=3, dimB=3, num_eig0=0,
                    num_point=num_point, pureb_kext=pureb_kext, tag_cha=True, label0='Tiles', label1='Pyramid')
        plt.close(fig)
        with open(datapath, 'wb') as fid:
            pickle.dump(all_data, fid)

    FONTSIZE = 16
    label0 = 'Tiles'
    label1 = 'Pyramid'
    beta_dm = all_data['beta_dm']
    beta_ppt = all_data['beta_ppt']
    beta_pureb = all_data['beta_pureb']
    beta_cha = all_data['beta_cha']
    theta0,norm0,theta1,norm1,hf_theta = pyqet.density_matrix_plane(dm_tiles, dm_pyramid)
    theta_list = np.linspace(-np.pi, np.pi, num_point)
    fig,ax = plt.subplots()
    hf0 = lambda theta,r: (r*np.cos(theta), r*np.sin(theta))
    for theta, label in [(theta0,label0),(theta1,label1)]:
        radius = 0.3
        ax.plot([0, radius*np.cos(theta)], [0, radius*np.sin(theta)], linestyle=':', label=label)
    ax.plot(*hf0(theta_list, beta_dm), label='DM')
    ax.plot(*hf0(theta_list, beta_ppt), linestyle='--', label='PPT')
    for ind0 in range(len(pureb_kext)):
        ax.plot(*hf0(theta_list, beta_pureb[ind0]), label=f'PureB({pureb_kext[ind0]})')
    ax.plot(*hf0(theta_list, beta_cha), label='CHA')
    ax.legend(fontsize='small')
    # ax.legend(fontsize=11, ncol=2, loc='lower right')
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE-4)
    ax.legend(fontsize=FONTSIZE-3, ncols=2, loc='lower right')
    fig.tight_layout()
    fig.savefig(hf_data('tiles_pyramid_boundary.png'), dpi=200)
    fig.savefig(hf_data('tiles_pyramid_boundary.pdf'))


def demo_kext_boundary_accuracy():
    datapath = hf_data('kext_boundary_accuracy.pkl')
    dimA = 2
    dimB = 2
    kext_list = list(range(4,12))[::-1] #[4,5,6,7,8,9,10,11]
    if os.path.exists(datapath):
        with open(datapath, 'rb') as fid:
            tmp0 = pickle.load(fid)
            beta_sdp = tmp0['beta_sdp']
            beta_pureb = tmp0['beta_pureb']
            dm_target_list = tmp0['dm_target_list']
    else:
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
        with open(datapath, 'wb') as fid:
            tmp0 = dict(beta_sdp=beta_sdp, beta_pureb=beta_pureb, dimA=dimA, dimB=dimB,
                        dm_target_list=dm_target_list, kext_list=kext_list)
            pickle.dump(tmp0, fid)

    FONTSIZE = 16
    ydata = np.abs(beta_pureb - beta_sdp)/beta_sdp
    ydata_mean = ydata.mean(axis=1)
    ydata_max = ydata.max(axis=1)
    fig,ax = plt.subplots()
    ax.plot(kext_list, ydata_mean, '-x', label=f'average of 100 samples')
    ax.plot(kext_list, ydata_max, '--x', label=f'maximum of 100 samples')
    # ax.fill_between(kext_list, ydata.min(axis=1), ydata.max(axis=1), alpha=0.3)
    ax.set_xlabel('extension number $k$', fontsize=FONTSIZE-1)
    ax.set_ylabel(r'relative error of boundary $||\vec{\sigma}||_2$', fontsize=FONTSIZE-1)
    ax.legend(fontsize=FONTSIZE-1)
    ax.set_yscale('log')
    ax.set_ylim(5e-5, None)
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE-3)
    fig.tight_layout()
    fig.savefig(hf_data('kext_boundary_accuracy.png'), dpi=200)
    fig.savefig(hf_data('kext_boundary_accuracy.pdf'))
    # with open('data/kext_boundary_accuracy.pkl', 'wb') as fid:
    #     tmp0 = dict(beta_sdp=beta_sdp, beta_pureb=beta_pureb, dimA=dimA, dimB=dimB,
    #                 dm_target_list=dm_target_list, kext_list=kext_list)
    #     pickle.dump(tmp0, fid)


def demo_purebQ_werner2_ree():
    datapath = hf_data('purebQ_werner2_ree.pkl')
    dim = 2
    alpha_boundary = 1/dim
    alpha_list = np.linspace(0, 1, 50)
    alpha_fine_list = np.linspace(alpha_boundary*0.95, min(1,alpha_boundary*1.34), 50)
    kext_layer_list = [(4,5),(8,9),(12,13)]
    if os.path.exists(datapath):
        with open(datapath, 'rb') as fid:
            tmp0 = pickle.load(fid)
            ree_purebQ = tmp0['ree_purebQ']
            ree_analytical = tmp0['ree_analytical']
            ree_fine_analytical = tmp0['ree_fine_analytical']
            ree_ppt = tmp0['ree_ppt']
            ree_cha = tmp0['ree_cha']
            alpha_fine_list = tmp0['alpha_fine_list']
            ree_purebQ_fine = tmp0['ree_purebQ_fine']
            ree_fine_ppt = tmp0['ree_fine_ppt']
            ree_fine_cha = tmp0['ree_fine_cha']
    else:
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

        with open(datapath, 'wb') as fid:
            tmp0 = dict(alpha_list=alpha_list, kext_layer_list=kext_layer_list, ree_purebQ=ree_purebQ, ree_analytical=ree_analytical,
                    ree_ppt=ree_ppt, ree_cha=ree_cha, alpha_fine_list=alpha_fine_list, ree_purebQ_fine=ree_purebQ_fine,
                    ree_fine_analytical=ree_fine_analytical, ree_fine_ppt=ree_fine_ppt, ree_fine_cha=ree_fine_cha)
            pickle.dump(tmp0, fid)

    FONTSIZE = 16
    fig,ax = plt.subplots(figsize=(6.4, 4.8))
    for ind0 in range(len(kext_layer_list)):
        tmp0 = f'PureB({kext_layer_list[ind0][0]}), #layer={kext_layer_list[ind0][1]}'
        ax.plot(alpha_list, ree_purebQ[ind0], color=tableau[ind0+3], label=tmp0)
    ax.plot(alpha_list, ree_analytical, ':', color=tableau[0], markersize=3, label='analytical', markerfacecolor='none')
    ax.plot(alpha_list, ree_ppt, '+', color=tableau[1], label='PPT')
    ax.plot(alpha_list, ree_cha, 'x', color=tableau[7], label='CHA')
    ax.legend(ncol=2, fontsize=FONTSIZE-2, loc='upper left')
    ax.set_xlim(0, 1)
    ax.set_xlabel(r'$\alpha$', fontsize=FONTSIZE)
    ax.set_ylabel('REE', fontsize=FONTSIZE)
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE-2)

    axin = ax.inset_axes([0.15, 0.24, 0.53, 0.47])
    for ind0 in range(len(kext_layer_list)):
        axin.plot(alpha_fine_list, ree_purebQ_fine[ind0], color=tableau[ind0+3])
    axin.plot(alpha_fine_list, ree_fine_analytical, ':', color=tableau[0], markersize=3, markerfacecolor='none')
    tmp0 = ree_fine_ppt.copy()
    tmp0[alpha_fine_list<alpha_boundary] = np.nan
    axin.plot(alpha_fine_list, tmp0, '+', color=tableau[1])
    axin.plot(alpha_fine_list, ree_fine_cha, 'x', color=tableau[7])
    axin.set_xlim(alpha_fine_list[0], alpha_fine_list[-1])
    # axin.set_ylim(1e-12, None)
    axin.set_yscale('log')
    axin.tick_params(axis='both', which='major', labelsize=FONTSIZE-3)
    axin.set_xticks([0.5, 0.55, 0.6, 0.65])
    hrect,hpatch = ax.indicate_inset_zoom(axin, edgecolor="red")
    hrect.set_xy((hrect.get_xy()[0], -0.02))
    hrect.set_height(0.05)
    fig.tight_layout()
    fig.savefig(hf_data('purebQ_werner2_ree.png'), dpi=200)
    fig.savefig(hf_data('purebQ_werner2_ree.pdf'))



if __name__=='__main__':
    plt.close('all')
    # demo_pure_critical_point_k_01()
    plot_werner3_isotropic3_ree()
    # demo_upb_bes_boundary()
    # demo_kext_boundary_accuracy()
    # demo_purebQ_werner2_ree()
