import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm



from ._misc import hf_intertropalte_dm, density_matrix_plane, density_matrix_boundary
from . import ppt
from . import cha
from .pureb import PureBosonicExt


def plot_dm0_dm1_plane(dm0, dm1, dimA, dimB, num_point=201, pureb_kext=None, tag_cha=False,
        num_eig0=None, label0=None, label1=None, num_cha_repeat=1):
    if label0 is None:
        label0 = r'$\rho_a$'
    if label1 is None:
        label1 = r'$\rho_b$'
    theta0,norm0,theta1,norm1,hf_theta = density_matrix_plane(dm0, dm1)
    if num_eig0 is None:
        tmp0 = np.linalg.eigvalsh(hf_intertropalte_dm(hf_theta(theta1), beta=density_matrix_boundary(dm1)[0]))
        num_eig0 = (tmp0<1e-7).sum()

    theta_list = np.linspace(-np.pi, np.pi, num_point)
    beta_ppt = np.zeros_like(theta_list)
    beta_dm = np.zeros_like(theta_list)
    eig_dm = np.zeros((dimA*dimB, len(theta_list)), dtype=np.float64)
    for ind0,x in enumerate(tqdm(theta_list)):
        dm_target = hf_theta(x)
        _,beta_ppt[ind0] = ppt.get_boundary(dm_target, dimA, dimB, within_dm=True)
        beta_dm[ind0] = density_matrix_boundary(dm_target)[0]
        eig_dm[:,ind0] = np.linalg.eigvalsh(hf_intertropalte_dm(dm_target, beta=beta_dm[ind0]))

    if tag_cha:
        dm_target_list = [hf_theta(x) for x in theta_list]
        beta_cha = cha.get_cha_boundary(dm_target_list, dimA, num_repeat=num_cha_repeat)[1]
    else:
        beta_cha = None
    if pureb_kext is not None:
        if not hasattr(pureb_kext, '__len__'):
            pureb_kext = list(pureb_kext)
        beta_pureb = np.zeros((len(pureb_kext), len(theta_list)), dtype=np.float64)
        for ind0 in range(len(pureb_kext)):
            model = PureBosonicExt(dimA, dimB, kext=pureb_kext[ind0])
            for ind1,x in enumerate(tqdm(theta_list, desc=f'PureB-{pureb_kext[ind0]}')):
                beta_pureb[ind0,ind1] = model.solve_boundary(hf_theta(x), alpha_cha=0, xtol=1e-4, threshold=1e-7, num_repeat=1, use_tqdm=False)[1]
    else:
        beta_pureb = None

    fig,ax = plt.subplots()
    hf0 = lambda theta,r: (r*np.cos(theta), r*np.sin(theta))
    for theta, label in [(theta0,label0),(theta1,label1)]:
        radius = 0.3
        ax.plot([0, radius*np.cos(theta)], [0, radius*np.sin(theta)], linestyle=':', label=label)
    ax.plot(*hf0(theta_list, beta_dm), label='DM')
    ax.plot(*hf0(theta_list, beta_ppt), linestyle='--', label='PPT')
    if beta_pureb is not None:
        for ind0 in range(len(pureb_kext)):
            ax.plot(*hf0(theta_list, beta_pureb[ind0]), label=f'PureB({pureb_kext[ind0]})')
    if beta_cha is not None:
        ax.plot(*hf0(theta_list, beta_cha), label='CHA')
    for ind0 in range(1, num_eig0):
        ax.plot(*hf0(theta_list, eig_dm[ind0]), label=rf'$\lambda_{ind0+1}$ dm')
    ax.legend(fontsize='small')
    # ax.legend(fontsize=11, ncol=2, loc='lower right')
    # ax.tick_params(axis='both', which='major', labelsize=11)
    fig.tight_layout()
    fig.savefig('tbd00.png', dpi=200)
    all_data = dict(dm0=dm0, dm1=dm1, theta_list=theta_list, eig_dm=eig_dm, beta_dm=beta_dm,
            beta_ppt=beta_ppt, beta_cha=beta_cha, pureb_kext=pureb_kext, beta_pureb=beta_pureb, num_eig0=num_eig0)
    return fig,ax,all_data


