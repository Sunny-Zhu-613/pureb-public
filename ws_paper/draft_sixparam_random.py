import os
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
tableau = ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1', '#C85200', '#898989', '#A2C8EC', '#FFBC79', '#CFCFCF']

plt.ion()

import pyqet
hf_data = lambda *x: os.path.join('..', 'data', *x)

np_rng = np.random.default_rng()

pyqet.cha._set_num_cpu(12)

def plot_sixparam_random():
    datapath = hf_data('sixparam_random.pkl')
    dimA = 3
    dimB = 3
    num_cha_repeat = 3
    num_pureb_repeat = 3
    pureb_kext_list = [8,16,32]
    num_sample = 100
    if os.path.exists(datapath):
        with open(datapath, 'rb') as fid:
            tmp0 = pickle.load(fid)
            param_list = tmp0['param_list']
            beta_ppt = tmp0['beta_ppt']
            beta_bcha = tmp0['beta_bcha']
            beta_pureb = tmp0['beta_pureb']
    else:
        param_list = np_rng.uniform(0, 2*np.pi, size=(num_sample,6))
        dm_target_list = [pyqet.upb.load_upb('sixparam', x, return_bes=True)[1] for x in param_list]

        alpha_ppt,beta_ppt,alpha_bcha,beta_bcha = pyqet.cha.get_ppt_bcha_boundary(dm_target_list, dimA, dimB, num_repeat=num_cha_repeat)

        beta_pureb = np.zeros((len(pureb_kext_list),len(dm_target_list)), dtype=np.float64)
        for ind0 in range(len(pureb_kext_list)):
            model = pyqet.pureb.PureBosonicExt(dimA, dimB, kext=pureb_kext_list[ind0])
            for ind1,dm_target in enumerate(tqdm(dm_target_list, desc=f'PureB')):
                beta_pureb[ind0,ind1] = model.solve_boundary(dm_target, alpha_bcha=alpha_bcha[ind1],
                        xtol=1e-4, threshold=1e-7, use_tqdm=False, num_repeat=num_pureb_repeat)[1]
        with open(datapath, 'wb') as fid:
            tmp0 = dict(param_list=param_list, beta_ppt=beta_ppt, beta_bcha=beta_bcha, beta_pureb=beta_pureb,
                    num_cha_repeat=num_cha_repeat, num_pureb_repeat=num_pureb_repeat, pureb_kext_list=pureb_kext_list)
            pickle.dump(tmp0, fid)

    FONTSIZE = 16
    fig,ax = plt.subplots()
    ind0 = np.argsort(beta_bcha)
    ax.plot(beta_bcha[ind0], label='CHA')
    ax.plot(beta_ppt[ind0], label='PPT')
    for ind1 in range(1, len(pureb_kext_list)):
        ax.plot(beta_pureb[ind1,ind0], label=f'PureB({pureb_kext_list[ind1]})')
    ax.legend(fontsize=FONTSIZE)
    ax.grid()
    ax.set_xlim(0, len(ind0))
    ax.set_ylabel(r'$||\vec{\sigma}||_2$', fontsize=FONTSIZE)
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE-2)
    fig.tight_layout()
    fig.savefig(hf_data('sixparam_random.png'), dpi=200)
    fig.savefig(hf_data('sixparam_random.pdf'))

if __name__=='__main__':
    plt.close('all')
    plot_sixparam_random()
