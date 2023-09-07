import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import pyqet

dimA = 2
dimB = 2
kext_list = list(range(4,12))[::-1] #[4,5,6,7,8,9,10,11]
num_sample = 100

dm_target_list = np.stack([pyqet.random.rand_density_matrix(dimA*dimB) for _ in range(num_sample)])

# please run the following code separately 
alpha_qetlab = []
for kext in kext_list:
    print(kext)
    alpha_qetlab_i = pyqet.qetlab.qetlab_kext_boundary(dm_target_list, dimA, dimB, kext, tol=1e-5, use_BOS=True)
    alpha_qetlab.append(alpha_qetlab_i)
    with open(f'qetlab_kext{kext}.pkl', 'wb') as fid:
        tmp0 = dict(dm_target_list=dm_target_list, alpha_qetlab_i=alpha_qetlab_i, kext=kext)
        pickle.dump(tmp0, fid)
alpha_qetlab = np.stack(alpha_qetlab)

alpha_cha,beta_cha = pyqet.cha.get_cha_boundary(dm_target_list, dimA, num_repeat=1)

alpha_pureb = np.zeros((len(kext_list),num_sample), dtype=np.float64)
for ind0,kext in enumerate(kext_list):
    model = pyqet.pureb.PureBosonicExt(dimA, dimB, kext)
    for ind1 in tqdm(range(num_sample), desc=f'PureB({kext})'):
        alpha_pureb[ind0,ind1] = model.solve_boundary(dm_target_list[ind1], alpha_cha[ind1], num_repeat=3, xtol=1e-5, use_tqdm=False)[0]

all_data = dict(alpha_pureb=alpha_pureb, alpha_qetlab=alpha_qetlab, alpha_cha=alpha_cha, dm_target_list=dm_target_list)
# with open('data/kext_boundary_accuracy.pkl', 'wb') as fid:
#     pickle.dump(all_data, fid)

# alpha is different from beta by a factor which will be cancelled when calculating the relative error
ydata = np.abs(alpha_pureb - alpha_qetlab)/alpha_qetlab
ydata_mean = ydata.mean(axis=1)
ydata_max = ydata.max(axis=1)

fig,ax = plt.subplots()
ax.plot(kext_list, ydata_mean, '-x', label=f'average of {num_sample} samples')
ax.plot(kext_list, ydata_max, '--x', label=f'maximum of {num_sample} samples')
# ax.fill_between(kext_list, ydata.min(axis=1), ydata.max(axis=1), alpha=0.3)
ax.set_xlabel('k-ext', fontsize=12)
ax.set_ylabel(r'relative error of boundary $\beta$', fontsize=12)
ax.legend(fontsize=12)
# ax.set_title(rf'${dimA}\otimes {dimB}$, #sample={num_sample}, [min,mean,max]')
ax.set_ylim(5e-5, 0.5)
ax.set_yscale('log')
ax.tick_params(axis='both', which='major', labelsize=11)
fig.tight_layout()
fig.savefig('tbd00.png', dpi=200)
# fig.savefig('../data/kext_boundary_accuracy.png', dpi=200)
# fig.savefig('../data/kext_boundary_accuracy.pdf')
