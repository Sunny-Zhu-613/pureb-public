import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
tableau = ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1', '#C85200', '#898989', '#A2C8EC', '#FFBC79', '#CFCFCF']

import pyqet

np_rng = np.random.default_rng()

dimA = 3
dimB = 3
alpha_isotropic = 1/(dimA+1)
alpha_list = np.linspace(0, 1, 50)
alpha_fine_list = np.linspace(alpha_isotropic*0.95, min(1,alpha_isotropic*1.44), 50)

kext_list = [8, 16, 32, 64]
ree_pureb_list = []
ree_pureb_fine_list = []
for kext in kext_list:
    model_pureb = pyqet.pureb.PureBosonicExt(dimA, dimB, kext=kext)
    for alpha_i in tqdm(alpha_list):
        ree_pureb_list.append(model_pureb.minimize_distance(pyqet.isotropic_state(dimA, alpha_i), num_repeat=3, tol=1e-10, distance_kind='REE'))
    for alpha_i in tqdm(alpha_fine_list):
        ree_pureb_fine_list.append(model_pureb.minimize_distance(pyqet.isotropic_state(dimA, alpha_i), num_repeat=3, tol=1e-10, distance_kind='REE'))
ree_pureb_list = np.array(ree_pureb_list).reshape(len(kext_list), len(alpha_list))
ree_pureb_fine_list = np.array(ree_pureb_fine_list).reshape(len(kext_list), len(alpha_fine_list))

model_cha = pyqet.cha.AutodiffREE(num_state=500, dim0=dimA, dim1=dimB)
ree_cha = []
for alpha_i in tqdm(alpha_list):
    ree_cha.append(model_cha.minimize_distance(pyqet.isotropic_state(dimA, alpha_i), maxiter=100, num_repeat=3, distance_kind='REE')[0])
ree_cha = np.array(ree_cha)
ree_fine_cha = []
for alpha_i in tqdm(alpha_fine_list):
    ree_fine_cha.append(model_cha.minimize_distance(pyqet.isotropic_state(dimA, alpha_i), maxiter=100, num_repeat=3, distance_kind='REE')[0])
ree_fine_cha = np.array(ree_fine_cha)


ree_analytical = np.array([pyqet.isotropic_state_ree(dimA, x) for x in alpha_list])
ree_fine_analytical = np.array([pyqet.isotropic_state_ree(dimA, x) for x in alpha_fine_list])

tmp0 = np.stack([pyqet.isotropic_state(dimA,x) for x in alpha_list])
ree_ppt = pyqet.ppt.relative_entangled_entropy(tmp0, dimA, dimB)[0]
tmp0 = np.stack([pyqet.isotropic_state(dimA,x) for x in alpha_fine_list])
ree_fine_ppt = pyqet.ppt.relative_entangled_entropy(tmp0, dimA, dimB)[0]

fig,ax = plt.subplots(figsize=(6.4,4.8))
for ind0 in range(len(kext_list)):
    ax.plot(alpha_list, ree_pureb_list[ind0], color=tableau[ind0+3], label=f'PureB({kext_list[ind0]})')
ax.plot(alpha_list, ree_analytical, ':', color=tableau[0], markersize=3, label='analytical', markerfacecolor='none')
ax.plot(alpha_list, ree_ppt, '+', color=tableau[1], label='PPT')
ax.plot(alpha_list, ree_cha, 'x', color=tableau[2], label='CHA')
ax.legend(ncol=2, fontsize=12, loc='upper left')
ax.set_xlim(0, 1)
ax.set_xlabel(r'$\alpha$', fontsize=12)
ax.set_ylabel('relative entropy of entanglement', fontsize=12)
ax.tick_params(axis='both', which='major', labelsize=11)

axin = ax.inset_axes([0.1, 0.24, 0.47, 0.47])
for ind0 in range(len(kext_list)):
    axin.plot(alpha_fine_list, ree_pureb_fine_list[ind0], color=tableau[ind0+3], label=f'pureb k={kext_list[ind0]}')
axin.plot(alpha_fine_list, ree_fine_analytical, ':', color=tableau[0], markersize=3, label='analytical', markerfacecolor='none')
axin.plot(alpha_fine_list, ree_fine_ppt, '+', color=tableau[1], label='PPT')
axin.plot(alpha_fine_list, ree_fine_cha, 'x', color=tableau[2], label='CHA')
axin.set_xlim(alpha_fine_list[0], alpha_fine_list[-1])
axin.set_yscale('log')
axin.tick_params(axis='both', which='major', labelsize=11)
axin.set_xticks([0.25, 0.3, 0.35])
hrect,hpatch = ax.indicate_inset_zoom(axin, edgecolor="red")
hrect.set_xy((hrect.get_xy()[0], -0.02))
hrect.set_height(0.05)
fig.tight_layout()
fig.savefig('tbd00.png', dpi=200)
# fig.savefig('../data/20220818_isotropic.png', dpi=200)
# fig.savefig('../data/20220818_isotropic.pdf')

# with open('data/20220818_isotropic.pkl', 'wb') as fid:
#     tmp0 = dict(alpha_list=alpha_list, kext_list=kext_list, ree_pureb_list=ree_pureb_list, ree_analytical=ree_analytical,
#             ree_ppt=ree_ppt, ree_cha=ree_cha, alpha_fine_list=alpha_fine_list, ree_pureb_fine_list=ree_pureb_fine_list,
#             ree_fine_analytical=ree_fine_analytical, ree_fine_ppt=ree_fine_ppt, ree_fine_cha=ree_fine_cha)
#     pickle.dump(tmp0, fid)
