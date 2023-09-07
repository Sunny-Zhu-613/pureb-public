import os
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
tableau = ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1', '#C85200', '#898989', '#A2C8EC', '#FFBC79', '#CFCFCF']

import pyqet

hf_file = lambda *x: os.path.join('data', *x)

np_rng = np.random.default_rng()

dimA = 2
dimB = 2
num_XZ = 1
# (3,3,64,2145) (3,3,128,8385)
# (4,4,32,6545) (4,4,64,47905)

# num_layer = num_kext + 2 #for 2x2 bipartite, the number of parameter=4(k+1)-1
# num_layer = 50 #for 2x2 bipartite, #layer=50 should be big enough

alpha_werner = 1/dimA
alpha_list = np.linspace(0, 1, 50)
alpha_fine_list = np.linspace(alpha_werner*0.95, min(1,alpha_werner*1.34), 50)


# cha
model_cha = pyqet.cha.AutodiffREE(num_state=500, dim0=dimA, dim1=dimB)
ree_cha = []
for alpha_i in tqdm(alpha_list):
    ree_cha.append(model_cha.minimize_distance(pyqet.werner_state(dimA, alpha_i), maxiter=100, num_repeat=3, use_tqdm=False, distance_kind='REE')[0])
ree_cha = np.array(ree_cha)
ree_fine_cha = []
for alpha_i in tqdm(alpha_fine_list):
    ree_fine_cha.append(model_cha.minimize_distance(pyqet.werner_state(dimA, alpha_i), maxiter=100, num_repeat=3, use_tqdm=False, distance_kind='REE')[0])
ree_fine_cha = np.array(ree_fine_cha)

# import pickle
# with open(hf_file('20220918_nisq_werner.pkl'), 'rb') as fid:
#     tmp0 = pickle.load(fid)
#     ree_cha = tmp0['ree_cha']
#     ree_fine_cha = tmp0['ree_fine_cha']


# pureb
kext_layer_list = [(4,5),(8,9),(12,13)]
ree_pureb_list = []
ree_pureb_fine_list = []
for kext,num_layer in kext_layer_list:
    model_pureb = pyqet.pureb.QuantumPureBosonicExt(dimA, dimB, kext, num_XZ, num_layer)
    for alpha_i in tqdm(alpha_list):
        model_pureb.set_dm_target(pyqet.werner_state(dimA, alpha_i))
        ree_pureb_list.append(model_pureb.minimize_loss(num_repeat=3, print_freq=0, tol=1e-10, return_info=False))
    for alpha_i in tqdm(alpha_fine_list):
        model_pureb.set_dm_target(pyqet.werner_state(dimA, alpha_i))
        ree_pureb_fine_list.append(model_pureb.minimize_loss(num_repeat=3, print_freq=0, tol=1e-10, return_info=False))
ree_pureb_list = np.array(ree_pureb_list).reshape(len(kext_layer_list), len(alpha_list))
ree_pureb_fine_list = np.array(ree_pureb_fine_list).reshape(len(kext_layer_list), len(alpha_fine_list))

# Analytical method
ree_analytical = np.array([pyqet.werner_state_ree(dimA, x) for x in alpha_list])
ree_fine_analytical = np.array([pyqet.werner_state_ree(dimA, x) for x in alpha_fine_list])


tmp0 = np.stack([pyqet.werner_state(dimA,x) for x in alpha_list])
ree_ppt = pyqet.ppt.relative_entangled_entropy(tmp0, dimA, dimB)[0]
tmp0 = np.stack([pyqet.werner_state(dimA,x) for x in alpha_fine_list])
ree_fine_ppt = pyqet.ppt.relative_entangled_entropy(tmp0, dimA, dimB)[0]


# import pickle
# with open(hf_file('20220720_qpureb_werner3_k8.pkl'), 'wb') as fid:
#     tmp0 = dict(qee_werner_pureb=qee_werner_pureb, dimA=dimA, dimB=dimB, num_kext=num_kext,
#              num_XZ=num_XZ, num_layer=num_layer, alpha_list=alpha_list)
#     pickle.dump(tmp0, fid)

# z0 = pickle.load(open(hf_file('20220720_qpureb_werner3.pkl'), 'rb'))


fig,ax = plt.subplots(figsize=(6.4,4.8))
for ind0 in range(len(kext_layer_list)):
    tmp0 = f'PureB({kext_layer_list[ind0][0]}), #layer={kext_layer_list[ind0][1]}'
    ax.plot(alpha_list, ree_pureb_list[ind0], color=tableau[ind0+3], label=tmp0)
ax.plot(alpha_list, ree_analytical, ':', color=tableau[0], markersize=3, label='analytical', markerfacecolor='none')
ax.plot(alpha_list, ree_ppt, '+', color=tableau[1], label='PPT')
ax.plot(alpha_list, ree_cha, 'x', color=tableau[2], label='CHA')
ax.legend(ncol=2)
ax.set_xlim(0, 1)
ax.set_xlabel(r'$\alpha$')
ax.set_ylabel('relative entropy of entanglement')
# ax.set_title(f'Werner state $d={dimA}$')

axin = ax.inset_axes([0.15, 0.24, 0.53, 0.47])
for ind0 in range(len(kext_layer_list)):
    axin.plot(alpha_fine_list, ree_pureb_fine_list[ind0], color=tableau[ind0+3])
axin.plot(alpha_fine_list, ree_fine_analytical, ':', color=tableau[0], markersize=3, label='analytical', markerfacecolor='none')
tmp0 = ree_fine_ppt.copy()
tmp0[alpha_fine_list<alpha_werner] = np.nan
axin.plot(alpha_fine_list, tmp0, '+', color=tableau[1], label='PPT')
axin.plot(alpha_fine_list, ree_fine_cha, 'x', color=tableau[2], label='CHA')
axin.set_xlim(alpha_fine_list[0], alpha_fine_list[-1])
axin.set_yscale('log')
axin.set_xticks([0.5, 0.55, 0.6, 0.65])
hrect,hpatch = ax.indicate_inset_zoom(axin, edgecolor="red")
hrect.set_xy((hrect.get_xy()[0], -0.02))
hrect.set_height(0.05)
fig.tight_layout()
fig.savefig('tbd00.png', dpi=200)
# fig.savefig('data/20220918_nisq_werner.png', dpi=200)
# fig.savefig('data/20220918_nisq_werner.pdf')

# with open('data/20220918_nisq_werner.pkl', 'wb') as fid:
#     tmp0 = dict(alpha_list=alpha_list, kext_layer_list=kext_layer_list, ree_pureb_list=ree_pureb_list, ree_analytical=ree_analytical,
#             ree_ppt=ree_ppt, ree_cha=ree_cha, alpha_fine_list=alpha_fine_list, ree_pureb_fine_list=ree_pureb_fine_list,
#             ree_fine_analytical=ree_fine_analytical, ree_fine_ppt=ree_fine_ppt, ree_fine_cha=ree_fine_cha)
#     pickle.dump(tmp0, fid)

