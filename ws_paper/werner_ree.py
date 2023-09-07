import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
tableau = ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1', '#C85200', '#898989', '#A2C8EC', '#FFBC79', '#CFCFCF']

import pyqet

np_rng = np.random.default_rng()

dimA = 3
dimB = 3
# cha: 1 minute for 3x3, 7 minutes for 4x4

alpha_werner = 1/dimA
alpha_list = np.linspace(0, 1, 50)
# alpha_fine_list = np.linspace(alpha_werner*0.95, min(1,alpha_werner*1.4), 50)#dimA=3
alpha_fine_list = np.linspace(alpha_werner*0.95, min(1,alpha_werner*1.2), 50)

kext_list = [8, 16, 32, 64]
ree_pureb_list = []
ree_pureb_fine_list = []
for kext in kext_list:
    model_pureb = pyqet.pureb.PureBosonicExt(dimA, dimB, kext=kext)
    for alpha_i in tqdm(alpha_list):
        ree_pureb_list.append(model_pureb.minimize_distance(pyqet.werner_state(dimA, alpha_i), num_repeat=3, tol=1e-10, distance_kind='REE'))
    for alpha_i in tqdm(alpha_fine_list):
        ree_pureb_fine_list.append(model_pureb.minimize_distance(pyqet.werner_state(dimA, alpha_i), num_repeat=3, tol=1e-10, distance_kind='REE'))
ree_pureb_list = np.array(ree_pureb_list).reshape(len(kext_list), len(alpha_list))
ree_pureb_fine_list = np.array(ree_pureb_fine_list).reshape(len(kext_list), len(alpha_fine_list))


kext = 65536
model_pureb = pyqet.pureb.PureBosonicExt(dimA, dimB, kext=kext)
hf0 = lambda x: model_pureb.minimize_distance(pyqet.werner_state(dimA, x), num_repeat=1, tol=1e-10, distance_kind='REE')
# 0.47696940104166663
# Werner boundary, d=2, alpha(pureb//qetlab-BOS/qetlab)
# 4ext 0.64585//0.6668//0.6668
# 5ext 0.63661//0.63650//0.63650
# 6ext 0.61206//0.61556//0.61554
# 7ext 0.60019//0.60016//0.60017
# 8ext 0.58846//0.58840//NA
# 9ext 0.57918//0.57919//NA
# 10ext 0.57165//0.57168//NA
# 11ext 0.56547//0.56548//NA
# 12ext 0.56024//NA//NA
# 16ext 0.54575//NA//NA
# 512ext 0.50175//NA//NA
# 8192ext 0.50040//NA//NA
# 65536ext 0.50034//NA//NA

# use_PPT = 0;
# na = 2;
# hfBOS = @(alpha,kext) SymmetricExtension(full(WernerState(na, alpha)), kext, [na,na], use_PPT, 1);
# hfnoBOS = @(alpha,kext) SymmetricExtension(full(WernerState(na, alpha)), kext, [na,na], use_PPT, 0);


model_cha = pyqet.cha.AutodiffREE(num_state=500, dim0=dimA, dim1=dimB)
ree_cha = []
for alpha_i in tqdm(alpha_list):
    ree_cha.append(model_cha.minimize_distance(pyqet.werner_state(dimA, alpha_i), maxiter=100, num_repeat=3, distance_kind='REE')[0])
ree_cha = np.array(ree_cha)
ree_fine_cha = []
for alpha_i in tqdm(alpha_fine_list):
    ree_fine_cha.append(model_cha.minimize_distance(pyqet.werner_state(dimA, alpha_i), maxiter=100, num_repeat=3, distance_kind='REE')[0])
ree_fine_cha = np.array(ree_fine_cha)


ree_analytical = np.array([pyqet.werner_state_ree(dimA, x) for x in alpha_list])
ree_fine_analytical = np.array([pyqet.werner_state_ree(dimA, x) for x in alpha_fine_list])

tmp0 = np.stack([pyqet.werner_state(dimA,x) for x in alpha_list])
ree_ppt = pyqet.ppt.relative_entangled_entropy(tmp0, dimA, dimB)[0]
tmp0 = np.stack([pyqet.werner_state(dimA,x) for x in alpha_fine_list])
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
# ax.set_title(f'Werner state $d={dimA}$')

axin = ax.inset_axes([0.15, 0.24, 0.47, 0.47])
for ind0 in range(len(kext_list)):
    axin.plot(alpha_fine_list, ree_pureb_fine_list[ind0], color=tableau[ind0+3], label=f'pureb k={kext_list[ind0]}')
axin.plot(alpha_fine_list, ree_fine_analytical, ':', color=tableau[0], markersize=3, label='analytical', markerfacecolor='none')
tmp0 = ree_fine_ppt.copy()
tmp0[alpha_fine_list<alpha_werner] = np.nan
axin.plot(alpha_fine_list, tmp0, '+', color=tableau[1], label='PPT')
axin.plot(alpha_fine_list, ree_fine_cha, 'x', color=tableau[2], label='CHA')
axin.set_xlim(alpha_fine_list[0], alpha_fine_list[-1])
axin.set_yscale('log')
axin.tick_params(axis='both', which='major', labelsize=11)
if dimA==3:
    axin.set_xticks([0.33, 0.37, 0.41, 0.45])
elif dimA==2:
    axin.set_xticks([0.5, 0.54, 0.58])
hrect,hpatch = ax.indicate_inset_zoom(axin, edgecolor="red")
hrect.set_xy((hrect.get_xy()[0], -0.02))
hrect.set_height(0.05)
fig.tight_layout()
fig.savefig('tbd00.png', dpi=200)
# fig.savefig('../data/20220818_werner.png', dpi=200)
# fig.savefig('../data/20220818_werner.pdf')

# with open('data/20220818_werner.pkl', 'wb') as fid:
#     tmp0 = dict(alpha_list=alpha_list, kext_list=kext_list, ree_pureb_list=ree_pureb_list, ree_analytical=ree_analytical,
#             ree_ppt=ree_ppt, ree_cha=ree_cha, alpha_fine_list=alpha_fine_list, ree_pureb_fine_list=ree_pureb_fine_list,
#             ree_fine_analytical=ree_fine_analytical, ree_fine_ppt=ree_fine_ppt, ree_fine_cha=ree_fine_cha)
#     pickle.dump(tmp0, fid)
