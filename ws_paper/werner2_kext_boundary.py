import pyqet

dim = 2
dm0 = pyqet.werner_state(dim, alpha=1)
kext_list = [7,8]
werner_pureb_alpha_list = []
werner_qetlab_alpha_list = []

for kext in kext_list:
    model = pyqet.pureb.PureBosonicExt(dim, dim, kext=kext)
    pureb_alpha = model.solve_boundary(dm0, alpha_cha=0, num_repeat=3, xtol=1e-5, use_tqdm=False)[0]
    qelab_alpha = pyqet.qetlab.qetlab_kext_boundary(dm0, dim, dim, kext, tol=1e-5, use_BOS=True)
    werner_pureb_alpha = pureb_alpha*dim/(pureb_alpha+dim - 1)
    werner_qelab_alpha = qelab_alpha*dim/(qelab_alpha+dim - 1)
    werner_qetlab_alpha_list.append(werner_qelab_alpha)
    werner_pureb_alpha_list.append(werner_pureb_alpha)
    print(kext, werner_pureb_alpha, werner_qelab_alpha)

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
