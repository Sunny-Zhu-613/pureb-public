import os
import time
import tempfile
import subprocess
from tqdm import tqdm
import numpy as np
import scipy.io

_ppt_ree_script_fmt = '''
infile = "{infile}";
outfile = "{outfile}";

load(infile, "rho");
load(infile, "na");
load(infile, "nb");
old_size = size(rho);
na = double(na);
nb = double(nb);
rho = reshape(rho, na*nb, na*nb, []);
tau = zeros(size(rho));
ree_value = zeros(1, size(rho,3));
for ind0 = 1:size(rho,3)
    [tau(:,:,ind0),ree_value(ind0)] = ppt_ree(rho(:,:,ind0), na, nb);
    disp("%progress-tag%");
end

save(outfile, "tau", "ree_value", "rho");

function [tau,cvx_optval] = ppt_ree(rho, na, nb)
if nargin<2
    na = round(sqrt(size(rho, 1)));
    nb = na;
end
cvx_begin sdp quiet
    variable tau(na*nb,na*nb) hermitian;
    minimize (quantum_rel_entr(rho,tau)/log(2));
    tau >= 0;
    trace(tau) == 1;
    Tx(tau,2,[na nb]) >= 0; % Positive partial transpose constraint
cvx_end
end
'''

def PPT_relative_entangled_entropy(density_matrix, dimA, dimB, use_tqdm=True, workdir=None):
    # matlab -nodisplay -r "cd {workdir}; draft00; exit"
    if workdir is None:
        workdir_handle = tempfile.TemporaryDirectory()
        workdir = workdir_handle.name
    else:
        workdir_handle = None
    hf_file = lambda *x: os.path.join(workdir, *x)
    old_shape = density_matrix.shape
    assert (old_shape[-1]==(dimA*dimB)) and (old_shape[-2]==(dimA*dimB))
    density_matrix = density_matrix.reshape(-1, dimA*dimB, dimA*dimB)
    tmp0 = {'rho':density_matrix.transpose(1,2,0), 'na':dimA, 'nb':dimB}
    infile_path = 'tbd00_in.mat'
    outfile_path = 'tbd00_out.mat'
    scipy.io.savemat(hf_file(infile_path), tmp0)
    runfile = 'tbd00_runfile.m'
    with open(hf_file(runfile), 'w', encoding='utf-8') as fid:
        fid.write(_ppt_ree_script_fmt.format(infile=infile_path, outfile=outfile_path))

    tmp0 = runfile.rsplit(".",1)[0]
    cmd = ['matlab', 'nodisplay', '-r', f'cd {workdir}; {tmp0}; exit']
    if use_tqdm:
        with subprocess.Popen(cmd, stdout=subprocess.PIPE) as proc:
            all_stdout = ''
            with tqdm(total=density_matrix.shape[0]) as pbar:
                while True:
                    tmp0 = proc.stdout.read(10).decode('utf-8')
                    all_stdout = all_stdout + tmp0
                    tmp0 = all_stdout.split('%progress-tag%')
                    pbar.update(len(tmp0)-1)
                    all_stdout = tmp0[-1]
                    poll_ret = proc.poll()
                    if poll_ret is not None:
                        break
                    time.sleep(0.1)
    else:
        subprocess.run(cmd)

    tmp0 = scipy.io.loadmat(hf_file(outfile_path))
    ppt_ree = tmp0['ree_value'].reshape(-1) * np.log(2)
    tau_list = tmp0['tau'].reshape(dimA*dimB,dimA*dimB,-1).transpose(2,0,1)
    if workdir_handle is not None:
        workdir_handle.cleanup()
    return ppt_ree, tau_list




_boson_kext_script_fmt = '''
infile = "{infile}";
outfile = "{outfile}";

load(infile, "rho");
load(infile, "na");
load(infile, "nb");
load(infile, "kext");
na = double(na);
nb = double(nb);
rho = reshape(rho, na*nb, na*nb, []);
load(infile, "use_PPT");
load(infile, "use_BOS");

is_kext = zeros(1, size(rho,3));
for ind0 = 1:size(rho,3)
    is_kext(ind0) = SymmetricExtension(rho(:,:,ind0), double(kext), [na,nb], use_PPT, use_BOS);
    disp("%progress-tag%");
end

save(outfile, "is_kext");
'''
def qetlab_boson_kext(density_matrix, dimA, dimB, kext, use_PPT=False, use_BOS=False, use_tqdm=True, workdir=None):
    if workdir is None:
        workdir_handle = tempfile.TemporaryDirectory()
        workdir = workdir_handle.name
    else:
        workdir_handle = None
    hf_file = lambda *x: os.path.join(workdir, *x)
    old_shape = density_matrix.shape
    assert (old_shape[-1]==(dimA*dimB)) and (old_shape[-2]==(dimA*dimB))
    density_matrix = density_matrix.reshape(-1, dimA*dimB, dimA*dimB)
    tmp0 = {'rho':density_matrix.transpose(1,2,0), 'na':dimA, 'nb':dimB, 'kext':kext, 'use_PPT':use_PPT, 'use_BOS':use_BOS}
    infile_path = 'tbd00_in.mat'
    outfile_path = 'tbd00_out.mat'
    scipy.io.savemat(hf_file(infile_path), tmp0)
    runfile = 'tbd00_runfile.m'
    with open(hf_file(runfile), 'w', encoding='utf-8') as fid:
        fid.write(_boson_kext_script_fmt.format(infile=infile_path, outfile=outfile_path))
    tmp0 = runfile.rsplit(".",1)[0]
    cmd = ['matlab', 'nodisplay', '-r', f'cd {workdir}; {tmp0}; exit']
    if use_tqdm:
        with subprocess.Popen(cmd, stdout=subprocess.PIPE) as proc:
            all_stdout = ''
            with tqdm(total=density_matrix.shape[0]) as pbar:
                while True:
                    tmp0 = proc.stdout.read(10).decode('utf-8')
                    all_stdout = all_stdout + tmp0
                    tmp0 = all_stdout.split('%progress-tag%')
                    pbar.update(len(tmp0)-1)
                    all_stdout = tmp0[-1]
                    poll_ret = proc.poll()
                    if poll_ret is not None:
                        break
                    time.sleep(0.1)
    else:
        subprocess.run(cmd)
    tmp0 = scipy.io.loadmat(hf_file(outfile_path))
    is_kext = tmp0['is_kext'].reshape(-1).astype(np.bool_)
    if len(old_shape)==2:
        is_kext = is_kext.item()
    else:
        is_kext = is_kext.reshape(old_shape[:-2])
    if workdir_handle is not None:
        workdir_handle.cleanup()
    return is_kext


_boson_kext_boundary_script_fmt = '''
infile = "{infile}";
outfile = "{outfile}";

load(infile, "rho_list");
load(infile, "na");
load(infile, "nb");
load(infile, "tol");
load(infile, "qetlab_tol");
load(infile, "kext");
load(infile, "use_PPT");
load(infile, "use_BOS");
na = double(na);
nb = double(nb);
rho_list = reshape(rho_list, na*nb, na*nb, []);

dm_eye = eye(size(rho_list,1))/size(rho_list,1);
ret_alpha = zeros(size(rho_list,3), numel(kext));
for ind0=1:numel(kext)
    kext_i = kext(ind0);
    for ind1 = 1:size(rho_list,3)
        rho_i = rho_list(:,:,ind1);
        hf_alpha = @(x) x*rho_i + (1-x)*dm_eye;
        hf0 = @(x) SymmetricExtension(hf_alpha(x), double(kext_i), [na,nb], use_PPT, use_BOS, qetlab_tol);
        alpha0 = 0;
        tmp0 = sort(eig(rho_i));
        alpha1 = 1/(1-size(rho_i,1)*tmp0(1));
        if hf0(alpha1)
            alpha_i = alpha1;
        else
            for ind2=1:ceil(log2((alpha1-alpha0)/tol))
                alpha_i = (alpha0+alpha1)/2;
                if hf0(alpha_i)
                    alpha0 = alpha_i;
                else
                    alpha1 = alpha_i;
                end
            end
            alpha_i = (alpha0+alpha1)/2;
        end
        disp("%progress-tag%");
        ret_alpha(ind1,ind0) = real(alpha_i);
    end
end
save(outfile, "ret_alpha");
'''
def qetlab_kext_boundary(density_matrix, dimA, dimB, kext, use_PPT=False, use_BOS=True, tol=1e-4, qetlab_tol=None, workdir=None, use_tqdm=True):
    if qetlab_tol is None:
        qetlab_tol = tol/10
    if workdir is None:
        workdir_handle = tempfile.TemporaryDirectory()
        workdir = workdir_handle.name
    else:
        workdir_handle = None
    hf_file = lambda *x: os.path.join(workdir, *x)
    old_shape = density_matrix.shape
    assert (old_shape[-1]==(dimA*dimB)) and (old_shape[-2]==(dimA*dimB))
    density_matrix = density_matrix.reshape(-1, dimA*dimB, dimA*dimB)
    kext = np.asarray(kext)
    tmp0 = {'rho_list':density_matrix.transpose(1,2,0), 'na':dimA, 'nb':dimB, 'kext':kext.reshape(-1), 'tol':tol, 'use_PPT':use_PPT, 'use_BOS':use_BOS, 'qetlab_tol':qetlab_tol}
    infile_path = 'tbd00_in.mat'
    outfile_path = 'tbd00_out.mat'
    scipy.io.savemat(hf_file(infile_path), tmp0)
    runfile = 'tbd00_runfile.m'
    with open(hf_file(runfile), 'w', encoding='utf-8') as fid:
        fid.write(_boson_kext_boundary_script_fmt.format(infile=infile_path, outfile=outfile_path))
    tmp0 = runfile.rsplit(".",1)[0]
    cmd = ['matlab', 'nodisplay', '-r', f'cd {workdir}; {tmp0}; exit']

    if use_tqdm:
        with subprocess.Popen(cmd, stdout=subprocess.PIPE) as proc:
            all_stdout = ''
            with tqdm(total=density_matrix.shape[0]*kext.size) as pbar:
                while True:
                    tmp0 = proc.stdout.read(10).decode('utf-8')
                    all_stdout = all_stdout + tmp0
                    tmp0 = all_stdout.split('%progress-tag%')
                    pbar.update(len(tmp0)-1)
                    all_stdout = tmp0[-1]
                    poll_ret = proc.poll()
                    if poll_ret is not None:
                        break
                    time.sleep(0.1)
    else:
        subprocess.run(cmd)

    ret_alpha = scipy.io.loadmat(hf_file(outfile_path))['ret_alpha']
    if len(old_shape)==2:
        ret_alpha = ret_alpha.reshape(-1)
    else:
        ret_alpha = ret_alpha.reshape(*old_shape[:-2], -1)
    ret_alpha = np.moveaxis(ret_alpha, -1, 0)
    if kext.ndim==0:
        ret_alpha = ret_alpha[0]
    if workdir_handle is not None:
        workdir_handle.cleanup()
    return ret_alpha
