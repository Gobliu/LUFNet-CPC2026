import sys
sys.path.append( '../../')

import itertools
import torch
import numpy as np
from utils.system_logs              import system_logs
from utils.mydevice                 import mydevice
import matplotlib.pyplot as plt


def de(e,npar):
    """Compute mean energy drift per particle.

    Args:
        e (torch.Tensor): Energy tensor of shape (trajectory, nsamples).
        npar (int): Number of particles.

    Returns:
        tuple[np.ndarray, np.ndarray]: Mean drift and standard error arrays.
    """
    #shape = [trajectory, nsamples]
    e = e.clone().detach().cpu().numpy()

    mean_e_all =  np.mean(e, axis=1)
    mean_e = abs(mean_e_all - mean_e_all[0]) / npar
    std_err_e = np.zeros(mean_e.shape)

    # e_shift = (e - e[0])/npar
    # e_shift = e_shift.clone().detach().cpu().numpy()
    # mean_e = np.mean(e_shift,axis=1)
    # std_err_e = np.std(e_shift,axis=1) / np.sqrt(e.shape[1])
    return mean_e, std_err_e


if __name__ == '__main__':
    # python plot_e_conserve.py 64 0.71 0.46 2 1000 20 l
    # python plot_e_conserve.py 64 0.85 0.9 3  1000 20 l


    _ = mydevice()
    _ = system_logs(mydevice)
    system_logs.print_start_logs()

    if torch.cuda.is_available():
        map_location = lambda storage, loc: storage.cuda()
    else:
        map_location = 'cpu'


    torch.set_default_dtype(torch.float64)

    torch.manual_seed(34952)

    argv = sys.argv
    if len(argv) != 8:
        print('usage <programe> <npar> <rho> <temp> <dim> <saved_pt> <gamma> <region>' )
        quit()

    npar = int(argv[1])
    rho = argv[2]
    T = argv[3]
    dim = int(argv[4])
    saved_pt = int(argv[5])
    gamma = int(argv[6])
    region = argv[7]

    if dim == 2:
        print('load md data on 2d .......')

        ml_dict = { "tau_long": 0.1,
                     'dpt' : 600000,
                     "saved_model" : 747
                     }

        data = {
            "saved_dir": "../../../data_sets/gen_by_ML/2d/lt0.1dpt{}_{}/".format(ml_dict["dpt"], region),
             "energy1" : "../../../data_sets/gen_by_MD/2d/noML-metric-lt0.01every1t0.7t1000/n{}rho{}T{}/energy_gamma{}_tmax1000.pt".format(npar,rho,T,gamma),
            "energy2": "../../../data_sets/gen_by_ML/2d/lt0.1dpt{}_{}/n{}rho{}T{}/energy_gamma{}mb{}_nsteps10000.pt".format(ml_dict["dpt"],region,npar,rho,T,gamma,ml_dict["saved_model"])}

    elif dim == 3:
        print('load md data on 3d .......')

        ml_dict = { "tau_long": 0.02,
                     'dpt' : 180000,
                     "saved_model" : 101
                     }

        data = {
            "saved_dir": "../../../data_sets/gen_by_ML/{}d/lt0.02dpt{}_{}/".format(dim,ml_dict["dpt"], region),
            "energy1": "../../../data_sets/gen_by_MD/{}d/noML-metric-lt0.001every0.1t0.14t100/n{}rho{}T{}/energy_gamma{}_tmax100.pt".format(
                dim, npar, rho, T, gamma),
            "energy2": "../../../data_sets/gen_by_ML/{}d/lt0.02dpt{}_{}/n{}rho{}T{}/energy_gamma20LUF{}_tmax100.pt".format( dim, ml_dict["dpt"], region, npar, rho, T, ml_dict["saved_model"])}

    else:
        assert False , 'invalid load data given .... '

    saved_model = ml_dict["saved_model"]

    print('load md data file : ', data["energy1"]) # shape [trajectory, nsamples]
    print('load ml data file : ', data["energy2"])

    data1 = torch.load(data["energy1"],map_location=map_location)
    e1_append = data1["energy"][:saved_pt+1]

    data2 = torch.load(data["energy2"],map_location=map_location)
    e2_append = data2["energy"][:saved_pt+1]

    de_in = e1_append[0] - e2_append[0]
    assert (torch.mean(de_in*de_in).item()< 1e-8), print('error ....  difference btw states too big', 'de', de)
    #print(e1_append.shape, e2_append.shape, e3_append.shape)

    mean_e1, std_err_e1 = de(e1_append,npar)
    mean_e2, std_err_e2 = de(e2_append,npar)

    print('saved energy shape', mean_e1.shape, std_err_e1.shape)
    print('saved energy shape', mean_e2.shape, std_err_e2.shape)

    plt.figure(figsize=(8, 6))

    # t = np.arange(0, saved_pt * 0.1 +0.1  ,0.1)
    t = np.arange(0, saved_pt * 1 + 1, 1)

    plt.title(r'n{}rho{}T{}$\gamma${}; mb{}'.format(npar,rho,T,gamma,saved_model),fontsize=20)
    plt.ylabel(r'$|<e>-<e_0>|/n$', fontsize=16)
    plt.errorbar(t, mean_e2, yerr=std_err_e2, errorevery=50, capsize=5, color='k',
                 label=r'LUFnet $\tau=${}'.format(ml_dict['tau_long']))
    if dim == 2:
        plt.xlabel(
            'time' + '\n' + 'maximum time t=1000' + '\n' + r'100000 iter (MD at $\tau=0.01$)' + '\n' + r'10000 iter (ML at $\tau=0.1$)',
            fontsize=18)
        plt.errorbar(t, mean_e1, yerr=std_err_e1, errorevery=50, capsize=5, color='b', label=r'$\tau$=0.01')

    elif dim == 3 :
        plt.xlabel(
            'time' + '\n' + 'maximum time t=100' + '\n' + r'100000 iter (MD at $\tau=0.001$)' + '\n' + r'5000 iter (ML at $\tau=0.02$)',
            fontsize=18)
        plt.errorbar(t, mean_e1, yerr=std_err_e1, errorevery=50, capsize=5, color='b', label=r'$\tau$=0.001')

    plt.tick_params(axis='x',   labelsize=16)
    plt.tick_params(axis='y', labelsize=16)
    # plt.ylim(-0.3,0.5) # liquid -0.15,0.5 / gas -0.09,0.5 / lg -0.3,1.1
    plt.grid()
    # plt.legend(loc='center right', fontsize=16)

    # plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    # plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

    plt.legend(fontsize=12)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'../../analysis/{dim}d/figures/energy/npar{npar}rho{rho}gamma{gamma}_e_T{T}.pdf', bbox_inches='tight', dpi=200)
    plt.close()
