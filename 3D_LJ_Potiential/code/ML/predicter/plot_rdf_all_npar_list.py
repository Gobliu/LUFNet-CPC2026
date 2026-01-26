import sys
sys.path.append( '../../')

import torch
from utils.system_logs              import system_logs
from utils.mydevice                 import mydevice
import math
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    # python plot_rdf_all_npar_list.py  3.1 3.9 1.46 1.68
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

    saved_model = argv[1]
    y1limf = float(argv[2])
    y1limb = float(argv[3])
    y2limf = float(argv[4])
    y2limb = float(argv[5])

    print(f'saved model ... {saved_model}')
    load_file1 = f'../../analysis/3d/LUF{saved_model}/npar64rho0.85gamma20_rdf.txt'
    load_file2 = f'../../analysis/3d/LUF{saved_model}/npar128rho0.85gamma20_rdf.txt'
    load_file3 = f'../../analysis/3d/LUF{saved_model}/npar256rho0.85gamma20_rdf.txt'

    with open(load_file1) as f:
        rdf1 = np.genfromtxt(load_file1, filling_values=np.nan)
    print('load rdf1...',rdf1.shape)
    print(rdf1.shape)

    with open(load_file2) as f:
        rdf2 = np.genfromtxt(load_file2, filling_values=np.nan)
    print('load rdf2...',rdf2.shape)
    print(rdf2.shape)

    with open(load_file3) as f:
        rdf3 = np.genfromtxt(load_file3, filling_values=np.nan)
    print('load rdf3...',rdf3.shape)
    print(rdf3.shape)

    rdf = np.stack([rdf1,rdf2,rdf3])

    npar_list = ['n=64', 'n=128', 'n=256']

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 7), gridspec_kw = {'wspace':0,'hspace':0.05})

    for j in np.arange(len(npar_list)):
       print('rdf mc', npar_list[j], f'1st rmid {rdf[j, 1]:.3f}, {rdf[j, 2]:.3f}, gr, {rdf[j,3]:.3f}, {rdf[j,4]:.3f}',
                                    f'2nd rmid {rdf[j, 5]:.3f}, {rdf[j, 6]:.3f}, gr, {rdf[j, 7]:.3f}, {rdf[j, 8]:.3f}')
       print('rdf vv', npar_list[j], f'1st rmid {rdf[j, 9]:.3f}, {rdf[j, 10]:.3f}, gr, {rdf[j, 11]:.3f}, {rdf[j, 12]:.3f}',
                                    f'2nd rmid {rdf[j,13]:.3f}, {rdf[j,14]:.3f}, gr, {rdf[j, 15]:.3f}, {rdf[j, 16]:.3f}')
       print('rdf ml', npar_list[j], f'1st rmid {rdf[j, 17]:.3f}, {rdf[j, 18]:.3f}, gr, {rdf[j, 19]:.3f}, {rdf[j, 20]:.3f}',
                                    f'2nd rmid {rdf[j,21]:.3f}, {rdf[j,22]:.3f}, gr, {rdf[j, 23]:.3f}, {rdf[j, 24]:.3f}')
       print('')

       axes[0].errorbar([j+1], rdf[j, 3], yerr=rdf[j, 4], capsize=5,elinewidth=0.5,  color='k', markerfacecolor='none',marker='o',linestyle='none', markersize=12,label='MC')
       axes[0].errorbar([j+1], rdf[j, 11], yerr=rdf[j, 12], capsize=5,elinewidth=0.5, color='k', markerfacecolor='none',marker='^',linestyle='none', markersize=12,label='VV')
       axes[0].errorbar([j + 1], rdf[j, 19], yerr=rdf[j, 20], capsize=5, elinewidth=0.5, color='k',
                        markerfacecolor='none', marker='x', linestyle='none', markersize=12,label='LUFnet')

       axes[1].errorbar([j+1], rdf[j, 7], yerr=rdf[j, 8], capsize=5,elinewidth=0.5, color='k', markerfacecolor='none',marker='o',linestyle='none', markersize=12)
       axes[1].errorbar([j+1], rdf[j, 15], yerr=rdf[j, 16], capsize=5,elinewidth=0.5,  color='k', markerfacecolor='none',marker='^',linestyle='none', markersize=12)
       axes[1].errorbar([j + 1], rdf[j, 23], yerr=rdf[j, 24], capsize=5, elinewidth=0.5, color='k',
                        markerfacecolor='none', marker='x', linestyle='none', markersize=12)

       if j == 0 :
           axes[0].legend()

       # axes[0].set_ylim(y1limf,y1limb)
       # axes[0].set_yticks([y1limf, y1limb])

       # axes[0].set_yticks([7, 10, 13])
       # axes[0].set_yticks([4.6, 5.8, 7])
       # axes[0].set_yticks([3.2, 3.6])

       # axes[1].set_ylim(y2limf,y2limb)
       # axes[1].set_yticks([y2limf, y2limb])

       # axes[1].set_yticks([1.5, 2, 2.5])
       # axes[1].set_yticks([1.6, 2.2, 2.8])
       # axes[1].set_yticks([1.5, 1.6])

       axes[0].set_ylabel(r'$g(r_1)$',fontsize=14)
       axes[1].set_ylabel(r'$g(r_2)$',fontsize=14)

       #ax.plot(np.array(nrmid), np.array(ngr), color='k',label='t={:.2f}'.format(input_seq_idx * tau_long + step[j] * tau_long))
       for ax in axes.flat:
        ax.grid(True, axis='x', linestyle='--')
        # ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        # ax.tick_params(axis='y', which='both',  right=False, labelleft=False)
        # ax.set_yticks([1, 1.5,  2])
        # ax.set_yticklabels([1,"",2])
        ax.tick_params(axis='y', labelsize=14)
        ax.set_xticks(ticks=[1, 2, 3], labels=npar_list,fontsize=14)

    # plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()

    # plt.show()
    saved_dir =  f'../../analysis/3d/figures/rdf/rho0.85T0.9gamma20t0.35_rdf.pdf'
    plt.savefig(saved_dir, bbox_inches='tight', dpi=200)
    plt.close()
