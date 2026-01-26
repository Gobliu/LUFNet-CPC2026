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

    saved_model= argv[1]
    y1limf = float(argv[2])
    y1limb = float(argv[3])
    y2limf = float(argv[4])
    y2limb = float(argv[5])

    print(f'saved model ... {saved_model}')

    load_file11 = f'../../analysis/3d/LUF{saved_model}/npar64rho0.85gamma20_e.txt'
    load_file12 = f'../../analysis/3d/LUF{saved_model}/npar64rho0.85gamma20_cv.txt'

    load_file21 = f'../../analysis/3d/LUF{saved_model}/npar128rho0.85gamma20_e.txt'
    load_file22 = f'../../analysis/3d/LUF{saved_model}/npar128rho0.85gamma20_cv.txt'

    load_file31 = f'../../analysis/3d/LUF{saved_model}/npar256rho0.85gamma20_e.txt'
    load_file32 = f'../../analysis/3d/LUF{saved_model}/npar256rho0.85gamma20_cv.txt'

    with open(load_file11) as f:
        e1 = np.genfromtxt(load_file11, filling_values=np.nan)
        # shape [temp_list, 5] ; 5 is [(idx, vv_mean, vv_std, ml_mean, ml_std)]
    print('load energy...',e1.shape)

    with open(load_file12) as f:
        cv1 = np.genfromtxt(load_file12, filling_values=np.nan)
        # shape [temp_list, 5] ; 5 is [(idx, vv_mean, vv_std, ml_mean, ml_std)]
    print('load Cv...', cv1.shape)

    with open(load_file21) as f:
        e2 = np.genfromtxt(load_file21, filling_values=np.nan)
        # shape [temp_list, 5] ; 5 is [(idx, vv_mean, vv_std, ml_mean, ml_std)]
    print('load energy...', e2.shape)

    with open(load_file22) as f:
        cv2 = np.genfromtxt(load_file22, filling_values=np.nan)
        # shape [temp_list, 5] ; 5 is [(idx, vv_mean, vv_std, ml_mean, ml_std)]
    print('load Cv...', cv2.shape)

    with open(load_file31) as f:
        e3 = np.genfromtxt(load_file31, filling_values=np.nan)
        # shape [temp_list, 5] ; 5 is [(idx, vv_mean, vv_std, ml_mean, ml_std)]
    print('load energy...', e3.shape)

    with open(load_file32) as f:
        cv3 = np.genfromtxt(load_file32, filling_values=np.nan)
        # shape [temp_list, 5] ; 5 is [(idx, vv_mean, vv_std, ml_mean, ml_std)]
    print('load Cv...', cv3.shape)

    e = np.stack([e1, e2, e3])
    cv = np.stack([cv1, cv2, cv3])

    npar_list = ['n=64', 'n=128', 'n=256']

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7, 6), gridspec_kw = {'wspace':0,'hspace':0.05})

    for j in np.arange(len(npar_list)):

       print(f'e vv , {npar_list[j]}, {e[j,1]:.3f}, {e[j,2]:.3f}')
       print(f'e ml, {npar_list[j]}, {e[j,3]:.3f}, {e[j,4]:.3f}')
       print(f'cv vv, {npar_list[j]}, {cv[j, 1]:.3f}, {cv[j, 2]:.3f}')
       print(f'cv ml, {npar_list[j]}, {cv[j, 3]:.3f}, {cv[j, 4]:.3f}')
       print('')

       axes[0].errorbar([j+1], e[j, 1], yerr=e[j, 2], capsize=5,elinewidth=0.5,  color='k', markerfacecolor='none',marker='^',linestyle='none', markersize=12,label='VV')
       axes[0].errorbar([j+1], e[j, 3], yerr=e[j, 4], capsize=5,elinewidth=0.5, color='k', markerfacecolor='none',marker='x',linestyle='none', markersize=12,label='LUFnet')

       axes[1].errorbar([j+1], cv[j, 1], yerr=cv[j, 2], capsize=5,elinewidth=0.5, color='k', markerfacecolor='none',marker='^',linestyle='none', markersize=12)
       axes[1].errorbar([j+1], cv[j, 3], yerr=cv[j, 4], capsize=5,elinewidth=0.5,  color='k', markerfacecolor='none',marker='x',linestyle='none', markersize=12)
       #
       axes[0].set_ylim(y1limf,y1limb)
       axes[1].set_ylim(y2limf, y2limb)

       axes[0].set_ylabel(r'$\Delta$E',fontsize=14)
       axes[1].set_ylabel(r'$\Delta$Cv',fontsize=14)

       if j == 0 :
           axes[0].legend()

       for ax in axes.flat:
        ax.grid(True, axis='x', linestyle='--')
        # ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        # ax.tick_params(axis='y', which='both',  right=False, labelleft=False)
        # ax.set_yticks([1, 1.5,  2])
        # ax.set_yticklabels([1,"",2])
        ax.tick_params(axis='y', labelsize=14)
        ax.set_xticks(ticks=[1, 2, 3], labels=npar_list,fontsize=14)

    plt.tight_layout()
    # plt.show()
    saved_dir = f'../../analysis/3d/figures/e_cv/rho0.85T0.9gamma20t0.35_e_cv.pdf'
    plt.savefig(saved_dir, bbox_inches='tight', dpi=200)
    plt.close()
