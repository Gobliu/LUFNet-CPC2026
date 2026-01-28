import sys
sys.path.append( '../../')

import itertools
import torch
import numpy as np
from utils.system_logs              import system_logs
from utils.mydevice                 import mydevice
from hamiltonian.lennard_jones2d    import lennard_jones2d

def pack_data(qpl_list, idx):
    """Extract q/p/l snapshots at a given time index.

    Args:
        qpl_list (torch.Tensor): Trajectory tensor of shape
            (nsamples, 3, traj_len, nparticles, dim).
        idx (int): Time index to extract.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: q_init, p_init, l_init.
    """

    q_init = qpl_list[:,0,idx,:,:].clone().detach()
    p_init = qpl_list[:,1,idx,:,:].clone().detach()
    l_init = qpl_list[:,2,idx,:,:].clone().detach()

    q_init = mydevice.load(q_init)
    p_init = mydevice.load(p_init)
    l_init = mydevice.load(l_init)

    return q_init,p_init,l_init

def total_energy(potential_function, q_list, p_list, l_list):
    """Compute kinetic and potential energy for a batch.

    Args:
        potential_function: Object with a `total_energy` method.
        q_list (torch.Tensor): Positions of shape (nsamples, nparticles, dim).
        p_list (torch.Tensor): Momenta of shape (nsamples, nparticles, dim).
        l_list (torch.Tensor): Box sizes of shape (nsamples, nparticles, dim).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Kinetic and potential energies per sample.
    """
    pe = potential_function.total_energy(q_list, l_list)
    ke = torch.sum(p_list * p_list, dim=(1, 2)) * 0.5
    return ke, pe


if __name__ == '__main__':
    # md : python e_conserve.py 64 0.85 0.9 3 0.001 1000 20 None 1000 l
    # ml : python e_conserve.py 64 0.85 0.9 3 0.05 1000 20 065 180000 l

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
    if len(argv) != 11:
        print('usage <programe> <npar> <rho> <temp> <dim> <tau_long> <saved_tp> <gamma> <saved_model> <dpt> <region>' )
        quit()

    npar = argv[1]
    rho = argv[2]
    T = argv[3]
    dim = int(argv[4])
    tau_long = float(argv[5])
    saved_tp = int(argv[6])
    gamma = int(argv[7])
    saved_model = argv[8]
    dpt = int(argv[9])
    region = argv[10]

    if saved_model.strip() == "None":
       saved_model = None
    else:
       saved_model = saved_model.strip()

    if saved_model is None:
        if dim == 2:
            print('load md data on 2d .......')
            data = {
                     "filename" : '../../../data_sets/gen_by_MD/noML-metric-lt{}every1t0.7t100/n{}rho{}T{}/'.format(tau_long,npar,rho,T) +
                     'n{}rho{}T{}gamma{}.pt'.format(npar,rho,T,gamma),
                     "save_dir": "../../../data_sets/gen_by_MD/noML-metric-lt{}every0.1t0.7t100/n{}rho{}T{}/energy_gamma{}_tmax100.pt".format( tau_long,
                      npar, rho, T, gamma)}

        elif dim ==3:
            print('load md data on 3d .......')
            data = {
                     "filename": '../../../data_sets/gen_by_MD/{}d/noML-metric-lt{}every0.1t0.14t100/n{}rho{}T{}/'.format(dim, tau_long, npar, rho, T) +
                            'n{}rho{}T{}gamma{}.pt'.format(npar, rho, T, gamma),  # 3d
                      "save_dir": "../../../data_sets/gen_by_MD/{}d/noML-metric-lt{}every0.1t0.14t100/n{}rho{}T{}/energy_gamma{}_tmax100.pt".format( dim, tau_long,
                    npar, rho, T, gamma) }


    else:
        print(f'load ml data on {dim}d .......')
        data = {
                 "filename" : '../../../data_sets/gen_by_ML/lt{}dpt{}_{}/n{}rho{}T{}/'.format(tau_long,dpt,region,npar,rho,T) + 'pred_n{}len08ws08gamma{}LUF{}_tau{}.pt'.format(npar,gamma,saved_model,tau_long) ,
                 "save_dir": "../../../data_sets/gen_by_ML/lt{}dpt{}_{}/n{}rho{}T{}/".format(tau_long,dpt, region, npar, rho, T) + 'energy_gamma{}LUF{}_tmax100.pt'.format(
                 gamma, saved_model, saved_tp)}

    print(data)
    lj = lennard_jones2d()

    data1 = torch.load(data["filename"],map_location=map_location)
    qpl_traj = data1['qpl_trajectory']
    print('traj shape ', qpl_traj.shape)

    print('load filename ..', data["filename"])
    print('calc energy .......')

    tot_u_append = []
    tot_k_append = []
    tot_e_append = []
    for t in range(qpl_traj.shape[2]): # trajectory length
      q_list, p_list, l_list = pack_data(qpl_traj, t)
      tot_k, tot_u = total_energy(lj, q_list, p_list, l_list) # shape [nsamples]
      # print('t===', t, 'q', q_list.min().item(), q_list.max().item(), 'p', p_list.min().item(), p_list.max().item(), 'u', min(tot_u).item(),max(tot_u).item(), 'k', min(tot_k).item(), max(tot_k).item())
      tot_u_append.append(tot_u)
      tot_k_append.append(tot_k)
      tot_e_append.append(tot_u+tot_k)

    print('finished calc energy .....')
    tot_u_append = torch.stack(tot_u_append, dim=0)  # shape [trajectory, nsamples]
    tot_k_append = torch.stack(tot_k_append, dim=0)  # shape [trajectory, nsamples]
    tot_e_append = torch.stack(tot_e_append,dim=0) # shape [trajectory, nsamples]
    print(torch.min(tot_u_append).item(), torch.max(tot_u_append).item())

    torch.save({'pe':tot_u_append, 'ke':tot_k_append, 'energy': tot_e_append},data["save_dir"])
    print('save dir ..', data["save_dir"])
