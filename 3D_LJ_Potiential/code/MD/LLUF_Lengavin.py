"""MD module: LLUF Langevin update step for learned integrator.

Notes:
- Part of 3D Lennard-Jones MD workflow.
"""
import torch
import numpy as np
import torch.nn as nn
from utils.mydevice import mydevice
from hamiltonian.thermostat import thermostat_ML

from utils.pbc import pbc
from utils.utils import assert_nan

class LLUF_Lengavin(nn.Module):
    """Learned Langevin integrator step for LLUF model."""

    def __init__(self,prepare_data, LLUF_update_p, LLUF_update_q, tau_long, t_init=1, nnet=1):
        """Initialize the LLUF Langevin stepper.

        Args:
        prepare_data: Helper to build feature inputs for q/p.
        LLUF_update_p: Model to update p given features.
        LLUF_update_q: Model to update q given features.
        tau_long (float): Long time step size.
        t_init (float): Initial scale for learned tau.
        nnet (int): Number of learned tau values.
        """
        super().__init__()

        self.prepare_data = prepare_data
        self.LLUF_update_p = LLUF_update_p
        self.LLUF_update_q = LLUF_update_q
        self.tau_long = tau_long # 20250810 add tau_long
        self.tau_init = np.random.rand(nnet) * t_init  # change form 0.01 to 0.001

        self.tau = nn.Parameter(torch.tensor(self.tau_init, device=mydevice.get()))
        print(' velocity verletx ')

        #assert(self.modef == 'ff'),'hf mode not implemented in velocity_verlet3'

    # q_input_list [phi0,phi1,phi2,...] -- over time points
    # p_input_list [pi0,pi1,pi2,...]

    # 20250810 tau_long -- remove tau_long
    def one_step(self,q_input_list,p_input_list,q_pre,p_pre,l_list, gamma=0, temp=0):
        """Advance one learned Langevin step.

        Args:
        q_input_list (list[torch.Tensor]): Q feature history list.
        p_input_list (list[torch.Tensor]): P feature history list.
        q_pre (torch.Tensor): Previous q, shape [nsamples*nparticles, dim].
        p_pre (torch.Tensor): Previous p, shape [nsamples*nparticles, dim].
        l_list (torch.Tensor): Box sizes, shape [nsamples*nparticles, dim].
        gamma (float): Friction coefficient.
        temp (float): Temperature.

        Returns:
        Tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        Updated (q_input_list, p_input_list, q_cur, p_cur, l_list).
        """

        # q_input_list [phi0,phi1,phi2,...]  -- over time point
        # phi0.shape = [nsamples*nparticles, ngrids*DIM]
        # p_input_list [pi0,pi1,pi2,...]

        q_cur = q_pre  + p_pre * self.tau + self.LLUF_update_q(q_input_list, p_input_list, q_pre)
        q_cur = pbc(q_cur, l_list)

        q_input_next = self.prepare_data.prepare_q_feature_input(q_cur, l_list)
        q_input_list.append(q_input_next)
        q_input_list.pop(0)

        p_cur = p_pre + self.LLUF_update_p(q_input_list,p_input_list, q_cur) # SJ coord

        p_cur = thermostat_ML(p_cur, gamma, temp, self.tau_long)

        p_input_cur = self.prepare_data.prepare_p_feature_input(q_cur,p_cur,l_list)
        p_input_list.append(p_input_cur)
        p_input_list.pop(0)  # remove first element

        assert_nan(p_cur)
        assert_nan(q_cur)
        return q_input_list,p_input_list,q_cur,p_cur,l_list


    def nsteps(self,q_input_list,p_input_list,q_pre,p_pre,l_list):
        """Apply one chained update step (single-step wrapper).

        Args:
        q_input_list (list[torch.Tensor]): Q feature history list.
        p_input_list (list[torch.Tensor]): P feature history list.
        q_pre (torch.Tensor): Previous q, shape [nsamples*nparticles, dim].
        p_pre (torch.Tensor): Previous p, shape [nsamples*nparticles, dim].
        l_list (torch.Tensor): Box sizes, shape [nsamples*nparticles, dim].

        Returns:
        Tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        Updated (q_input_list, p_input_list, q_cur, p_cur, l_list).
        """

        #assert(n_chain==1),'MD/velocity_verletx,py: error only n_chain = 1 is implemented '

        # our mbpw-net model chain up to predict the new configuration for n-times
        q_input_list,p_input_list,q_cur,p_cur,l_list = \
                                  self.one_step(q_input_list,p_input_list,q_pre,p_pre,l_list)

        return q_input_list,p_input_list,q_cur,p_cur,l_list
