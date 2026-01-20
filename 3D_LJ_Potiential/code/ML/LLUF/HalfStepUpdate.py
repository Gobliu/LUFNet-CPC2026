import torch
import numpy as np
import torch.nn as nn
from utils.mydevice import mydevice

from ML.LLUF.SingleParticle import SingleParticle
from ML.LLUF.MultiParticles import MultiParticles
from ML.LLUF.ReadoutStep import ReadoutStep
from utils.force_stat          import force_stat


class HalfStepUpdate(nn.Module):

    """Class HalfStepUpdate.
    
    Notes
    -----
    Computes a learned half-step update for positions or momenta in the integrator.
    """
    def __init__(self, prepare_data_obj, single_particle_net, multi_particle_net, readout_step_net, t_init, nnet=1):
        """Function __init__.
        
        Parameters
        ----------
        prepare_data_obj : Any
            PrepareData instance for constructing features.
        single_particle_net : Any
            Single-particle network module.
        multi_particle_net : Any
            Multi-particle network module.
        readout_step_net : Any
            Readout network module for updates.
        t_init : Any
            Initial time step scale for learnable tau.
        nnet : Any
            Number of learnable tau parameters.
        """
        super().__init__()

        self.prepare_data = prepare_data_obj  # prepare_data object
        self.single_par = SingleParticle(single_particle_net)
        self.multi_par = MultiParticles(multi_particle_net)
        self.update_step = ReadoutStep(readout_step_net)

        self.tau_init = np.random.rand(nnet) * t_init  # change form 0.01 to 0.001
        self.tau = nn.Parameter(torch.tensor(self.tau_init, device=mydevice.get()))
        self.f_stat = force_stat()

    # for update q <- q + tau[2]*p + f_q
    # see LLUF_MD for use of this function

    def forward(self,q_input_list,p_input_list,q_prev):
        """Function forward.
        
        Parameters
        ----------
        q_input_list : Any
            List of position feature tensors over the input trajectory.
        p_input_list : Any
            List of momentum feature tensors over the input trajectory.
        q_prev : Any
            Previous-step positions tensor.
        
        Returns
        -------
        Any
            Scaled per-particle update tensor for the half step.
        """
        x = self.prepare_data.cat_qp(q_input_list,p_input_list)
        # shape [nsamples, nparticles, traj_len, ngrids * DIM * (q,p)]
        x = self.single_par.eval(x)
        # shape [nsample, nparticle, embed_dim]
        x = self.multi_par.eval(x, q_prev)
        # shape=[nsample,nparticle,embed_dim]
        x = self.update_step.eval(x)
        # shape=[nsample,nparticle,dim=2]
        self.f_stat.accumulate(x)
        return x * torch.abs(self.tau) # return the update step


