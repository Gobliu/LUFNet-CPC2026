import torch
import torch.nn as nn
#from utils.mydevice import mydevice

from ML.LLUF.PhiFeatures import PhiFeatures
from ML.LLUF.PsiFeatures import PsiFeatures

from utils.pbc import pbc
from utils.pbc import _delta_state
# import matplotlib.pyplot as plt # 20250809 nscc no module ...

class PrepareData(nn.Module):

    # grid_object is the object to make grid center at every particle
    # e.g. HexGrids for making multiple layers of hexagonal grids
    # e.g. SingleGrid for making only one grid at each particle center
    """Class PrepareData.
    
    Notes
    -----
    Builds per-particle feature tensors for model inputs over a trajectory window.
    """
    def __init__(self, net, grid_object):
        """Function __init__.
        
        Parameters
        ----------
        net : Any
            PyTorch module used for feature extraction or updates.
        grid_object : Any
            Grid generator that returns grid points per particle.
        """
        super().__init__()
        # net : mb4pw -- use to extract features for position variable of grid point
        self.net = net
        self.grid_object = grid_object
        self.phi_features = PhiFeatures(self.grid_object,net)
        self.psi_features = PsiFeatures(self.grid_object)

    # ===================================================
    def cat_qp(self,q_input_list,p_input_list):
        # # q_input_list : mb net [phi0, phi1, phi2, ...] # phi0=(phi0x,phi0y)
        # # phi shape [nsamples, nparticles, ngrids * DIM]
        # # p_input_list : mb net [pi0, pi1, pi2, ...] # pi0=(pi0x,pi0y)
        # # return mbnet qp_cat phi, pi along time
        """Function cat_qp.
        
        Parameters
        ----------
        q_input_list : Any
            List of position feature tensors over the input trajectory.
        p_input_list : Any
            List of momentum feature tensors over the input trajectory.
        
        Returns
        -------
        Any
            Concatenated q/p feature tensor for the trajectory window.
        """
        q_input_list = torch.stack(q_input_list,dim=2)
        p_input_list = torch.stack(p_input_list,dim=2)
        # # phi shape [nsamples, nparticles, traj_len, ngrids * DIM]

        qp_cat = torch.cat((q_input_list,p_input_list), dim=-1)
        # shape [nsamples, nparticles, traj_len, ngrids * DIM * (q,p)]

        return qp_cat

    # ===================================================
    # def make_mask(self,nsamples,nparticles,dim):
    #     # mask to mask out only hexagonal grids centered at i-th particle
    #     # and then use to make zero of fields which interact with itself
    #     self.mask = torch.ones([nsamples,nparticles,nparticles,self.ngrids,dim],device=mydevice.get())
    #     dia = torch.diagonal(self.mask,dim1=1,dim2=2) # [nsamples, ngrids, dim, nparticles]
    #     dia.fill_(0.0)

    # ===================================================
    def prepare_q_feature_input(self, q_list, l_list):  # make dqdp for n particles
        """Function prepare_q_feature_input.
        
        Parameters
        ----------
        q_list : Any
            Particle positions tensor of shape (nsamples, nparticles, dim).
        l_list : Any
            Periodic box lengths tensor, broadcastable to positions.
        
        Returns
        -------
        Any
            Phi feature tensor for positions.
        """
        ret = self.phi_features(q_list,l_list)
        # print('phi feature shape',ret.shape) # 20250803: print shape
        return ret
    # ===================================================
    def prepare_p_feature_input(self, q_list, p_list, l_list):  # make dqdp for n particles
        """Function prepare_p_feature_input.
        
        Parameters
        ----------
        q_list : Any
            Particle positions tensor of shape (nsamples, nparticles, dim).
        p_list : Any
            Particle momenta/velocities tensor of shape (nsamples, nparticles, dim).
        l_list : Any
            Periodic box lengths tensor, broadcastable to positions.
        
        Returns
        -------
        Any
            Psi feature tensor for momenta.
        """
        ret = self.psi_features(q_list,p_list,l_list)
        # print('psi feature shape',ret.shape) # 20250803: print shape
        return ret

