import torch
import torch.nn as nn
#from utils.mydevice import mydevice

from ML.LLUF.PhiFeatures import PhiFeatures
from ML.LLUF.PsiFeatures import PsiFeatures

from utils.pbc import pbc
from utils.pbc import _delta_state
# import matplotlib.pyplot as plt # 20250809 nscc no module ...

class PrepareData(nn.Module):
    """Prepare LLUF grid-based features for q and p trajectories.

    Args:
    net (nn.Module): Network used for phi feature extraction.
    grid_object: Grid generator object (e.g., SingleGrid, HexGrids).
    """

    # grid_object is the object to make grid center at every particle
    # e.g. HexGrids for making multiple layers of hexagonal grids
    # e.g. SingleGrid for making only one grid at each particle center
    def __init__(self, net, grid_object):
        """Initialize feature preparation modules."""
        super().__init__()
        # net : mb4pw -- use to extract features for position variable of grid point
        self.net = net
        self.grid_object = grid_object
        self.phi_features = PhiFeatures(self.grid_object,net)
        self.psi_features = PsiFeatures(self.grid_object)

    # ===================================================
    def cat_qp(self,q_input_list,p_input_list):
        """Concatenate q and p feature lists along the channel dimension.

        Args:
        q_input_list (list[torch.Tensor]): Position features across time.
        p_input_list (list[torch.Tensor]): Momentum features across time.

        Returns:
        torch.Tensor: Concatenated tensor of shape
        (nsamples, nparticles, traj_len, ngrids * DIM * (q,p)).
        """
        # # q_input_list : mb net [phi0, phi1, phi2, ...] # phi0=(phi0x,phi0y)
        # # phi shape [nsamples, nparticles, ngrids * DIM]
        # # p_input_list : mb net [pi0, pi1, pi2, ...] # pi0=(pi0x,pi0y)
        # # return mbnet qp_cat phi, pi along time
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
        """Generate phi features from positions.

        Args:
        q_list (list[torch.Tensor]): Position trajectory list.
        l_list (list[torch.Tensor]): Box size list per sample.

        Returns:
        torch.Tensor: Phi features for q.
        """
        ret = self.phi_features(q_list,l_list)
        # print('phi feature shape',ret.shape) # 20250803: print shape
        return ret
    # ===================================================
    def prepare_p_feature_input(self, q_list, p_list, l_list):  # make dqdp for n particles
        """Generate psi features from positions and momenta.

        Args:
        q_list (list[torch.Tensor]): Position trajectory list.
        p_list (list[torch.Tensor]): Momentum trajectory list.
        l_list (list[torch.Tensor]): Box size list per sample.

        Returns:
        torch.Tensor: Psi features for p.
        """
        ret = self.psi_features(q_list,p_list,l_list)
        # print('psi feature shape',ret.shape) # 20250803: print shape
        return ret

