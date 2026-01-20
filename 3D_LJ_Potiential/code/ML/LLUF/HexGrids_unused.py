import torch
import numpy as np
from utils.mydevice import mydevice
from utils.pbc import pbc
import matplotlib.pyplot as plt

class HexGrids:

    # b_list = list of grid distance
    # a_list = list of grid off set angle
    """Class HexGrids.
    
    Notes
    -----
    Generates 2D hexagonal grid points around each particle (unused in 3D).
    """
    def __init__(self,b_list,a_list):
        """Function __init__.
        
        Parameters
        ----------
        b_list : Any
            List of grid radii or lattice constants for grid layers.
        a_list : Any
            List of angular offsets (radians) for grid layers.
        """
        self.b = b_list
        nlayers = len(b_list)
        grids = []

        for b,a in zip(b_list,a_list):
            grids.append(self.hex_grids_list(b,a))
        # now combine all the grids
        grids_ncenter = torch.cat(grids,dim=0)
        # grids_ncenter.shape is [6*nlayers, 2]
        self.all_grids = mydevice.load(grids_ncenter)
        self.ngrids = nlayers*6

    # ===================================================
    # b = grid distance
    # a = angle offset
    def hex_grids_list(self,b,a):

        """Function hex_grids_list.
        
        Parameters
        ----------
        b : Any
            Grid radius or lattice constant.
        a : Any
            Angular offset (radians).
        
        Returns
        -------
        Any
            Tensor of 6 grid points around the origin.
        """
        dt = 2*np.pi/6.0
        grids = []
        for g in range(6):
            theta = g*dt+a
            bcost = b*np.cos(theta)
            bsint = b*np.sin(theta)
            grids.append([bcost,bsint])
        grids_ncenter = torch.tensor(grids)
        # grids_ncenter.shape is [6, 2]
        return grids_ncenter
    # ===================================================
    # for each particle, make the grid points around it
    # return all the grid points of all particles
    #
    def __call__(self,q,l_list):
        '''make_grids function to shift 6 grids points at (0,0) to each particle position as center'''

        l_list = torch.unsqueeze(l_list, dim=2)
        # l_list.shape is [nsamples, nparticles, 1, DIM]

        l_list = l_list.repeat_interleave(self.all_grids.shape[0], dim=2)
        # l_list.shape is [nsamples, nparticles, ngrids, DIM]

        q_list = torch.unsqueeze(q, dim=2)
        # q_list.shape is [nsamples, nparticles, 1, DIM=(x coord, y coord)]

        grids_ncenter = self.all_grids + q_list  # broadcast
        # all_grids.shape = [6*nlayers,2] = [ngrids,2]
        # grids_ncenter.shape is [ngrids, 2] + [nsamples, nparticles, 1, DIM] => [nsamples, nparticles, ngrids, DIM=2]

        #self.show_grids_nparticles(q, grids_ncenter, l_list[0,0,0], 'before')
        pbc(grids_ncenter, l_list)  # pbc - for grids
        #self.show_grids_nparticles(q, grids_ncenter,l_list[0,0,0], 'before')

        grids_ncenter = grids_ncenter.view(-1, q.shape[1] * self.all_grids.shape[0], q.shape[2])
        # shape is [nsamples, nparticles * ngrids, DIM=(x,y)]
        return grids_ncenter
    # ===================================================

    def show_grids_nparticles(self, q_list, uli_list, boxsize, name):

        """Function show_grids_nparticles.
        
        Parameters
        ----------
        q_list : Any
            Particle positions tensor of shape (nsamples, nparticles, dim).
        uli_list : Any
            Grid point positions tensor for all particles.
        boxsize : Any
            Simulation box size per dimension.
        name : Any
            Plot label or tag.
        
        Returns
        -------
        None
            None
        """
        bs = boxsize.detach().numpy()

        for i in range(1):  # show two samples

            plt.title('{} pbc....'.format(name))
            plt.xlim(-bs[0] / 2, bs[0] / 2)
            plt.ylim(-bs[1] / 2, bs[1] / 2)
            plt.plot(uli_list[i, :, :, 0].detach().numpy(), uli_list[i, :, :, 1].detach().numpy(), marker='.', color='k',
                     linestyle='none', markersize=12)
            plt.plot(q_list[i, :, 0].detach().numpy(), q_list[i, :, 1].detach().numpy(), marker='x', color='r',
                     linestyle='none', markersize=12)
            plt.show()
            plt.close()




