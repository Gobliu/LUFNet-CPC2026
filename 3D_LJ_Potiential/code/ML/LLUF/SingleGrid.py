import torch
from utils.mydevice import mydevice

class SingleGrid:
    """Generate a single grid point at each particle center."""

    def __init__(self):
        """Initialize a one-point grid (center only)."""
        # grids_ncenter.shape is [6*nlayers, 2]
        grids_ncenter = torch.tensor([[0,0]]) # shape [1,2]
        self.all_grids = mydevice.load(grids_ncenter)
        self.ngrids = 1

    # ===================================================
    # for each particle, make the grid points around it
    # return all the grid points of all particles
    #
    def __call__(self,q,l_list):
        """Shift a single grid point to each particle position.

        Args:
            q (torch.Tensor): Positions of shape (nsamples, nparticles, dim).
            l_list (torch.Tensor): Box sizes of shape (nsamples, nparticles, dim).

        Returns:
            torch.Tensor: Grid centers of shape (nsamples, nparticles * ngrids, dim).
        """

        l_list = torch.unsqueeze(l_list, dim=2)
        # l_list.shape is [nsamples, nparticles, 1, DIM]

        l_list = l_list.repeat_interleave(self.all_grids.shape[0], dim=2)

        q_list = torch.unsqueeze(q, dim=2)
        # q_list.shape is [nsamples, nparticles, 1, DIM=(x coord, y coord)]

        grids_ncenter = self.all_grids + q_list  # broadcast
        # all_grids.shape = [1,2]
        # grids_ncenter.shape is [1, 2] + [nsamples, nparticles, 1, DIM] => [nsamples, nparticles, 1, DIM=2]

        # dont need pbc: pbc(grids_ncenter, l_list)  # pbc - for grids
        # self.show_grids_nparticles(q, grids_ncenter,l_list[0,0,0])

        grids_ncenter = grids_ncenter.view(-1, q.shape[1] * self.all_grids.shape[0], q.shape[2])
        # shape is [nsamples, nparticles * ngrids, DIM=(x,y)]
        return grids_ncenter
    # ===================================================




