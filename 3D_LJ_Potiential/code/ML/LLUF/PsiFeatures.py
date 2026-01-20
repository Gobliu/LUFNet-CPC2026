import torch
from utils.mydevice import mydevice
from utils.pbc import pbc
class PsiFeatures:

    """Class PsiFeatures.
    
    Notes
    -----
    Computes momentum-based (psi) features on grid points using inverse-distance weights.
    """
    def __init__(self,grid_object): #b_list,a_list):
        """Function __init__.
        
        Parameters
        ----------
        grid_object : Any
            Grid generator that returns grid points per particle.
        """
        self.grid_object = grid_object

        self.dim = grid_object.dim # 20250807

    # ===================================================
    def __call__(self, q_list, p_list, l_list):  # make dqdp for n particles
        """Function __call__.
        
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
            Psi feature tensor of shape (nsamples, nparticles, ngrids * dim).
        """
        uli_list = self.grid_object(q_list, l_list)  # position at grids
        # shape is [nsamples, nparticles * ngrids, DIM=(x,y,z)] # 20250807

        if self.grid_object.ngrids > 1:
            p_var = self.gen_pvar(q_list, p_list, uli_list, l_list, self.grid_object.ngrids)  # velocity fields - no change
            # v_fields.shape [nsamples, npartice, ngrids * DIM]
        else:
            p_var = mydevice.load(torch.zeros(p_list.shape))
            # shape is [ nsamples, nparticles, DIM ]

        return p_var


    # ===================================================
    def gen_pvar(self, q, p, uli_list, l_list, ngrids):  # velocity fields

        """Function gen_pvar.
        
        Parameters
        ----------
        q : Any
            Positions tensor.
        p : Any
            Momenta/velocity tensor.
        uli_list : Any
            Grid point positions tensor for all particles.
        l_list : Any
            Periodic box lengths tensor, broadcastable to positions.
        ngrids : Any
            Number of grid points per particle.
        
        Returns
        -------
        Any
            Relative momentum features per grid point.
        """
        nsamples, nparticles, DIM = p.shape

        l_list = torch.unsqueeze(l_list, dim=2)
        # l_list.shape is [nsamples, nparticles, 1, DIM]

        l_list = l_list.repeat_interleave(uli_list.shape[1], dim=2)
        # l_list.shape is [nsamples, nparticles, nparticles * ngrids, DIM]

        _, d_sq = self.dpair_pbc_sq(q, uli_list, l_list)
        # d_sq.shape is [nsamples, nparticles, nparticles * ngrids]

        # r^2 nearest distance weight
        weights = 1 / (d_sq + 1e-10)
        # weights.shape is [nsamples, nparticles, nparticles * ngrids]

        weights = torch.unsqueeze(weights, dim=-1)
        # w_thrsh.shape is [nsamples, nparticles, nparticles * ngrids, 1]

        p_list = torch.unsqueeze(p, dim=2)
        # p_list.shape is [nsamples, nparticles, 1, DIM]

        wp = weights * p_list
        # wp.shape [nsamples, nparticles, nparticles * ngrids, DIM]

        wp_nume = torch.sum(wp, dim=1)
        # wp_nume.shape [nsamples,  nparticles * ngrids, DIM]

        wp_deno = torch.sum(weights, dim=1)
        # wp_deno.shape is [nsamples, nparticles * ngrids, 1]

        p_ngrids = wp_nume / wp_deno
        # p_grids.shape [nsamples,  nparticles * ngrids, DIM]

        p_ngrids = p_ngrids.view(nsamples, nparticles, ngrids, DIM)
        # p_ngrids.shape [nsamples, npartice, grids, DIM]
        # p_list.shape is [nsamples, nparticles, 1, DIM]

        # relative momentum : center of particles
        relative_p = p_ngrids - p_list
        # relative_p.shape [nsamples, npartice, grids, DIM]

        gen_pvar = relative_p.view(nsamples, nparticles, ngrids * DIM)
        # relative_p.shape [nsamples, npartice, grids*DIM]
        return gen_pvar


    # ===================================================
    def dpair_pbc_sq(self, q, uli_list, l_list):  #

        # all list dimensionless
        """Function dpair_pbc_sq.
        
        Parameters
        ----------
        q : Any
            Positions tensor.
        uli_list : Any
            Grid point positions tensor for all particles.
        l_list : Any
            Periodic box lengths tensor, broadcastable to positions.
        
        Returns
        -------
        Any
            Tuple of (pairwise displacements, squared distances).
        """
        q_state = torch.unsqueeze(q, dim=2)
        # shape is [nsamples, nparticles, 1, DIM=(x,y)]

        uli_list = torch.unsqueeze(uli_list, dim=1)
        # shape is [nsamples, 1, nparticles * ngrids, DIM=(x,y)]

        paired_grid_q = uli_list - q_state
        # paired_grid_q.shape is [nsamples, nparticles, nparticles * ngrids, DIM]

        pbc(paired_grid_q, l_list)

        dd = torch.sum(paired_grid_q * paired_grid_q, dim=-1)
        # dd.shape is [nsamples, nparticles, nparticles * ngrids]

        return paired_grid_q, dd

