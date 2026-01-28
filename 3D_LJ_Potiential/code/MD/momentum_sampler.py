import torch
import numpy as np
from parameters.MC_parameters import MC_parameters

class momentum_sampler:
    """Sample particle momenta consistent with a Boltzmann distribution.

    Args:
        nsamples (int): Number of samples (trajectories) to generate.
    """

    _obj_count = 0

    def __init__(self, nsamples):
        """Initialize sampler state.

        Args:
            nsamples (int): Number of samples (trajectories) to generate.
        """

        momentum_sampler._obj_count += 1
        assert (momentum_sampler._obj_count == 1),type(self).__name__ + " has more than one object"

        self.vel = np.zeros((nsamples, MC_parameters.nparticle, MC_parameters.DIM))
        print('momentum_sampler initialized: nsamples ',nsamples)

    def momentum_samples(self, mass=1):
        """Generate momentum samples with Maxwell-Boltzmann statistics.

        Args:
            mass (float): Particle mass used to scale momenta.

        Returns:
            torch.Tensor: Momentum samples, shape [nsamples, nparticle, DIM].
        """
        # 'generate': 'maxwell'
        sigma = np.sqrt( MC_parameters.temperature )  # sqrt(kT/m)
        self.vel = np.random.normal(0, 1, (self.vel.shape)) * sigma # make sure shape correct
        momentum = torch.tensor(self.vel) * mass

        return momentum
