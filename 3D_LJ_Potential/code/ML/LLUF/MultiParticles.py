import torch
import torch.nn as nn

class MultiParticles(nn.Module):
    """Apply a multi-particle network to per-particle embeddings.

    Args:
    net (nn.Module): Network that consumes flattened embeddings and positions.
    """

    def __init__(self, net):
        """Initialize the wrapper."""
        super().__init__()
        self.net = net

    # take in output of single_particle
    # and output to update_step
    # x.shape=[nsample,nparticle,embed_dim]
    # q_previous.shape=[nsample,nparticle,dim]
    # output shape=[nsample,nparticle,embed_dim]
    def eval(self, x, q_previous):
        """Compute updated embeddings from particle features and positions.

        Args:
        x (torch.Tensor): Per-particle embeddings of shape
        (nsample, nparticle, embed_dim).
        q_previous (torch.Tensor): Previous positions of shape
        (nsample, nparticle, dim).

        Returns:
        torch.Tensor: Updated embeddings of shape (nsample, nparticle, embed_dim).
        """
        nsample, nparticle, embed_dim = x.shape
        x = x.reshape(nsample * nparticle, embed_dim)
        y = self.net(x, q_previous)
        return y.reshape(nsample, nparticle, -1)
