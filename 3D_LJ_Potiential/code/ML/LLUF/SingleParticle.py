import torch
import torch.nn as nn

class SingleParticle(nn.Module):
    """Apply a per-particle network to trajectory inputs.

    Args:
    net (nn.Module): Network applied to each particle trajectory.
    """

    # two networks first one for updating p
    # second one for updating q
    # networks are all mb-net, no more pw-net
    def __init__(self, net):
        """Initialize the wrapper."""
        super().__init__()
        self.net = net

    # take in prepared data and output feature for
    # multi-particle layers
    # input x.shape [nsample, nparticle, traj_len, ngrid * DIM * (q,p)]
    # output.shape = [nsample, nparticle, embed_dim]
    def eval(self,x):
        """Compute per-particle embeddings.

        Args:
        x (torch.Tensor): Input tensor of shape
        (nsample, nparticle, traj_len, ngrid * DIM * (q,p)).

        Returns:
        torch.Tensor: Output tensor of shape (nsample, nparticle, embed_dim).
        """
        nsample,nparticle,traj_len,_ = x.shape
        x = x.reshape(nsample*nparticle,traj_len,-1)
        output = self.net(x)
        return output.reshape(nsample, nparticle, -1)
