import torch.nn as nn

class ReadoutStep(nn.Module):
    """Readout network applied to per-particle embeddings.

    Args:
        net (nn.Module): Network that maps embeddings to outputs (e.g., updates).
    """

    def __init__(self, net):
        """Initialize the readout wrapper."""
        super().__init__()
        self.net = net

    # take in output of single_particle
    # and output to update_step
    # x.shape=[nsample,nparticle,embed_dim]
    # output shape=[nsample,nparticle,dim]
    def eval(self, x):
        """Compute per-particle readout.

        Args:
            x (torch.Tensor): Embeddings of shape (nsample, nparticle, embed_dim).

        Returns:
            torch.Tensor: Readout tensor of shape (nsample, nparticle, dim).
        """
        nsample,nparticle,embed_dim = x.shape
        x = x.reshape(nsample*nparticle,embed_dim)
        y = self.net(x)
        return y.reshape(nsample,nparticle,-1)
