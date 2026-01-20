import torch.nn as nn

class ReadoutStep(nn.Module):

    """Class ReadoutStep.
    
    Notes
    -----
    Maps embeddings to per-particle update vectors.
    """
    def __init__(self, net):
        """Function __init__.
        
        Parameters
        ----------
        net : Any
            PyTorch module used for feature extraction or updates.
        """
        super().__init__()
        self.net = net

    # take in output of single_particle
    # and output to update_step
    # x.shape=[nsample,nparticle,embed_dim]
    # output shape=[nsample,nparticle,dim]
    def eval(self, x):
        """Function eval.
        
        Parameters
        ----------
        x : Any
            Input tensor.
        
        Returns
        -------
        Any
            Per-particle update tensor.
        """
        nsample,nparticle,embed_dim = x.shape
        x = x.reshape(nsample*nparticle,embed_dim)
        y = self.net(x)
        return y.reshape(nsample,nparticle,-1)
