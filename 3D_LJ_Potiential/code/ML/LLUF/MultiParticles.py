import torch
import torch.nn as nn

class MultiParticles(nn.Module):

    """Class MultiParticles.
    
    Notes
    -----
    Applies a multi-particle network (e.g., GNN) to particle embeddings.
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
    # q_previous.shape=[nsample,nparticle,dim]
    # output shape=[nsample,nparticle,embed_dim]
    def eval(self, x, q_previous):
        """Function eval.
        
        Parameters
        ----------
        x : Any
            Input tensor.
        q_previous : Any
            Previous-step positions tensor.
        
        Returns
        -------
        Any
            Updated particle embeddings tensor.
        """
        nsample, nparticle, embed_dim = x.shape
        x = x.reshape(nsample * nparticle, embed_dim)
        y = self.net(x, q_previous)
        return y.reshape(nsample, nparticle, -1)

