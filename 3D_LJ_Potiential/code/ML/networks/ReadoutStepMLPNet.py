import torch
import torch.nn as nn

class ReadoutStepMLPNet(nn.Module):

    """Class ReadoutStepMLPNet.
    
    Notes
    -----
    MLP readout head that maps embeddings to update vectors.
    """
    def __init__(self,input_dim,output_dim,nnodes,p,readout=True):
        """Function __init__.
        
        Parameters
        ----------
        input_dim : Any
            Input feature dimension.
        output_dim : Any
            Output feature dimension.
        nnodes : Any
            Hidden layer width.
        p : Any
            Momenta/velocity tensor.
        readout : Any
            Whether to apply the readout MLP (False uses identity).
        """
        print('!!!!! update step mlp_net', input_dim, output_dim, nnodes, p)
        super().__init__()

        hidden_nodes = nnodes
        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_nodes),
                                 nn.Dropout(p),
                                 nn.Linear(hidden_nodes, hidden_nodes),
                                 nn.Dropout(p),
                                 nn.Linear(hidden_nodes, output_dim),
                                 nn.Tanh()) if readout else nn.Identity()


    def forward(self,x):
        # x shape [nsamples * nparticles, embed_dim]
        """Function forward.
        
        Parameters
        ----------
        x : Any
            Input tensor.
        
        Returns
        -------
        Any
            Output tensor of shape (batch, output_dim).
        """
        x = self.mlp(x)
        # x shape [nsamples * nparticles, 2]
        return x

