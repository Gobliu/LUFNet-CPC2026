import torch
import torch.nn as nn

class ReadoutStepMLPNet(nn.Module):
    """MLP readout head for per-particle updates.

    Args:
        input_dim (int): Input feature dimension.
        output_dim (int): Output dimension.
        nnodes (int): Hidden layer width.
        p (float): Dropout probability.
        readout (bool, optional): If False, use identity mapping.
    """

    def __init__(self,input_dim,output_dim,nnodes,p,readout=True):
        """Initialize the readout MLP."""
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
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (nsamples * nparticles, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (nsamples * nparticles, output_dim).
        """
        # x shape [nsamples * nparticles, embed_dim]
        x = self.mlp(x)
        # x shape [nsamples * nparticles, 2]
        return x

