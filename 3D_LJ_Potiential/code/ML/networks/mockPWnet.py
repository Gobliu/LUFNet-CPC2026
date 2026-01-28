import torch.nn as nn
import torch

class mockPWnet(nn.Module):
    """Simple mock pairwise network for testing.

    Args:
        input_dim (int): Input feature dimension (unused).
        output_dim (int): Output dimension; must be 2.
        nnodes (int): Hidden width (unused).
        init_weights (str): Initialization type (unused).
    """

    # input is torch.cat(dq_sq, dp_sq)

    def __init__(self,input_dim,output_dim,nnodes,init_weights):
        """Initialize the mock network."""
        super().__init__()
        assert output_dim==2,'make output dim =2'

    # x.shape = [batch,1]
    def forward(self,x):
        """Compute a deterministic two-channel transform.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, 1).

        Returns:
            torch.Tensor: Output tensor of shape (batch, 2).
        """
        s1 = x*x + 3.2
        s2 = x*(x+1) - .2
        return torch.stack((s1,s2),dim=-1) # shape = [batch,2]
