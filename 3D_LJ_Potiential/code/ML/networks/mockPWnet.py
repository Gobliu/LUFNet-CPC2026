import torch.nn as nn
import torch

class mockPWnet(nn.Module):

    # input is torch.cat(dq_sq, dp_sq)

    """Class mockPWnet.
    
    Notes
    -----
    Toy pairwise network for testing feature generation.
    """
    def __init__(self,input_dim,output_dim,nnodes,init_weights):
        """Function __init__.
        
        Parameters
        ----------
        input_dim : Any
            Input feature dimension.
        output_dim : Any
            Output feature dimension.
        nnodes : Any
            Hidden layer width.
        init_weights : Any
            Weight initialization scheme (e.g., "relu" or "tanh").
        """
        super().__init__()
        assert output_dim==2,'make output dim =2'

    # x.shape = [batch,1]
    def forward(self,x):
        """Function forward.
        
        Parameters
        ----------
        x : Any
            Input tensor.
        
        Returns
        -------
        Any
            Mock pairwise output tensor.
        """
        s1 = x*x + 3.2
        s2 = x*(x+1) - .2
        return torch.stack((s1,s2),dim=-1) # shape = [batch,2]
