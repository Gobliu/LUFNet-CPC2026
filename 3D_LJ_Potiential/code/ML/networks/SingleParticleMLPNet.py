import torch.nn as nn
import torch

class SingleParticleMLPNet(nn.Module):

    """Class SingleParticleMLPNet.
    
    Notes
    -----
    MLP encoder for per-particle trajectory features.
    """
    def __init__(self,input_dim,output_dim,nnodes,init_weights,p):
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
        p : Any
            Momenta/velocity tensor.
        """
        print('!!!!! single par mlp_net', input_dim, output_dim, nnodes, init_weights, p)
        super().__init__()

        hidden_nodes = nnodes
        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_nodes),
                                 nn.Dropout(p),
                                 nn.Linear(hidden_nodes, hidden_nodes),
                                 nn.Dropout(p),
                                 nn.Linear(hidden_nodes, hidden_nodes),
                                 nn.Dropout(p),
                                 nn.Linear(hidden_nodes, output_dim),
                                 nn.Tanh())
                                 #nn.ReLU())

        if init_weights == 'tanh':
            self.mlp.apply(self.init_weights_tanh)
        else:
            self.mlp.apply(self.init_weights_relu)

    def init_weights_tanh(self,m): # m is layer that is nn.Linear
        """Function init_weights_tanh.
        
        Parameters
        ----------
        m : Any
            Layer module passed by PyTorch initialization hook.
        
        Returns
        -------
        None
            None
        """
        if type(m) == nn.Linear:
            # set the xavier_gain neither too much bigger than 1, nor too much less than 1
            # recommended gain value for the given nonlinearity function
            # tanh gain=5/3
            #print('init weight; tanh......')
            nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('tanh'))
            #m.weight.data = m.weight.data*1e-1
            m.bias.data.fill_(0.0)

    def init_weights_relu(self,m): # m is layer that is nn.Linear
        """Function init_weights_relu.
        
        Parameters
        ----------
        m : Any
            Layer module passed by PyTorch initialization hook.
        
        Returns
        -------
        None
            None
        """
        if type(m) == nn.Linear:
            # set the xavier_gain neither too much bigger than 1, nor too much less than 1
            # recommended gain value for the given nonlinearity function
            # tanh gain=5/3
            #print('init weight; relu.......')
            nn.init.kaiming_normal_(m.weight,nonlinearity='relu')
            m.bias.data.fill_(0.0)

    def forward(self,x):
        # input x.shape [nsample * nparticle, traj_len, ngrid * DIM * (q,p)]
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
        x = x.reshape(x.size(0), -1)
        x = self.mlp(x)
        # x shape [nsamples * nparticles, 2]
        return x

