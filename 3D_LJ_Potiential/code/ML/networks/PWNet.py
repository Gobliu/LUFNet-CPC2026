import torch.nn as nn
import torch

class PWNet(nn.Module):

    # input is torch.cat(dq_sq, dp_sq)

    """Class PWNet.
    
    Notes
    -----
    Pairwise network mapping distance features to vector contributions.
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

        hidden_nodes = nnodes
        h1 = hidden_nodes
        #h1 = max(hidden_nodes,input_dim)
        h2 = hidden_nodes
        h3 = hidden_nodes
        h4 = hidden_nodes
        h5 = output_dim
        fc1 = nn.Linear(input_dim,h1,bias=True)
        fc2 = nn.Linear(h1,h2,bias=True)
        fc3 = nn.Linear(h2,h3,bias=True)
        fc4 = nn.Linear(h3,h4,bias=True)
        fc5 = nn.Linear(h4,h5,bias=True)

        self.output_dim = output_dim

        self.layers = nn.ModuleList([fc1,fc2,fc3,fc4,fc5])

        if init_weights == 'tanh':
            self.layers.apply(self.init_weights_tanh)
        else:
            self.layers.apply(self.init_weights_relu)

        self.inv_max_force = 1./10.0
        #self.inv_max_force = 1./2.0
        self.inv_max_expon = 3
        

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

    def relu(self,x):
        """Function relu.
        
        Parameters
        ----------
        x : Any
            Input tensor.
        
        Returns
        -------
        Any
            Elementwise ReLU activation output.
        """
        return torch.relu(x)

    def tanh(self,x):
        """Function tanh.
        
        Parameters
        ----------
        x : Any
            Input tensor.
        
        Returns
        -------
        Any
            Elementwise tanh activation output.
        """
        return torch.tanh(x)

    def factor(self,dq):
        """Function factor.
        
        Parameters
        ----------
        dq : Any
            Distance or displacement magnitude used for scaling.
        
        Returns
        -------
        Any
            Distance-based scaling factor.
        """
        return 1.0/( dq**self.inv_max_expon + self.inv_max_force )

    def forward(self,x):
        """Function forward.
        
        Parameters
        ----------
        x : Any
            Input tensor.
        
        Returns
        -------
        Any
            Pairwise output tensor scaled by distance factor.
        """
        dq=x
        # pwnet input:[nsamples * nparticles * nparticles * ngrids, 1]
        for m in self.layers:
            x = m(x)
            if m != self.layers[-1]:
                x = self.relu(x)
            else:
                x = self.tanh(x) #;print('pw layer', m, x)
        w = self.factor(dq)
        # pwnet output: x shape [nsamples*nparticles*nparticles, 2]
        # pwnet output for mbnet input : x shape [nsamples * nparticles * nparticles * ngrids, 2]
        return x*w

