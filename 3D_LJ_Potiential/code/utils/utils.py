import torch
from inspect        import currentframe, getframeinfo
from utils.mydevice import mydevice

# ===================================================
def assert_nan(x):
    """Raise and exit if a tensor contains NaNs.

    Args:
        x (torch.Tensor): Tensor to check.
    """
    cframe = currentframe().f_back
    filename = getframeinfo(cframe).filename
    lineno = cframe.f_lineno
    masknan = torch.isnan(x)
    if masknan.any() == True:
        print(filename,' line ',lineno,' has nan')
        quit()
 
# ===================================================
def print_compute_tree(name,node):
    """Render and save a compute graph.

    Args:
        name (str): Output filename prefix.
        node: Autograd graph node.
    """
    dot = make_dot(node)
    #print(dot)
    dot.render(name)

# ===================================================
def check_data(loader,data_set,tau_traj_len,tau_long,tau_short,nitr,append_strike):

    """Run data consistency checks using MD trajectories.

    Args:
        loader: Data loader with train_loader attribute.
        data_set: Dataset with check_md_trajectory method.
        tau_traj_len (float): Trajectory length in time units.
        tau_long (float): Long time step size.
        tau_short (float): Short time step size.
        nitr (int): Number of iterations for trajectory checks.
        append_strike (int): Stride for checking.
    """
    label_idx = int(tau_traj_len//tau_long)
    for qpl_input,qpl_label in loader.train_loader:

        q_traj,p_traj,q_label,p_label,l_init = pack_data(qpl_input,qpl_label)
        data_set.check_md_trajectory(q_traj,p_traj,q_label,p_label,l_init,label_idx,
                                     tau_short,nitr,append_strike)


# ===================================================

def pack_data(qpl_input, qpl_label):

    """Unpack input/label tensors and move to device.

    Args:
        qpl_input (torch.Tensor): Input tensor [batch, 3, traj, nparticles, dim].
        qpl_label (torch.Tensor): Label tensor [batch, 2, traj, nparticles, dim].

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        (q_traj, p_traj, q_label, p_label, l_init).
    """
    q_traj = qpl_input[:,0,:,:,:].clone().detach() #.requires_grad_()
    q_traj = q_traj.permute(1,0,2,3)
    # shape [trajectory,nsamples,nparticles,dim]
    p_traj = qpl_input[:,1,:,:,:].clone().detach() #.requires_grad_()
    p_traj = p_traj.permute(1,0,2,3)
    l_init = qpl_input[:,2,0,:,:].clone().detach() #.requires_grad_()
    # l_init.shape is [nsamples,nparticles,DIM]
    q_label = qpl_label[:,0,:,:,:].clone().detach() #.requires_grad_()
    p_label = qpl_label[:,1,:,:,:].clone().detach() #.requires_grad_()

    q_traj = mydevice.load(q_traj)
    p_traj = mydevice.load(p_traj)
    l_init = mydevice.load(l_init)
    q_label = mydevice.load(q_label)
    p_label = mydevice.load(p_label)

    return q_traj,p_traj,q_label,p_label,l_init

# ===================================================

def print_dict(name,thisdict):
    """Print a dictionary with a labeled header.

    Args:
        name (str): Label to print.
        thisdict (dict): Dictionary to print.
    """
    print(name,'dict ============== ')
    for key,value in thisdict.items(): print(key,':',value)




