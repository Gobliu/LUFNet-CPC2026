import torch
from inspect        import currentframe, getframeinfo
from utils.mydevice import mydevice

# ===================================================
def assert_nan(x):
    """Function assert_nan.
    
    Parameters
    ----------
    x : Any
        TODO: Describe x.
    
    Returns
    -------
    None
        TODO: Describe return value.
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
    """Function print_compute_tree.
    
    Parameters
    ----------
    name : Any
        TODO: Describe name.
    node : Any
        TODO: Describe node.
    
    Returns
    -------
    None
        TODO: Describe return value.
    """
    dot = make_dot(node)
    #print(dot)
    dot.render(name)

# ===================================================
def check_data(loader,data_set,tau_traj_len,tau_long,tau_short,nitr,append_strike):

    """Function check_data.
    
    Parameters
    ----------
    loader : Any
        TODO: Describe loader.
    data_set : Any
        TODO: Describe data_set.
    tau_traj_len : Any
        TODO: Describe tau_traj_len.
    tau_long : Any
        TODO: Describe tau_long.
    tau_short : Any
        TODO: Describe tau_short.
    nitr : Any
        TODO: Describe nitr.
    append_strike : Any
        TODO: Describe append_strike.
    
    Returns
    -------
    None
        TODO: Describe return value.
    """
    label_idx = int(tau_traj_len//tau_long)
    for qpl_input,qpl_label in loader.train_loader:

        q_traj,p_traj,q_label,p_label,l_init = pack_data(qpl_input,qpl_label)
        data_set.check_md_trajectory(q_traj,p_traj,q_label,p_label,l_init,label_idx,
                                     tau_short,nitr,append_strike)


# ===================================================

def pack_data(qpl_input, qpl_label):

    """Function pack_data.
    
    Parameters
    ----------
    qpl_input : Any
        TODO: Describe qpl_input.
    qpl_label : Any
        TODO: Describe qpl_label.
    
    Returns
    -------
    Any
        TODO: Describe return value.
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
    """Function print_dict.
    
    Parameters
    ----------
    name : Any
        TODO: Describe name.
    thisdict : Any
        TODO: Describe thisdict.
    
    Returns
    -------
    None
        TODO: Describe return value.
    """
    print(name,'dict ============== ')
    for key,value in thisdict.items(): print(key,':',value)





