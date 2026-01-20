
   
def total_momentum(p_list):

    # p_list shape [nsamples,nparticles,dim]
    """Function total_momentum.
    
    Parameters
    ----------
    p_list : Any
        TODO: Describe p_list.
    
    Returns
    -------
    Any
        TODO: Describe return value.
    """
    p_total = torch.sum(p_list,dim=1)
    return p_total


