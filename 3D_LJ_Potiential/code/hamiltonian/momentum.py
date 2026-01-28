
   
def total_momentum(p_list):

    # p_list shape [nsamples,nparticles,dim]
    """Compute total momentum per sample.

    Args:
        p_list (torch.Tensor): Momenta, shape [nsamples, nparticles, dim].

    Returns:
        torch.Tensor: Total momentum, shape [nsamples, dim].
    """
    p_total = torch.sum(p_list,dim=1)
    return p_total

