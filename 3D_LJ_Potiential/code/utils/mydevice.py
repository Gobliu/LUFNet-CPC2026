# make this into a singleton
import torch

# copyright: code created by Lee Hwee Kuan 04September2021
# 
# !! please test code, code not tested with cuda yet
#

class mydevice(object):

    """Class mydevice.
    
    Notes
    -----
    TODO: Add class details.
    """
    __instance = None

    def __new__(cls, *args, **kwargs):
        """Function __new__.
        
        Parameters
        ----------
        *args : Any
            TODO: Describe *args.
        **kwargs : Any
            TODO: Describe **kwargs.
        
        Returns
        -------
        Any
            TODO: Describe return value.
        """
        if not mydevice.__instance:
            mydevice.__instance = object.__new__(cls)
        return mydevice.__instance

    def __init__(self):
        """Function __init__.
        """
        use_cuda = torch.cuda.is_available()
        mydevice.__instance.value = torch.device("cuda" if use_cuda else "cpu")
        print('device singleton constructed for ',mydevice.__instance.value)
        
    @staticmethod
    def load(x):
        """Function load.
        
        Parameters
        ----------
        x : Any
            TODO: Describe x.
        
        Returns
        -------
        Any
            TODO: Describe return value.
        """
        return x.to(mydevice.__instance.value)

    @staticmethod
    def get():
        """Function get.
        
        Returns
        -------
        Any
            TODO: Describe return value.
        """
        return mydevice.__instance.value
    
    @staticmethod
    def device_name():
        """Function device_name.
        
        Returns
        -------
        None
            TODO: Describe return value.
        """
        print('device singleton constructed for ',mydevice.__instance.value)
# ================================================


def verify_device(specified_device):

    """Function verify_device.
    
    Parameters
    ----------
    specified_device : Any
        TODO: Describe specified_device.
    
    Returns
    -------
    None
        TODO: Describe return value.
    """
    target = torch.device(specified_device)

    if mydevice.get() != target: 
        print('WARNING: requested device unavailable', target)

    # this is to test if loading to GPU will throw an error
    a = torch.tensor([1])
    mydevice.load(a)


