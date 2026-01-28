# make this into a singleton
import torch

# copyright: code created by Lee Hwee Kuan 04September2021
# 
# !! please test code, code not tested with cuda yet
#

class mydevice(object):

    """mydevice class."""
    __instance = None

    def __new__(cls, *args, **kwargs):
        """Create or return the singleton instance."""
        if not mydevice.__instance:
            mydevice.__instance = object.__new__(cls)
        return mydevice.__instance

    def __init__(self, force_cuda=True):
        """Initialize device selection.

        Args:
            force_cuda (bool): Require CUDA; raise if unavailable.
        """
        use_cuda = torch.cuda.is_available()
        if force_cuda and not use_cuda:
            raise RuntimeError(
                "CUDA is required but not available. Set force_cuda: false in main_config.yaml/maintest_config.yaml "
                "to allow CPU fallback."
            )
        mydevice.__instance.value = torch.device("cuda" if use_cuda else "cpu")
        print('device singleton constructed for ',mydevice.__instance.value)
        
    @staticmethod
    def load(x):
        """Move tensor to the configured device.

        Args:
            x (torch.Tensor): Tensor to move.

        Returns:
            torch.Tensor: Tensor on the configured device.
        """
        return x.to(mydevice.__instance.value)

    @staticmethod
    def get():
        """Get the configured torch.device."""
        return mydevice.__instance.value
    
    @staticmethod
    def device_name():
        """Print the configured device."""
        print('device singleton constructed for ',mydevice.__instance.value)
# ================================================


def verify_device(specified_device):

    """Verify the active device matches a requested device string.

    Args:
        specified_device (str): Device string (e.g., "cpu", "cuda").
    """
    target = torch.device(specified_device)

    if mydevice.get() != target: 
        print('WARNING: requested device unavailable', target)

    # this is to test if loading to GPU will throw an error
    a = torch.tensor([1])
    mydevice.load(a)
