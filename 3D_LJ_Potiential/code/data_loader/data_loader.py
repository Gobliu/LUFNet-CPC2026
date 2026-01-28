import torch
from data_loader.data_io import data_io
from torch.utils.data import Dataset
from data_loader.check_load_data import check_load_data
from utils.seed import seed_worker

# qp_list.shape = [nsamples, (q,p,l)=3, trajectory=2 (input,label), nparticle, DIM]
# ===========================================================
class torch_dataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, filename, traj_len_idx, saved_pair_steps, label_idx): # 20250809
        """
        Args:
            filename (string): Numpy file for data and label
        """

        qpl_list,tau_short,_ = data_io.read_trajectory_qpl(filename) # returns qpl,tau_short,tau_long
        # qpl_list.shape = [nsamples, (q,p,boxsize)=3, trajectory, nparticles, DIM = 2 or 3]
        self.qpl_list_input   = qpl_list[:,:,0:traj_len_idx:saved_pair_steps,:,:] # 20250809 to adjust tau_long
        # qp_list_input.shape = [nsamples, (q,p,boxsize)=3, trajectory, nparticles, DIM = 2 or 3]
        self.qpl_list_label   = qpl_list[:,:,traj_len_idx:label_idx+saved_pair_steps:saved_pair_steps,:,:] # 20250809 to adjust tau_long
        # qp_list_label.shape is [nsamples,nparticles,DIM = 2 or 3]
        self.data_boxsize   =  qpl_list[:,2,0,:,:]
        #data_boxsize.shape is [nsamples,nparticles,DIM = 2 or 3]
        #self.data_tau_short = tau_short
        #self.data_tau_long  = float(tau_long)

        self.check_load = check_load_data(self.qpl_list_input,self.qpl_list_label)
        #self.check_load.check(tau_short)

    def __len__(self):
        ''' Denotes the total number of samples '''
        return self.qpl_list_input.shape[0]

    def __getitem__(self, idx):
        ''' Generates one sample of data '''
        # Select sample
        if idx >= self.__len__():
            raise ValueError('idx ' + str(idx) +' exceed length of data: ' + str(self.__len__()))
        return self.qpl_list_input[idx], self.qpl_list_label[idx] 

# ===========================================================
class my_data:
    """my_data class."""
    def __init__(self,train_filename,val_filename,test_filename,tau_long, window_sliding, saved_pair_steps,
                      tau_traj_len,train_pts=0,val_pts=0,test_pts=0): # 20250809

        # 20250809 to adjust tau_long
        """__init__ function.

Args:
    train_filename (str): Training dataset path.
    val_filename (str): Validation dataset path.
    test_filename (str): Test dataset path.
    tau_long (float): Long time step size.
    window_sliding (int): Window length for integration.
    saved_pair_steps (int): Pairing stride in saved trajectory.
    tau_traj_len (float): Trajectory length in time units.
    train_pts (int): Optional subsample size for training.
    val_pts (int): Optional subsample size for validation.
    test_pts (int): Optional subsample size for testing.
    """
        traj_len_index = round(tau_traj_len/tau_long) * saved_pair_steps
        label_index = int((traj_len_index - saved_pair_steps) + window_sliding * saved_pair_steps)
        print('tau_traj_len',tau_traj_len, 'tau_long', tau_long, 'saved_pair_steps', saved_pair_steps)
        print('load my data ... traj len index', traj_len_index, 'window sliding', window_sliding, 'label index', label_index)

        print("load train set .........")
        self.train_set = torch_dataset(train_filename, traj_len_index, saved_pair_steps, label_index)
        print("load valid set .........")
        self.val_set   = torch_dataset(val_filename, traj_len_index, saved_pair_steps, label_index)
        print("load test set .........")
        self.test_set  = torch_dataset(test_filename, traj_len_index, saved_pair_steps, label_index)

        # perform subsampling of data when specified
        # this is important when we need to perform quick debugging with
        # few data points
        if train_pts > 0:
            if train_pts > len(self.train_set):
                print('available ', len(self.train_set))
                raise ValueError("ERROR: request more than subspace set")
            self.train_set = self.sample(self.train_set, train_pts)

        if val_pts > 0:
            if val_pts > len(self.val_set):
                print('available ', len(self.val_set))
                raise ValueError("ERROR: request more than subspace set")
            self.val_set = self.sample(self.val_set, val_pts)

        if test_pts > 0:
            if test_pts > len(self.test_set):
                print('available ', len(self.test_set))
                raise ValueError("ERROR: request more than subspace set")
            self.test_set = self.sample(self.test_set, test_pts)

        print('my_data initialized : train_filename ',train_filename,' val_filename ',
               val_filename,' test_filename ',test_filename,' train_pts ',train_pts,
              ' val_pts ',val_pts,' test_pts ',test_pts)

    # ===========================================================
    def check_md_trajectory(self,q_init,p_init,q_final,p_final,l_list,neval,tau,nitr,append_strike):
        """check_md_trajectory function.

Args:
    q_init (torch.Tensor): Initial positions over trajectory.
    p_init (torch.Tensor): Initial momenta over trajectory.
    q_final (torch.Tensor): Final positions.
    p_final (torch.Tensor): Final momenta.
    l_list (torch.Tensor): Box sizes.
    neval (int): Label index to compare.
    tau (float): Time step.
    nitr (int): Number of steps.
    append_strike (int): Snapshot stride.

Returns:
    None
    """
        assert(self.train_set.check_load.md_trajectory(q_init,p_init,q_final,p_final,
                            l_list,neval,tau,nitr,append_strike )),'data_loader.py:82 error'
    # ===========================================================
    # sample data_set with num_pts of points
    # ===========================================================
    def sample(self, data_set, num_pts):

        # perform subsampling of data when specified
        # this is important when we need to perform quick debugging with
        # few data points
        """sample function.

Args:
    data_set (torch_dataset): Dataset to subsample.
    num_pts (int): Number of samples to keep.

Returns:
    torch_dataset: Subsampled dataset.
    """
        if num_pts > 0:
            if num_pts > len(data_set):
                print("error: request more than CIFAR10 set")
                print('available ',len(self.train_set))
                quit()

        data_set.qpl_list_input = data_set.qpl_list_input[:num_pts]
        data_set.qpl_list_label = data_set.qpl_list_label[:num_pts]

        return data_set

# ===========================================================
class data_loader:
    #  data loader upon this custom dataset
    """data_loader class."""
    def __init__(self, data_set, batch_size, seed=None):

        """__init__ function.

Args:
    data_set (my_data): Dataset wrapper with train/val/test splits.
    batch_size (int): Batch size for loaders.
    seed (int | None): Seed for deterministic shuffling.
    """
        self.data_set = data_set
        self.batch_size = batch_size
        self.seed = seed

        use_cuda = torch.cuda.is_available()
        kwargs = {'num_workers': 8, 'pin_memory': True} if use_cuda else {}
        # num_workers: the number of processes that generate batches in parallel.
        print('kwargs ',kwargs, 'batch_size ', batch_size)

        generator = None
        worker_init_fn = None
        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(seed)
            worker_init_fn = seed_worker

        self.train_loader = torch.utils.data.DataLoader(self.data_set.train_set,
                            batch_size=batch_size, shuffle=True,
                            generator=generator, worker_init_fn=worker_init_fn, **kwargs)

        self.val_loader   = torch.utils.data.DataLoader(self.data_set.val_set,
                            batch_size=batch_size, shuffle=True,
                            generator=generator, worker_init_fn=worker_init_fn, **kwargs)

        self.test_loader  = torch.utils.data.DataLoader(self.data_set.test_set,
                            batch_size=batch_size, shuffle=False,
                            generator=generator, worker_init_fn=worker_init_fn, **kwargs)
