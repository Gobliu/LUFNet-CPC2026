import torch
import yaml
from pathlib import Path

from ML.trainer.trainer import trainer
from utils import utils, check_param_dict
from utils.system_logs import system_logs
from utils.seed import set_global_seed
from utils.mydevice import mydevice
from data_loader.data_loader import data_loader
from data_loader.data_loader import my_data


def _cfg(section, key, default=None, required=False):
    """Get config value from nested section, with flat-key fallback."""
    section_cfg = args.get(section, {})
    if isinstance(section_cfg, dict) and key in section_cfg:
        return section_cfg[key]
    if key in args:
        return args[key]
    if required:
        raise KeyError(f"Missing required config key: {section}.{key} (or top-level {key})")
    return default


def main():

    """Run training using parameters from train_config.yaml."""
    force_cuda = _cfg('system', 'force_cuda', True)
    _ = mydevice(force_cuda=force_cuda)
    _ = system_logs(mydevice)
    system_logs.print_start_logs()

    torch.set_default_dtype(torch.float64)
    #torch.autograd.set_detect_anomaly(True)

    seed = _cfg('system', 'seed', 34952)
    deterministic = _cfg('system', 'deterministic', True)
    set_global_seed(seed, deterministic)

    single_parnet_type = _cfg('model', 'single_parnet_type', required=True)
    multi_parnet_type = _cfg('model', 'multi_parnet_type', required=True)
    readout_net_type = _cfg('model', 'readout_net_type', required=True)
    trans_layer = _cfg('model', 'trans_layer', required=True)
    gnn_layer = _cfg('model', 'gnn_layer', required=True)
    nnode = _cfg('model', 'nnode', required=True)
    pw4mb_nnodes = _cfg('model', 'pw4mb_nnodes', 128)
    pw_output_dim = _cfg('model', 'pw_output_dim', 3)
    d_model = _cfg('model', 'd_model', 256)
    nhead = _cfg('model', 'nhead', 8)
    net_dropout = _cfg('model', 'net_dropout', 0.0)
    edge_attention = _cfg('model', 'edge_attention', True)
    init_weights = _cfg('model', 'init_weights', 'relu')
    tau_init = _cfg('model', 'tau_init', 1)

    tau_long = _cfg('simulation', 'tau_long', required=True)
    saved_pair_steps = _cfg('simulation', 'saved_pair_steps', required=True)
    window_sliding = _cfg('simulation', 'window_sliding', required=True)
    tau_traj_len_factor = _cfg('simulation', 'tau_traj_len_factor', 8)
    tau_traj_len = _cfg('simulation', 'tau_traj_len', tau_traj_len_factor * tau_long)
    allowed_window_sliding = _cfg('simulation', 'allowed_window_sliding',
                                  [1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 14, 16])

    ngrid = _cfg('grid', 'ngrid', required=True)
    b = _cfg('grid', 'b', required=True)
    a = _cfg('grid', 'a', required=True)

    optimizer = _cfg('optimizer', 'optimizer', 'Adam')
    maxlr = _cfg('optimizer', 'maxlr', required=True)
    grad_clip = _cfg('optimizer', 'grad_clip', 0.5)

    loss_weights = _cfg('loss', 'loss_weights', required=True)
    rthrsh = _cfg('loss', 'rthrsh', required=True)
    ew = _cfg('loss', 'ew', required=True)
    repw = _cfg('loss', 'repw', required=True)
    poly_deg = _cfg('loss', 'poly_deg', required=True)

    train_file = _cfg('data', 'train_file', required=True)
    valid_file = _cfg('data', 'valid_file', required=True)
    test_file = _cfg('data', 'test_file', required=True)
    dpt_train = _cfg('data', 'dpt_train', required=True)
    dpt_valid = _cfg('data', 'dpt_valid', required=True)
    dpt_test = _cfg('data', 'dpt_test', dpt_valid)
    batch_size = _cfg('data', 'batch_size', required=True)

    start_epoch = _cfg('run', 'start_epoch', required=True)
    end_epoch = _cfg('run', 'end_epoch', 20)
    loadfile = _cfg('run', 'loadfile', required=True)
    save_dir = _cfg('run', 'save_dir', './results/baseline_run')
    nitr = _cfg('run', 'nitr', required=True)
    tau_short = _cfg('run', 'tau_short', 1e-4)
    append_strike = _cfg('run', 'append_strike', nitr)
    ckpt_interval = _cfg('run', 'ckpt_interval', 1)
    val_interval = _cfg('run', 'val_interval', 1)
    verb = _cfg('run', 'verb', 1)

    traindict = {"loadfile"     : loadfile,  # to load previously trained model
                 "net_nnodes"   : nnode,   # number of nodes in neural nets
                 "pw4mb_nnodes" : pw4mb_nnodes,   # number of nodes in neural nets
                 "pw_output_dim" : pw_output_dim, # 20250803: change from 2D to 3D, psi
                 "init_weights"   : init_weights, #relu
                 "optimizer" : optimizer,
                 "single_particle_net_type" : single_parnet_type,         
                 "multi_particle_net_type"  : multi_parnet_type,        
                 "readout_step_net_type"    : readout_net_type,       
                 "n_encoder_layers" : trans_layer,
                 "n_gnn_layers" : gnn_layer,
                 "edge_attention" : edge_attention,
                 "d_model"      : d_model,
                 "nhead"        : nhead,
                 "net_dropout"  : net_dropout,    # 1: all zero ; 0 : not dropout ; 0.9 : 90% zero
                 "grad_clip"    : grad_clip,    # clamp the gradient for neural net parameters
                 "tau_traj_len" : tau_traj_len,  # n evaluations in integrator
                 "tau_long"     : tau_long,
                 "saved_pair_steps" : saved_pair_steps, # 20250809 in the saved time points, pair every steps with ai model time step size
                 "loss_weights"  : loss_weights,
                 "window_sliding" : window_sliding,  # number of times to do integration before cal the loss
                 "ngrids"       : ngrid,   # 6*len(b_list)
                 "b_list"       : b,       # grid lattice constant for multibody interactions
                 "a_list"       : a,       #[np.pi/8], #
                 "maxlr"        : maxlr,   # starting learning rate # HK
                 "tau_init"     : tau_init,       # starting learning rate
                 }

    if traindict["window_sliding"] not in allowed_window_sliding:
        print('window_sliding is not valid, need ', allowed_window_sliding)
        quit()

    lossdict = { "polynomial_degree" : poly_deg, # 4
                 "rthrsh"            : rthrsh, #0.7
                 "e_weight"          : ew,
                 "reg_weight"        : repw}

    data = {"train_file": train_file, # 20250809
            "valid_file": valid_file,
            "test_file": test_file,
            "train_pts" : dpt_train,
            "vald_pts"  : dpt_valid,
            "test_pts"  : dpt_test,
             "batch_size": batch_size,
             "window_sliding"   : traindict["window_sliding"] }

    maindict = { "start_epoch"     : start_epoch,
                 "end_epoch"       : end_epoch,
                 "save_dir"        : save_dir, # directory to save trained models
                 "tau_short"       : tau_short,
                 "nitr"            : nitr, # for check md trajectories # 1000 for ai tau 0.1; 100 for ai tau 0.01
                 "append_strike"   : append_strike, # for check md trajectories e.g. based on ground truth tau 0.0001, 100 for tau=0.01, 1000 for tau=0.1
                 "ckpt_interval"   : ckpt_interval, # for check pointing
                 "val_interval"    : val_interval, # no use of valid for now
                 "verb"            : verb  } # peroid for printing out losses


    utils.print_dict('trainer', traindict)
    utils.print_dict('loss', lossdict)
    utils.print_dict('data', data)
    utils.print_dict('main', maindict)

    print('begin ------- check param dict -------- ',flush=True)
    check_param_dict.check_maindict(traindict)
    check_param_dict.check_datadict(data)
    check_param_dict.check_traindict(maindict, traindict["tau_long"])
    print('end   ------- check param dict -------- ')

    Path(maindict["save_dir"]).mkdir(parents=True, exist_ok=True)

    data_set = my_data(data["train_file"],data["valid_file"],data["test_file"],
                       traindict["tau_long"],traindict["window_sliding"],traindict["saved_pair_steps"],
                       traindict["tau_traj_len"],data["train_pts"],data["vald_pts"],data["test_pts"])
    loader = data_loader(data_set, data["batch_size"], seed=seed)  # 20250908

    #utils.check_data(loader,data_set,traindict["tau_traj_len"],
    #           traindict["tau_long"],maindict["tau_short"],
    #           maindict["nitr"],maindict["append_strike"])

    train = trainer(traindict,lossdict)

    train.load_models()

    print('begin ------- initial learning configurations -------- ')
    train.verbose(0,'init_config')
    print('end  ------- initial learning configurations -------- ')

    for e in range(maindict["start_epoch"], maindict["end_epoch"]):

        cntr = 0
        for qpl_input,qpl_label in loader.train_loader:

            mydevice.load(qpl_input)
            q_traj,p_traj,q_label,p_label,l_init = utils.pack_data(qpl_input, qpl_label)
            # q_traj,p_ttaj [traj,nsamples,nparticles,dim]
            # q_label,p_label,l_init [nsamples,nparticles,dim]

            train.one_step(q_traj,p_traj,q_label,p_label,l_init)
            cntr += 1
            if cntr%10==0: print('.',end='',flush=True)

        print(cntr,'batches \n')

        if e%maindict["verb"]==0: 
            train.verbose(e+1,'train')
            system_logs.record_memory_usage(e+1)
            print('time use for ',maindict["verb"],'epoches is: ',end='')
            system_logs.record_time_usage(e+1)

        if e%maindict["ckpt_interval"]==0: 
            filename = str(Path(maindict["save_dir"]) / f"mbpw{e+1:06d}.pth")
            print('saving file to ',filename)
            train.checkpoint(filename)

        if e%maindict["val_interval"]==0: 
            train.loss_obj.clear()
            for qpl_input,qpl_label in loader.val_loader:
                q_traj,p_traj,q_label,p_label,l_init = utils.pack_data(qpl_input, qpl_label)
                train.eval(q_traj,p_traj,q_label,p_label,l_init)
            train.verbose(e+1,'eval')

    system_logs.print_end_logs()

if __name__=='__main__':
    yaml_config_path = 'train_config.yaml'
    with open(yaml_config_path, 'r') as f:
        args = yaml.safe_load(f)
    main()
