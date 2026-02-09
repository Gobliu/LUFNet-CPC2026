import torch
import time
import sys
from pathlib import Path
from ML.trainer.trainer          import trainer
from ML.predicter.predicter      import predicter
from utils                       import check_param_dict
from utils                       import utils
from utils.system_logs           import system_logs
from utils.seed                  import set_global_seed
from utils.mydevice              import mydevice
from data_loader.data_loader import data_loader
from data_loader.data_loader import my_data
import yaml


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
    # python test_main.py 256 >  log/n256rho0.85T0.9LUF065_tau0.05

    """Run evaluation/inference using test_config.yaml."""
    force_cuda = _cfg('system', 'force_cuda', True)
    _ = mydevice(force_cuda=force_cuda)
    _ = system_logs(mydevice)
    system_logs.print_start_logs()

    torch.set_default_dtype(torch.float64)
    seed = _cfg('system', 'seed', 34952)
    deterministic = _cfg('system', 'deterministic', True)
    set_global_seed(seed, deterministic)

    argv = sys.argv
    if len(argv) > 1:
        npar = int(argv[1])
    else:
        npar = int(_cfg('eval', 'npar', 64))

    single_parnet_type = _cfg('model', 'single_parnet_type', required=True)
    multi_parnet_type = _cfg('model', 'multi_parnet_type', required=True)
    readout_net_type = _cfg('model', 'readout_net_type', required=True)
    nnode = _cfg('model', 'nnode', 128)
    pw4mb_nnodes = _cfg('model', 'pw4mb_nnodes', 128)
    pw_output_dim = _cfg('model', 'pw_output_dim', 3)
    trans_layer = _cfg('model', 'trans_layer', 2)
    gnn_layer = _cfg('model', 'gnn_layer', 2)
    d_model = _cfg('model', 'd_model', 256)
    nhead = _cfg('model', 'nhead', 8)
    net_dropout = _cfg('model', 'net_dropout', 0.0)
    edge_attention = _cfg('model', 'edge_attention', True)
    init_weights = _cfg('model', 'init_weights', 'relu')

    tau_long = _cfg('simulation', 'tau_long', required=True)
    saved_pair_steps = _cfg('simulation', 'saved_pair_steps', required=True)
    window_sliding = _cfg('simulation', 'window_sliding', required=True)
    tau_traj_len_factor = _cfg('simulation', 'tau_traj_len_factor', 8)
    tau_traj_len = _cfg('simulation', 'tau_traj_len', tau_traj_len_factor * tau_long)

    ngrid = _cfg('grid', 'ngrid', required=True)
    b = _cfg('grid', 'b', required=True)
    a = _cfg('grid', 'a', required=True)

    optimizer = _cfg('optimizer', 'optimizer', 'Adam')
    maxlr = _cfg('optimizer', 'maxlr', 1e-5)
    grad_clip = _cfg('optimizer', 'grad_clip', 0.5)
    tau_init = _cfg('optimizer', 'tau_init', 1)

    loss_weights = _cfg('loss', 'loss_weights', required=True)
    poly_deg = _cfg('loss', 'poly_deg', 4)
    rthrsh = _cfg('loss', 'rthrsh', 0.67)
    ew = _cfg('loss', 'ew', 1)
    repw = _cfg('loss', 'repw', 10)

    train_file = _cfg('data', 'train_file', required=True)
    valid_file = _cfg('data', 'valid_file', required=True)
    test_file = _cfg('data', 'test_file', required=True)
    train_pts = _cfg('data', 'train_pts', 10)
    valid_pts = _cfg('data', 'valid_pts', 10)
    test_pts = _cfg('data', 'test_pts', 10)
    batch_size = _cfg('data', 'batch_size', 2)

    ckpt_path = _cfg('run', 'ckpt_path', required=True)
    save_dir = _cfg('run', 'save_dir', './results/test_folder/')
    nitr = _cfg('run', 'nitr', required=True)
    tau_short = _cfg('run', 'tau_short', 1e-4)
    check_append_strike = _cfg('run', 'append_strike', nitr)

    ml_steps = _cfg('eval', 'ml_steps', required=True)
    append_strike = _cfg('eval', 'append_strike', required=True)
    gamma = _cfg('eval', 'gamma', required=True)
    temp = _cfg('metadata', 'temp', required=True)

    traindict = { "loadfile": ckpt_path,  # to load previously trained model
                  "net_nnodes": nnode,  # number of nodes in neural nets for force
                  "pw4mb_nnodes" : pw4mb_nnodes,
                  "pw_output_dim" : pw_output_dim, # 20250812: change from 2D to 3D, psi
                  "init_weights"   : init_weights,
                  "optimizer" : optimizer,
                  "single_particle_net_type" : single_parnet_type,
                  "multi_particle_net_type"  : multi_parnet_type,
                  "readout_step_net_type"    : readout_net_type,
                  "n_encoder_layers": trans_layer,
                  "n_gnn_layers" : gnn_layer,
                  "edge_attention" : edge_attention,
                  "d_model": d_model,
                  "nhead": nhead,
                  "net_dropout": net_dropout,  # 1: all zero ; 0 : not dropout ; 0.9 : 90% zero
                  "grad_clip": grad_clip,  # clamp the gradient for neural net parameters
                  "tau_traj_len": tau_traj_len,  # n evaluations in integrator
                  "tau_long": tau_long,
                  "saved_pair_steps": saved_pair_steps,
                  "loss_weights": loss_weights,
                  "window_sliding": window_sliding,  # number of times to do integration before cal the loss
                  "ngrids": ngrid,  # 6*len(b_list)
                  "b_list": b,  # # grid lattice constant for multibody interactions
                  "a_list": a,  # [np.pi/8], #
                  "ml_steps" : ml_steps, # ai 0.1 -> 10000; ai 0.02 -> 50000; ai 0.01 -> 100000
                  "append_strike" : append_strike, # ai 0.1 -> 10; ai 0.02 -> 50; ai 0.01 -> 100
                  "maxlr"    : maxlr,
                  "tau_init": tau_init, # starting learning rate
                  "gamma" : gamma
                 }

    lossdict = { "polynomial_degree" : poly_deg,
                 "rthrsh"       : rthrsh, #0.7 0.67
                 "e_weight"    : ew,
                 "reg_weight"    : repw}

    data = { "train_file": train_file,
             "valid_file": valid_file,
             "test_file" : test_file,
             "train_pts" : train_pts,
             "vald_pts"  : valid_pts,
             "test_pts"  : test_pts,
             "batch_size": batch_size}
    
    maindict = {"save_dir"     : save_dir,
                 "nitr"         : nitr,  # for check md trajectories
                 "tau_short"    : tau_short,
                 "append_strike": check_append_strike, # for check md trajectories
               }

    Path(maindict["save_dir"]).mkdir(parents=True, exist_ok=True)

    utils.print_dict('data',data)

    print(traindict)
    print(maindict)

    print('begin ------- check param dict -------- ')
    check_param_dict.check_maindict(traindict)
    check_param_dict.check_traindict(maindict,traindict["tau_long"])
    # check_param_dict.check_testdict(maindict)

    print('end   ------- check param dict -------- ')
    tau_traj_len = traindict["tau_traj_len"]
    tau_long = traindict["tau_long"]

    traj_len_prep = round(tau_traj_len / tau_long) * saved_pair_steps - saved_pair_steps

    print('traj len prep ', traj_len_prep, 't max', traindict['ml_steps']*tau_long , 'predict ml step ', traindict['ml_steps'])

    data_set = my_data(data["train_file"],data["valid_file"],data["test_file"],
                       traindict["tau_long"],traindict["window_sliding"],traindict["saved_pair_steps"],
                       traindict["tau_traj_len"],data["train_pts"],data["vald_pts"],data["test_pts"])  # 20250912

    loader = data_loader(data_set, data["batch_size"], seed=seed)

    train = trainer(traindict,lossdict)
    train.load_models()

    train.mlvv.eval()
    
    predict = predicter(train.prepare_data_obj, train.mlvv)

    with torch.no_grad():

      cntr = 0 
    
      for qpl_input,qpl_label in loader.test_loader:

          mydevice.load(qpl_input)
          q_traj,p_traj,q_label,p_label,l_init = utils.pack_data(qpl_input, qpl_label)

          q_input_list,p_input_list,q_cur,p_cur = predict.prepare_input_list(q_traj,p_traj,l_init)
          qpl_in = torch.unsqueeze(torch.stack((q_cur,p_cur,l_init),dim=1),dim=2) # use concat inital state

          qpl_batch = []
          start_time = time.time()

          for t in range(traindict['ml_steps']):

              print('====== t=',round(traj_len_prep + t * tau_long,3),' window sliding ', t+1,
                    't=', round((t+1) * tau_long + traj_len_prep,3), flush=True)
              #print('q',q_next.shape,'p', p_next.shape,'l', l_next.shape)

              q_input_list,p_input_list, q_predict, p_predict, l_init = predict.eval(q_input_list,p_input_list,q_cur,p_cur,l_init, t+1, gamma, temp)

              #if (t+1)%20 == 0: quit()
              qpl_list = torch.stack((q_predict, p_predict, l_init), dim=1)

              if (t + 1) % traindict['append_strike'] == 0:
                qpl_batch.append(qpl_list)

              q_cur = q_predict 
              p_cur = p_predict 

              #if t >= 1: quit()
              # print('t= ',t, 'CPU memory % used:', psutil.virtual_memory()[2], '\n')
  
          sec = time.time() - start_time
          #sec = sec / maindict["nitr"]
          #mins, sec = divmod(sec, 60)
          print("{} nitr --- {:.03f} sec ---".format(traindict['ml_steps'],sec))
          print("samples {}, one forward step timing --- {:.03f} sec ---".format(data["batch_size"], sec/traindict['ml_steps']))

          qpl_batch = torch.stack(qpl_batch, dim=2) # shape [nsamples,3, traj_len, nparticles,dim]
          # qpl_batch [nsamples,3,traj,nparticles,dim]

          print('====== load no batch ' , cntr, '==== shape ', qpl_in.shape,qpl_batch.shape)
          qpl_batch_cat = torch.cat((qpl_in,qpl_batch), dim=2) # stack traj inital + window-sliding
 
          tmp_filename = str(Path(maindict["save_dir"]) / f"{traindict['tau_long']}_id{cntr}.pt")
          print('saved qpl list shape', qpl_batch_cat.shape)
          torch.save({'qpl_trajectory': qpl_batch_cat, 'tau_short':maindict['tau_short'], 'tau_long' : traindict["tau_long"]},tmp_filename )
          # if i == 3*step: quit()
          cntr += 1
          #if cntr%10==0: print('.', end='', flush=True)

  
    #   qpl_list2 = torch.cat(qpl_list_sample, dim=0)
      # qpl_list [nsamples,3,traj,nparticles,dim]
 
    system_logs.print_end_logs()

if __name__=='__main__':
    yaml_config_path = 'test_config.yaml'
    with open(yaml_config_path, 'r') as f:
        args = yaml.safe_load(f)
    main()
