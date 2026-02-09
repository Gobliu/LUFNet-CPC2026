import subprocess
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union, Dict, List

def run_baseline_results(log_dir: Union[str, Path]) -> None:
    """
    Python equivalent of:
    log_dir='results/baseline_run'
    cat $log_dir/log* > $log_dir/log_combine
    ./utils/show_results.sh $log_dir/log_combine
    """
    # 1. Path Setup using pathlib
    base_path = Path(log_dir)
    log_combine_file = base_path / "log_combine"
    script_path = script_dir / "utils" / "show_results.sh"

    # 2. Defensive Programming: Assertions and Explicit Errors
    assert base_path.exists(), f"Log directory not found: {base_path}"
    assert script_path.exists(), f"Results script not found: {script_path}"
    
    # Ensure the script is executable (POSIX)
    if not script_path.is_file() or not (script_path.stat().st_mode & 0o111):
        raise PermissionError(f"Script {script_path} is not executable or missing.")

    try:
        # 3. Exact execution of 'cat $log_dir/log* > $log_dir/log_combine'
        print(f"Cleaning up previous combine files in {base_path}...")
        for stale_file in base_path.glob("log_combine*"):
            stale_file.unlink()

        print(f"Concatenating logs in {base_path} using shell cat...")
        cat_command = f"cat {base_path}/log* > {log_combine_file}"
        subprocess.run(cat_command, shell=True, check=True)
        
        # 4. Execute the results script
        print(f"Executing {script_path} on {log_combine_file}...")
        subprocess.run(
            [str(script_path), str(log_combine_file)],
            check=True,
            capture_output=False,
            text=True
        )
        
    except subprocess.CalledProcessError as e:
        print(f"Error: Command failed with exit code {e.returncode}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise

def generate_log_dict(log_dir: Union[str, Path], save_path: Union[str, Path]) -> Dict[str, str]:
    """
    Generates a dictionary mapping metric keys to their respective log file paths.
    Always saves the dictionary to the provided save_path in JSON format.
    """
    base_path = Path(log_dir)
    prefix = base_path.name 
    
    metrics = {
        "trainqrmse": "qrmse.txt",
        "validqrmse": "qrmse_eval.txt",
        "trainqshape": "qshape.txt",
        "validqshape": "qshape_eval.txt",
        "trainprmse": "prmse.txt",
        "validprmse": "prmse_eval.txt",
        "trainpshape": "pshape.txt",
        "validpshape": "pshape_eval.txt",
        "trainermse": "ermse.txt",
        "validermse": "ermse_eval.txt",
        "traineshape": "eshape.txt",
        "valideshape": "eshape_eval.txt",
        "trainrelurep": "relurep.txt",
        "validrelurep": "relurep_eval.txt",
        "trainpoly": "poly.txt",
        "validpoly": "poly_eval.txt",
        "trainrep": "rep.txt",
        "validrep": "rep_eval.txt",
        "traintau": "tau.txt",
        "validtau": "tau_eval.txt",
        "outputx": "outputx.txt",
        "outputy": "outputy.txt",
        "train": "total.txt",
        "valid": "total_eval.txt"
    }

    log_dict = {}
    for key_suffix, file_suffix in metrics.items():
        key = f"{prefix}_{key_suffix}"
        file_path = base_path / f"log_combine_{file_suffix}"
        log_dict[key] = str(file_path)
    
    output_path = Path(save_path)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(log_dict, f, indent=2)
    print(f"Successfully saved log dictionary to {output_path}")
        
    return log_dict

def plot_loss_weights(data: Dict[str, str], run_prefix: str, weight_str: str, title: str) -> None:
    """
    Parses log data and generates a 4x4 plot grid of metrics.
    Replaces the functionality of loss_weight.py.
    """
    # Parse loss weights
    loss_weights: List[float] = []
    for w in weight_str.split(','):
        try:
            loss_weights.append(float(eval(w)))
        except (ValueError, SyntaxError, NameError):
            continue

    def load_metric(suffix: str) -> np.ndarray:
        key = f"{run_prefix}{suffix}"
        assert key in data, f"Required key '{key}' missing from log dictionary."
        path = Path(data[key])
        assert path.exists(), f"Log file for '{key}' not found at: {path}"
        # Using invalid_raise=False to handle potential trailing corruption or headers
        return np.genfromtxt(path, invalid_raise=False)

    # Load required arrays line by line
    tqrmse = load_metric('trainqrmse')
    vqrmse = load_metric('validqrmse')
    tqshape = load_metric('trainqshape')
    vqshape = load_metric('validqshape')
    tprmse = load_metric('trainprmse')
    vprmse = load_metric('validprmse')
    tpshape = load_metric('trainpshape')
    vpshape = load_metric('validpshape')
    termse = load_metric('trainermse')
    vermse = load_metric('validermse')
    teshape = load_metric('traineshape')
    veshape = load_metric('valideshape')
    trelureg = load_metric('trainrelurep')
    vrelureg = load_metric('validrelurep')
    
    # Process Epochs and Metrics
    tepoch = tqrmse[:, 1]
    vepoch = vqrmse[:, 1]

    qrmse_t = tqrmse[:, 5:]
    qrmse_v = vqrmse[:, 5:]
    prmse_t = tprmse[:, 5:]
    prmse_v = vprmse[:, 5:]
    ermse_t = termse[:, 5:]
    ermse_v = vermse[:, 5:]
    relureg_t = trelureg[:, 5:]
    relureg_v = vrelureg[:, 5:]

    # Plotting Logic
    plt.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(16, 14))

    for i in range(4):
        # Column 0: Q RMSE
        ax[i, 0].plot(tepoch, qrmse_t[:, 2*i+1], 'bo-', label='train', zorder=2)
        ax[i, 0].plot(vepoch, qrmse_v[:, 2*i+1], 'o-', label='valid', c='orange', zorder=1)
        
        # Column 1: P RMSE
        ax[i, 1].plot(tepoch, prmse_t[:, 2*i+1], 'bo-', label='train', zorder=2)
        ax[i, 1].plot(vepoch, prmse_v[:, 2*i+1], 'o-', label='valid', c='orange', zorder=1)
        
        # Column 2: E RMSE
        ax[i, 2].plot(tepoch, ermse_t[:, 2*i+1], 'bo-', label='train', zorder=2)
        ax[i, 2].plot(vepoch, ermse_v[:, 2*i+1], 'o-', label='valid', c='orange', zorder=1)
        ax[i, 2].set_ylim([-0.05, 100000])
        
        # Column 3: ReLU Reg
        ax[i, 3].plot(tepoch, relureg_t[:, 2*i+1], 'bo-', label='train', zorder=2)
        ax[i, 3].plot(vepoch, relureg_v[:, 2*i+1], 'o-', label='valid', c='orange', zorder=1)
        
        ax[i, 0].set_ylabel(r'$L_{}$'.format(2*i+2), fontsize=18)

    ax[0, 0].set_title('q L2 norm', fontsize=18)
    ax[0, 1].set_title('p L2 norm', fontsize=18)
    ax[0, 2].set_title('e L2 norm', fontsize=18)
    ax[0, 3].set_title('rep ReLU', fontsize=18)

    for i in range(4):
        for j in range(4):
            ax[i, j].grid(True, linestyle='--', alpha=0.7)
            ax[i, j].legend(loc='upper right', fontsize=12)
        ax[3, i].set_xlabel('epochs', fontsize=18)

    fig.suptitle(f"# {title}", fontsize=20, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save/Show
    plot_path = Path(data.get(f"{run_prefix}trainqrmse")).parent / f"{run_prefix}summary_plot.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Summary plot saved to: {plot_path}")
    plt.show()
    plt.close()

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent.parent
    
    # Parameters
    run_name = "baseline_run"
    plot_title = "dpt=20; ai tau =0.05; batch size 16; lr 1e-5; poly deg=4"
    loss_weights_arg = "0,1/8,0,1/4,0,1/2,0,1"
    dict_output_file = script_dir / "load_file.dict"
    
    # 1. Process log files (cat and show_results.sh)
    target_log_dir = script_dir / "results" / run_name
    run_baseline_results(target_log_dir)
    
    # 2. Generate and save the metadata dictionary
    results_map = generate_log_dict(target_log_dir, save_path=dict_output_file)

    # 3. Direct Plotting (Internalizing loss_weight.py logic)
    print(f"Generating summary plots for {run_name}...")
    plot_loss_weights(
        data=results_map, 
        run_prefix=f"{run_name}_", 
        weight_str=loss_weights_arg, 
        title=plot_title
    )
