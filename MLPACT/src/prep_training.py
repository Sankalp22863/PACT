# Script to randomly select a set of 10-15 power traces and run them with PACT

# Importing modules
import pandas as pd
import os
import argparse
import glob
import os
import random
import subprocess
import numpy as np
import sys
import shutil
import shlex
from pathlib import Path

# ptraces in the ptrace directory are the first 30 ptraces in aminhaji/MLCAD_MLPACT/different_ptraces
# example config and modelparams files are from aminhaji/PACT/PACT/Intel

def parse_args():
        
    parser = argparse.ArgumentParser()  # Randomly select power traces and run PACT to generate outputs
    parser.add_argument('--ptrace-dir', default = '../Data/example_ptraces/')  # Directory containing power trace files (.csv)
    parser.add_argument('--floorplan', default = '../Data/example_flp.csv')  # Path for floorplan file
    parser.add_argument('--config', default = '../Data/example_config.config')  # Path for config files (Intel.config)
    parser.add_argument('--modelparams', default = '../Data/example_modelParams.config')  # Path for modelParams file
    parser.add_argument('--output-dir', default = '../Data/train/')  # Directory where grid.steady files will be saved
    parser.add_argument('--n', type=int, default = 10)  # Number of power traces to sample (10-15)
    parser.add_argument('--pact-script', default = './PACT.py')  # Path to PACT script
    parser.add_argument('--select-mode', choices = ['random', 'zscore', 'median_mse', 'median_mae'], default = 'median_mse')

    return vars(parser.parse_args())

# ---------------------------------------------
# Creating necessary output directories
def create_directories(args):
    os.makedirs(os.path.join(args['ptrace_dir'], "train_ptraces"), exist_ok=True)      # Output directory for train ptraces
    os.makedirs(os.path.join(args['ptrace_dir'], "test_ptraces"), exist_ok=True)       # Output directory for test ptraces
    os.makedirs(args['output_dir'], exist_ok = True)                 # Output directory for lcf files

# ---------------------------------------------
# loading and stacking powertrace data
def stack_ptraces(csv_paths):
    """
    Load and stack all ptraces, aligning to the union of numeric columns.
    Missing columns are filled with NaN so shapes match.
    Returns:
      stack: (num_files, units, times_union)
      numeric_colnames: tuple[str, ...] (the union, in natural order) - a list of the numeric headers from each file, used to figure out the union of time columns to be supported across all files
    """
    all_numeric_dfs, column_names_list = [], []

    for path in csv_paths:
        df = pd.read_csv(path)
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            raise ValueError(f"No numeric columns in {os.path.basename(csv_path)}")    
        all_numeric_dfs.append(numeric_df)
        column_names_list.append(tuple(numeric_df.columns.tolist()))
        
    # Build union of columns
    union_column_names = set()
    for names in column_names_list:
        for column_name in names:
            union_column_names.add(column_name)

    # Natural sort: "Power" first, then Power1, Power2, ...
    def power_key(c):
        if c == "Power":
            return 0
        try:
            return int(str(c).replace("Power", ""))
        except Exception:
            return float("inf")

    union_column_names = tuple(sorted(union_column_names, key=power_key))
        
    # Reindex each DF to the union (fill missing with NaN), then stack
    aligned_matrices = []
    for numeric_df in all_numeric_dfs:
        aligned_df = numeric_df.reindex(columns=union_column_names)
        aligned_matrices.append(aligned_df.to_numpy(dtype=np.float64))

    # Final shape check across files
    shapes = {m.shape for m in aligned_matrices}
    if len(shapes) != 1:
        raise RuntimeError(f"Matrix shapes differ across files after alignment: {shapes}")

    stack = np.stack(aligned_matrices, axis=0)  # (K, U, T)
    return stack, union_column_names
    
# ---------------------------------------------
# Ranking helpers - for mean/z-score and median/mse and median/mae

def rank_zscore(csv_paths, reduction='sum'):
    """
    For each (unit,time) cell, compute mean & std across files.
    Score each file by SUM or MEAN of |Z| over all cells. Larger => more atypical.
    Returns (sorted_paths, sorted_scores).
    """
    stack, first_cols = stack_ptraces(csv_paths)              # (K, U, T)
    mean_matrix = np.nanmean(stack, axis=0)                   # (U, T)
    std_matrix  = np.nanstd(stack,  axis=0, ddof=0)           # (U, T), ddof = delta degrees of freedom
    std_matrix[std_matrix == 0] = 1.0                         # avoid /0
    
    z_abs = np.abs((stack - mean_matrix) / std_matrix)     # (K, U, T)
    z_mean = np.nanmean(z_abs, axis=(1, 2))                # (K,) <- calculating mean instead of sum

    order = np.argsort(-z_mean)                            # largest first
    return [csv_paths[i] for i in order], z_mean[order]

def rank_median_distance(csv_paths, metric='mse'):
    """
    For each (unit, time) cell, compute median across files.
    Score each file by SUM of per-cell squared (MSE) or absolute (MAE) deviations comapred to median matrix
    Returns (sorted_paths, sorted_scores).
    """
    stack, first_cols = stack_ptraces(csv_paths)              # (K, U, T)
    median_matrix = np.nanmedian(stack, axis=0)               # (U, T)
    diff = stack - median_matrix                              # (K, U, T)

    if metric == 'mse':
        per_cell_errs = diff * diff                         # squared deviations
    elif metric == 'mae':
        per_cell_errs = np.abs(diff)                        # absolute deviations

    errs_mean = np.nanmean(per_cell_errs, axis=(1, 2))         # (K,) <- calculating mean instead of sum
    order = np.argsort(-errs_mean)                             # largest first
    return [csv_paths[i] for i in order], errs_mean[order]

# ---------------------------------------------
# Creating lcf file (.csv) similar to Intel_ID1_lcf.csv
def run_pact(ptrace_name, args):
    
    # Formatting the lcf file
    data = {
        'Layer':           [0],
        'FloorplanFile':   [args['floorplan']],
        'Thickness (m)':   [0.00075],
        'PtraceFile':      [args['ptrace_dir'] + ptrace_name],
        'LateralHeatFlow': [True],
    }
    
    lcf_data = pd.DataFrame(data)
    lcf_filename = f"{ptrace_name}_lcf.csv"
    lcf_data.to_csv(os.path.join(args['output_dir'], lcf_filename), index = False)
    
    print(lcf_filename)                 
    
    # Importing PACT required modules and running PACT    
    
    os.system(f"module load openmpi/3.1.4_gnu-10.2.0 xyce/7.4 && python {args['pact_script']} {args['output_dir']}/{lcf_filename} {args['config']} {args['modelparams']} --gridSteadyFile {args['output_dir']}/{ptrace_name}.grid.steady")
    
    
    
    # calling PACT using subprocess instead of os.system
    '''
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))   # .../Ronni/src

    # Build one shell command: load solver modules, then run PACT *as a module*
    module_load = "module load openmpi/3.1.4_gnu-10.2.0 xyce/7.4"  # adjust to your cluster
    lcf_path          = os.path.join(args['output_dir'], lcf_filename)
    grid_steady_path  = os.path.join(args['output_dir'], f"{ptrace_name}.grid.steady")
    
    bash_cmd = (
        f"{module_load} && "
        f"{shlex.quote(sys.executable)} -m PACT.PACT "
        f"{shlex.quote(lcf_path)} "
        f"{shlex.quote(args['config'])} "
        f"{shlex.quote(args['modelparams'])} "
        f"--gridSteadyFile {shlex.quote(grid_steady_path)}"
    )

    # Ensure the child can import the PACT package even if cwd is Data/test
    env = os.environ.copy()
    env['PYTHONPATH'] = SCRIPT_DIR + os.pathsep + env.get('PYTHONPATH', '')

    # Run inside Data/test so any relative files (e.g., *.cir, solver outputs) land there
    subprocess.run(
        ["bash", "-lc", bash_cmd],
        check=True,
        cwd=os.path.dirname(grid_steady_path),
        env=env,
    )
    '''

# ---------------------------------------------
# Finding all .csv files in directory with ALL ptraces, than randomly selecting n ptraces from all ptraces

def select_ptraces(args):
    
    # List all CSV files in the specified ptrace directory; glob.glob does a filesystem search using the pattern '*.csv'
    # os.path.join builds the full search path by concatenating the directory and wildcard
    all_ptraces = sorted(glob.glob(os.path.join(args['ptrace_dir'], '*.csv')),
                     key=os.path.basename)
    
    n = min(args['n'], len(all_ptraces))
    print(f"Selected {n} power traces for PACT runs")
    mode = args['select_mode']    
    
    '''
    valid_modes = {'random', 'zscore', 'median_mse', 'median_mae'}
    if mode not in valid_modes:
        print(f"Error: invalid --select-mode '{mode}'. Valid options: {sorted(valid_modes)}.", file=sys.stderr)
        sys.exit(2)
    '''

    if mode == 'random':  
        rng = random.Random(0)                      # <- local, deterministic
        selected = rng.sample(all_ptraces, n)
        for ptrace in selected:
            print(os.path.basename(ptrace))
    
    elif mode == 'zscore':
        ranked_paths, ranked_scores = rank_zscore(all_ptraces)
        selected = ranked_paths[:n]
        for ptrace, score in zip(selected, ranked_scores[:len(selected)]):
            print(f"{os.path.basename(ptrace):<45} score={score:.6f}")

    elif mode == 'median_mse':
        ranked_paths, ranked_scores = rank_median_distance(all_ptraces, metric='mse')
        selected = ranked_paths[:n]
        for ptrace, score in zip(selected, ranked_scores[:len(selected)]):
            print(f"{os.path.basename(ptrace):<45} score={score:.6f}")
            
    elif mode == 'median_mae':
        ranked_paths, ranked_scores = rank_median_distance(all_ptraces, metric='mae')        
        selected = ranked_paths[:n]
        for ptrace, score in zip(selected, ranked_scores[:len(selected)]):
            print(f"{os.path.basename(ptrace):<45} score={score:.6f}")
            
    else:
        print(f"Error: invalid --select-mode '{mode}'. \nValid options: 'random', 'zscore', 'median_mse', 'median_mae'")
        sys.exit(2)
            
    for ptrace in selected:
        print("\n ------------------------------------- \n" + os.path.basename(ptrace))
        # Call function to create lcf file for the ptrace + run PACT
        run_pact(os.path.basename(ptrace), args)
        shutil.move(os.path.join(args['ptrace_dir'], os.path.basename(ptrace)), os.path.join(args['ptrace_dir'], "train_ptraces"))
        
    print (f"\nMoved {n} ptraces from {args['ptrace_dir']} to {os.path.join(args['ptrace_dir'], 'train_ptraces')}")
        
# ---------------------------------------------
# Move remaining ptraces to test
def move_all_files(src_dir, dest_dir):
    
    num_files = sum(1 for file in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, file)))
    
    # Loop over all items in src_dir
    for file in os.listdir(src_dir):
        file_path = os.path.join(src_dir, file)

        # Move only files (skip directories)
        if os.path.isfile(file_path):
            shutil.move(file_path, dest_dir)

    print(f"\nMoved {num_files} from {src_dir} to {dest_dir}")
            
# ---------------------------------------------
# Main: parse arguments + call functions

def main():
    args = parse_args()
    
    create_directories(args) # Preparing output, train, and test directories
    move_all_files(os.path.join(args['ptrace_dir'], 'train_ptraces'), args['ptrace_dir']) # Train (from last run) -> ptrace dir
    move_all_files(os.path.join(args['ptrace_dir'], 'test_ptraces'), args['ptrace_dir']) # Test (from last run) -> ptrace dir
    
    select_ptraces(args) # selects n ptraces; create lcf files + run with pact
    move_all_files(args['ptrace_dir'], os.path.join(args['ptrace_dir'], 'test_ptraces')) # Ptrace dir -> test
    
if __name__ == '__main__':  # Makes sure it can run from command line
    main()