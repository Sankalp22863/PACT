# Usage:
# python3 MLPACT_v2.py --modelParamsFile ../Data/example_modelParams.config --configFile ../Data/example_config.config --floorplanFile ../Data/example_flp.csv --lcfFile ../Data/example_lcf.csv --power-data-csv ../Data/example_power_data.csv --ptrace-dir ../Data/example_ptraces/ --output-dir ../Data --n-training-samples 10 --visualize False --n-components-features 50 --n-components-targets 50

import os
import subprocess
import argparse
import MLPACT_MLCAD as MLPACT


def parse_args():
    parser = argparse.ArgumentParser(description="MLPACT pipeline")
   
   #for all
    parser.add_argument('--modelParamsFile', required=True)
    parser.add_argument('--configFile', required=True)
    parser.add_argument('--floorplanFile', required=True)
    parser.add_argument('--lcfFile', required=True)
    parser.add_argument("--power-data-csv", required = True)
    parser.add_argument('--ptrace-dir', required=True)
    parser.add_argument('--output-dir', required=True) # have a path for where train and test dirs will be created

    #for prep_training
    parser.add_argument('--n-training-samples', type=int, default=10)  
    parser.add_argument('--pact-script', default=None)     
    parser.add_argument('--select_mode', default=None)  

    #for prep_testing_iterate
    parser.add_argument('--init', default=None)              
    parser.add_argument('--steady', default=None)            
    parser.add_argument('--gridSteadyFile', default=None) 

    #for MLCAD
    parser.add_argument('--visualize', action='store_true', help="Enable visualization")
    parser.add_argument("--n-components-features", type = int, default = 50)
    parser.add_argument("--n-components-targets", type = int, default = 50)

    args = vars(parser.parse_args())
    
    train_dir = os.path.join(args['output_dir'], "train")
    os.makedirs(train_dir, exist_ok=True)
    test_dir = os.path.join(args['output_dir'], "test")
    os.makedirs(test_dir, exist_ok=True)
    
    args.update({
        "train_dir" : train_dir,
        "test_dir" : test_dir
    })
    
    return args


def run_prep_training(args):
    cmd = [
        "python3", "prep_training.py",
        "--ptrace-dir", args['ptrace_dir'],
        "--floorplan", args['floorplanFile'],
        "--config", args['configFile'],
        "--modelparams", args['modelParamsFile'],
        "--output-dir", args['train_dir']
    ]
    if args['n_training_samples'] is not None:
        cmd += ["--n", str(args['n_training_samples'])]
    if args['pact_script'] is not None:
        cmd += ["--pact-script", args['pact_script']]
    if args['select_mode'] is not None:
        cmd += ["--select-mode", args['select_mode']]
    subprocess.run(cmd, check=True)

    
def MLCAD_train(args):
    power_tensors, temp_tensors = MLPACT.load_train_data(args)
    scaler_X, pca_features, beta, pca_targets, scaler_y = MLPACT.train_model(args, power_tensors, temp_tensors)
    
    return scaler_X, pca_features, beta, pca_targets, scaler_y
 
    
def testing_iterate(args, scaler_X, pca_features, beta, pca_targets, scaler_y):
    # calling prep_testing iteratively
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    lcf_file = os.path.abspath(os.path.join(SCRIPT_DIR, args['lcfFile']))
    print (lcf_file)
    config_file = os.path.abspath(os.path.join(SCRIPT_DIR, args['configFile']))
    model_params = os.path.abspath(os.path.join(SCRIPT_DIR, args['modelParamsFile']))
    script_path = os.path.abspath(os.path.join(SCRIPT_DIR, "prep_testing.py"))
    output_dir = os.path.abspath(args['test_dir'])
    test_ptrace_dir = os.path.abspath(os.path.join(args['ptrace_dir'], 'test_ptraces'))

    os.makedirs(output_dir, exist_ok=True)

    files = sorted(os.listdir(test_ptrace_dir))
    if not files:
        raise RuntimeError(f"No ptrace CSVs found in {test_ptrace_dir}.")
        
    for i, file in enumerate(files):
        if not (file.startswith("inteli7") and file.endswith("ptrace.csv")):
            continue
        ptrace_path = os.path.join(test_ptrace_dir, file)
        new_lcf = os.path.join(output_dir, f"temp_lcf_{os.path.basename(file)}.csv")
        print(new_lcf)
        with open(lcf_file, "r") as fin, open(new_lcf, "w") as fout:
            header = next(fin)
            fout.write(header)
            for line in fin:
                parts = line.strip().split(",")
                parts[3] = ptrace_path
                fout.write(",".join(parts) + "\n")

        grid_steady_file = os.path.join(output_dir, f"{os.path.basename(file)}.grid.steady")
        print ("\n" + grid_steady_file)

        os.system(f"module load python3/3.6.5 gcc/5.5.0 fftw/3.3.8 netcdf/4.6.1 blis/0.6.0 openmpi/3.1.4_gnu-10.2.0 xyce/7.4 && python3 {script_path} {new_lcf} {config_file} {model_params} {output_dir} --gridSteadyFile {grid_steady_file}")

        cir_file = os.path.join(output_dir, f"{os.path.basename(file)}.cir")
        
        # -------------------------------------------
        MLPACT.predict_temp(args, cir_file, scaler_X, pca_features, beta, pca_targets, scaler_y)    

    
def main():
    args = parse_args()
    
    run_prep_training(args)
    
    scaler_X, pca_features, beta, pca_targets, scaler_y = MLCAD_train(args)
    testing_iterate(args, scaler_X, pca_features, beta, pca_targets, scaler_y)

if __name__ == "__main__":
    main()