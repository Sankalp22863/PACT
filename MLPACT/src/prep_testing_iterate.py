# Usage:
# python3 prep_testing_iterate.py --ptrace_dir ../Data/example_ptraces/test_ptraces --output_dir ../Data/test/

import os
import subprocess
import argparse
from ChipStack import ChipStack

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description="Run prep_testing.py on multiple ptrace files.")
parser.add_argument('--lcfFile', default= "../Data/example_lcf.csv")
parser.add_argument('--configFile', default = '../Data/example_config.config')
parser.add_argument('--modelParamsFile', default = '../Data/example_modelParams.config')
parser.add_argument('--ptrace_dir', dest='ptrace_dir', default = '../Data/example_ptraces/test_ptraces')
parser.add_argument('--output_dir', help='Directory for output files', default = '--output_dir ../Data/test/')
parser.add_argument('--init', dest='initFile', default=None)
parser.add_argument('--steady', dest='steadyFile', default=None)
parser.add_argument('--gridSteadyFile', dest='gridSteadyFile', default= "../Data/test/Intel.grid.steady")

args = parser.parse_args()

lcf_file = os.path.abspath(os.path.join(SCRIPT_DIR, args.lcfFile))
print (lcf_file)
config_file = os.path.abspath(os.path.join(SCRIPT_DIR, args.configFile))
model_params = os.path.abspath(os.path.join(SCRIPT_DIR, args.modelParamsFile))
testing_path = os.path.abspath(os.path.join(SCRIPT_DIR, "prep_testing.py"))
output_dir = os.path.abspath(args.output_dir)
ptrace_dir = os.path.abspath(args.ptrace_dir)

os.makedirs(output_dir, exist_ok=True)

files = sorted(os.listdir(ptrace_dir)) 
for i, file in enumerate(files):
        if not (file.startswith("inteli7") and file.endswith("ptrace.csv")):
            continue
        ptrace_path = os.path.join(ptrace_dir, file)
        new_lcf = os.path.join(output_dir, f"temp_lcf_{os.path.basename(file)}.csv")
        with open(lcf_file, "r") as fin, open(new_lcf, "w") as fout:
            header = next(fin)
            fout.write(header)
            for line in fin:
                parts = line.strip().split(",")
                parts[3] = ptrace_path
                fout.write(",".join(parts) + "\n")

        grid_steady_file = os.path.join(output_dir, f"{os.path.basename(file)}.grid.steady")
        print ("\n" + grid_steady_file)
        
        os.system(f"module load python3/3.6.5 gcc/5.5.0 fftw/3.3.8 netcdf/4.6.1 blis/0.6.0 openmpi/3.1.4_gnu-10.2.0 xyce/7.4 && python3 {testing_path} {new_lcf} {config_file} {model_params} {output_dir} --gridSteadyFile {grid_steady_file}")