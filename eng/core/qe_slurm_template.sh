#!/bin/bash
#SBATCH --job-name={name}         # Job name
#SBATCH --nodes=1                 # Number of nodes
#SBATCH --ntasks-per-node={n_cores}    # Number of MPI tasks per node
#SBATCH --cpus-per-task=1         # Number of CPU cores per task
#SBATCH --time=00:01:00           # Time limit hrs:min:sec
#SBATCH --deadline=now+2         # Total time limit including pending (2 minutes from submission)
#SBATCH --output={output_dir}/qe_%j.log   # Standard output log
#SBATCH --error={output_dir}/qe_%j.err    # Standard error log

# Load required modules
module purge
module load {module_name}

# Run QE with MPI
{qe_prefix} pw.x < {input_path} > {output_path}
