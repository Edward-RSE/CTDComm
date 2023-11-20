#!/bin/bash
# Usage (for iridis/slurm):
# # Not yet written

#######################
## Not yet set up!!! ##
#######################
#SBATCH --ntasks=1      # Number of tasks
#SBATCH --nodes=1       # Number of nodes
#SBATCH --cpus-per-task=10 # Uses 40 CPUs for 4 workers (i.e. a full node)
#SBATCH --time=60:00:00 # Walltime (maximum is 60 hours for batch)

# Specific setup for my cluster and virtual environment
# module load cuda/10.2 # I don't think we're actually running cuda