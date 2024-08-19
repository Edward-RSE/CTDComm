#!/bin/bash
#
# Basic setup so I don't have to spend ages waiting for the big save files to add and commit
# Takes a commit message aas an optional input

#SBATCH --ntasks=1      # Number of tasks
#SBATCH --nodes=1       # Number of nodes
#SBATCH --cpus-per-task=1
#SBATCH --time=8:00:00 # Walltime (maximum is 60 hours for batch)

git add "./saved/predator_prey_*" #-A .

git commit -m "Data dump part 1, Predator-Prey"

# git push origin dev # Doesn't work because it needs my password