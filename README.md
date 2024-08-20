# CAVE Implementation
This is the codebase for Communication-Augmented Value Estimation. The implementation is in two domains: Predator-Prey and Traffic-Junction.

## Requirements
* OpenAI Gym
* PyTorch 1.5 (CPU)
* Predator-Prey and Traffic Junction [Environments](https://github.com/apsdehal/ic3net-envs)


## Testing Environment Setup
* Predator-Prey and Traffic Junction (from IC3Net)
  ```
  cd envs/ic3net-envs
  python setup.py develop
  ```
  
## Training CAVE and Baselines (except MAGIC)
- `cd baselines`
- Run `python run_baselines.py --help` to check all the options.
- Use `--commnet` to enable CommNet, `--ic3net` to enable IC3Net, `--tarcomm` and `--ic3net` to enable TarMAC-IC3Net, and `--gacomm` to enable GA-Comm.
- Use `--cave` to enable CAVE with TarMAC or GA-Comm.
- Each combination of method and environment has its own bash script, in `/scripts/`.
- Most of these scripts were designed to be used in [SLURM](https://slurm.schedmd.com/) for HPC. For example, run TarMAC+CAVE in the Predator-Prey 10-agent scenario with `sbatch train_pp_hard_tarmac_cave.sh [seed=1]`.
- Most scripts should still work as standard bash scripts. For example, run TarMAC+CAVE in the Predator-Prey 10-agent scenario with `sh train_pp_hard_tarmac_cave.sh [seed=1]`.
  - If using `sh`, we recommend piping the output to a LOG file. This can be done by adding  `| tee <log_filename>.log` to the Python call.
- A seed can be provided as an optional argument when calling a bash script. By default, the seed is 1.
- Multiple runs with different seeds can be set up using `sbatch train_multiple_experiments <experiment_script>.sh [num_runs]`.


## Training MAGIC
- Run `python main.py --help` to check all the options.  
- Use `--first_graph_complete` and `--second_graph_complete` to set the corresponding communication graph of the first round and second round to be complete (disable the sub-scheduler), respectively.  
- Use `--comm_mask_zero` to block the communication.
- Example calls
  - Predator-Prey 5-agent scenario: `sh train_pp_medium.sh`
  - Predator-Prey 5-agent scenario: `sbatch train_pp_medium_slurm.sh`
  - Predator-Prey 5-agent scenario: `sbatch train_pp_medium_slurm_cave.sh`
  - Predator-Prey 10-agent scenario: `sbatch train_pp_hard_slurm.sh`

## Visualization
* Plot with multiple log files  
  Use plot_script.py (log files in saved/):
  ```
  python plot.py <path/to/file with all data for an environment> <figure filename> Reward
  python plot.py <path/to/file with all data for an environment> <figure filename> Success
  python plot.py <path/to/file with all data for an environment> <figure filename> Steps-Taken
  ```

## References
The training framework is adapted from [MAGIC](https://github.com/CORE-Robotics-Lab/MAGIC)

MAGIC's repository was, in turn, adapted from [IC3Net](https://github.com/IC3Net/IC3Net)
