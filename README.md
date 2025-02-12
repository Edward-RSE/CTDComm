# CTDComm

This is the repository for CTDComm...

## Requirements

You can install the requirements by using the `.yml` files in the `python_envs/` directory, or you can use the (currently
most tested method) installation script `python_envs/install-venv.sh` which will create a Python 3.12 virtual environment
in a directory named `.venv`.

If you choose to use the conda environment files, you will need to install the learning environments in `envs/`
manually.

## Training CAVE and Baselines (except MAGIC)

- Run `python run_baselines.py --help` to check all the options.
- Use `--commnet` to enable CommNet, `--ic3net` to enable IC3Net, `--tarcomm` and `--ic3net` to enable TarMAC-IC3Net, and `--gacomm` to enable GA-Comm.
- Use `--cave` to enable CAVE with TarMAC or GA-Comm.
- Each combination of method and environment has its own bash script, in `scripts/`.
- Most of these scripts were designed to be used in [SLURM](https://slurm.schedmd.com/) for HPC. For example, run TarMAC+CAVE in the Predator-Prey 10-agent scenario with `sbatch train_pp_hard_tarmac_cave.sh [seed=1]`.
- Most scripts should still work as standard bash scripts. For example, run TarMAC+CAVE in the Predator-Prey 10-agent scenario with `sh train_pp_hard_tarmac_cave.sh [seed=1]`.
  - If using `sh`, we recommend piping the output to a LOG file. This can be done by adding  `| tee <log_filename>.log` to the Python call.
- A seed can be provided as an optional argument when calling a bash script. By default, the seed is 1.
- Multiple runs with different seeds can be set up using `sbatch train_multiple_experiments <experiment_script>.sh [num_runs]`.

## References

The training framework is adapted from [MAGIC](https://github.com/CORE-Robotics-Lab/MAGIC). MAGIC's repository was,
in turn, adapted from [IC3Net](https://github.com/IC3Net/IC3Net)
