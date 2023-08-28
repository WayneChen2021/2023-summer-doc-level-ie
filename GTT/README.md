# Running GTT

## Via G2
1. Modify number of GPUs in [slurm](slurm/gtt.sub) and [bash](gtt-master/model_gtt/run_pl.sh) files
2. `sbatch --requeue slurm/gtt.sub`