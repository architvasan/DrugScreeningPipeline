#!/bin/bash
#PBS -N openmm
#PBS -l select=1
#PBS -l walltime=12:00:00
#PBS -q preemptable
#PBS -l filesystems=home:grand:eagle
#PBS -A datascience
#PBS -o logs/
#PBS -e logs/
#PBS -m abe
#PBS -M avasan@anl.gov
cd /eagle/datascience/avasan/Workflow/DrugScreeningPipeline/Simulations
module load conda/2023-10-04
conda activate /eagle/datascience/avasan/envs/open_mm_forcefield

#cd /lus/grand/projects/datascience/avasan/Simulations/8GCY_Apo/run

python simulate_mpi_node3.py > node3.log
python simulate_mpi_node3_prod.py > node3p.log
