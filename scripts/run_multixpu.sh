#!/bin/bash -l
#SBATCH --job-name="3D_MultiXPU"
#SBATCH --output=3D_MultiXPU.%j.o
#SBATCH --error=3D_MultiXPU.%j.e
#SBATCH --time=12:00:00
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --account class04

# module load daint-gpu
# module load Julia/1.9.3-CrayGNU-21.09-cuda

export MPICH_RDMA_ENABLED_CUDA=0
export IGG_CUDAAWARE_MPI=0

srun -n8 bash -c 'julia NavierStokes3D_multixpu.jl'