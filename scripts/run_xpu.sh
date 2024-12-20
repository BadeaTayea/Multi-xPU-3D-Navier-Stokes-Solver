#!/bin/bash -l
#SBATCH --job-name="3D_XPU"
#SBATCH --output=3D_XPU.%j.o
#SBATCH --error=3D_XPU.%j.e
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --account class04

# export MPICH_RDMA_ENABLED_CUDA=0
# export IGG_CUDAAWARE_MPI=0

# srun -n1 bash -c 'julia NavierStokes3D_xpu.jl'

srun julia NavierStokes3D_xpu.jl
