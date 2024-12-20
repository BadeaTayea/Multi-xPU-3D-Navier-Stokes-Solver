"""
Testing framework for the 3D Navier-Stokes solver. This script includes the following tests:

1. **Saving and Reading Arrays**:
   - Ensures arrays are correctly saved and loaded, particularly in a multi-XPU (GPU) environment.
   - This validation is critical for ensuring consistent behavior across distributed computing contexts.
   
2. **Setting a Sphere in a 3D Domain**:
   - Verifies the correct initialization of a sphere centered at the origin, within a domain where the concentration is initially set to 1.
   - This test is significant for multi-XPU contexts and confirms proper domain partitioning and synchronization.

3. **Reference Test**:
   - Performs a simple benchmark on a grid with dimensions `nz=63` and `nx=ny=31` over `nt=10` time steps.
   - Compares pressure and velocity fields against established ground truth to validate solver correctness after modifications.
"""

using Test
using MPI

# Initialize MPI
MPI.Init()

"""
Reference Test:
Verifies that the computed pressure and velocity fields match the ground truth on a predefined grid and time-step configuration.
"""

println("Starting reference test...")

# Include the reference test implementation from an external script.
include("./NavierStokes3D_multixpu_testing.jl")

println("Reference test passed.")
println()


"""
Unit Test: Saving and Loading Arrays
Validates that arrays are correctly saved and reloaded in a multi-XPU context. Asserts that data integrity is preserved across distributed systems.
"""

# Define the dimensions of the 3D domain.
nx, ny, nz = 8, 8, 16

# Initialize the global grid 
me, dims, nprocs, coords = init_global_grid(nx, ny, nz;init_MPI=false)

# Convert the coordinates to an array
coords = Data.Array(coords)

# Initialize the local array on each MPI process.
if me==0
    array = [i + j + k for i in 1:nx, j in 1:ny, k in 1:nz]
    A_loc = Data.Array(array)
else
    A_loc = @zeros(nx,ny,nz)
end

A_loc = A_loc.*2

# Update the halo regions of the array for domain decomposition.
update_halo!(A_loc)

# Compute the dimensions of the global domain (excluding halo regions).
nx_v, ny_v, nz_v = (nx - 2) * dims[1], (ny - 2) * dims[2], (nz - 2) * dims[3]

# Allocate space for the inner and global arrays.
A_inn = zeros(nx - 2, ny - 2, nz - 2) 
A_save = zeros(nx_v, ny_v, nz_v)

# Gather the inner array data into the global array.
A_inn.= Array(A_loc[2:end-1, 2:end-1, 2:end-1])
gather!(A_inn, A_save)

# Save the global array from the master process.
if me == 0
    save_array("./out/A_out", A_save)
end

# Initialize arrays for validation.
B = zeros(nx_v,ny_v,nz_v)
B_exact = [i + j + k for i in 2:nx-1, j in 2:ny-1, k in 2:nz-1]
B_exact.=B_exact.*2

# Load the saved array and validate correctness on the master process.
if me==0
    load_array("./out/A_out",B)
    println("Testing: Saving and loading of arrays...")
    @test B[1:nx-2,1:ny-2,1:nz-2]==B_exact
    if dims[1]>1 # if testing is performed on more than one GPU
        @test all(B[nx-1:end,ny-1:end,nz-1:end].==0)
    end
    println("Test passed.")
    println()
end


"""
Unit test: Testing setting sphere method, especially useful on multixpu, asserting if the sphere is indeed centered at the origin of the domain. 
"""

# Initialize velocity and concentration arrays.
Vx        = @zeros(nx+1,ny,nz)
Vy        = @zeros(nx  ,ny+1,nz)
Vz        = @zeros(nx  ,ny, nz+1)
C         = ones(Float64, nx,ny,nz)
C         = Data.Array(C)

# Define the domain dimensions and sphere parameters.
lz = 1.0 
lx_lz, ly_lz, r_lz = 0.5, 0.5, 0.05
ox_lz, oy_lz, oz_lz = 0.0, 0.0, 0.0

lx, ly = lx_lz * lz, ly_lz * lz
ox, oy, oz = ox_lz * lz, oy_lz * lz, oz_lz * lz
r2 = (r_lz * lz)^2

# Compute grid spacing.
dx, dy, dz = lx / nx_g(), ly / ny_g(), lz / nz_g()

# Set the sphere within the domain using parallelized computation.
@parallel set_sphere!(C,Vx,Vy,Vz,ox,oy,oz,lx,ly,lz,dx,dy,dz,r2,nx,ny,nz,coords)

# Gather the global concentration array for validation.
C_inn=zeros(nx-2,ny-2,nz-2)
C_glob=zeros(nx_v, ny_v, nz_v)
C_inn.= Array(C[2:end-1, 2:end-1, 2:end-1])
gather!(C_inn, C_glob)

# Validate the sphere placement on the master process.
if me==0
    println("Testing: Setting sphere inside the domain...")
    @test C_glob[Int(nx_v/2),Int(ny_v/2),Int(nz_v/2)]==0.0
    println("Test passed.")
end

# Finalize MPI
MPI.Finalize()




