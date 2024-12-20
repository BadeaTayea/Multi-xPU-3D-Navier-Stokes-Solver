"""
# Testing and Verification

This script implements a 3D iterative solver for the incompressible Navier-Stokes equations, 
optimized for execution on multi-xpu. The solver handles fluid dynamics across a spherical object 
and includes functionality for saving and loading intermediate states, as well as testing to ensure 
correctness of the simulation.

# Testing Workflow

The script includes built-in testing to verify simulation results:
- At the 10th time step, arrays for pressure and velocity components (`Vx`, `Vy`, `Vz`) are saved and compared against ground truth arrays.
- The `@test` macro is used to assert equality between simulation outputs and pre-saved ground truth values. 
- Any discrepancies in the results are flagged during testing.

## Key Functions for Testing:
- `save_array`: Saves a 3D array to a binary file.
- `load_array`: Loads a 3D array from a binary file.
- `@test`: Verifies correctness of simulation results against pre-saved ground truth arrays.

# Usage

1. Set `USE_GPU = true` for GPU execution, or `USE_GPU = false` for CPU execution.
2. Run the script to execute the solver.
3. Intermediate results are saved in the `./out/` directory at specified checkpoints.
4. Testing is automatically performed during the simulation if the 10th time step is reached.
"""


const USE_GPU = false
ENV["JULIA_TEST_VERBOSITY"] = "verbose"
using ParallelStencil
using ImplicitGlobalGrid
using ParallelStencil.FiniteDifferences3D

@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3)
else
    @init_parallel_stencil(Threads, Float64, 3)
end

using LinearAlgebra, Printf, MPI
using MAT, Test

function save_array(Aname,A)
    fname = string(Aname, ".bin")
    out = open(fname, "w"); write(out, A); close(out)
end

function load_array(Aname, A)
    fname = string(Aname, ".bin")
    fid = open(fname)  
    read!(fid, A)
    close(fid)
end


"""
    max_g(array)

Calculate the global maximum value of an array across all MPI processes.

# Arguments
- `array`: A numerical array for which the global maximum is calculated.

# Returns
- The maximum value of the array across all processes.
"""
max_g(A) = (max_l = maximum(A); MPI.Allreduce(max_l, MPI.MAX, MPI.COMM_WORLD))


"""
    avx(array)

Compute the average of adjacent elements along the x-direction.

# Arguments
- `array`: A three-dimensional array.

# Returns
- A reduced array containing the x-direction average.
"""
@views avx(A) = 0.5 .*(A[1:end-1,:,:] .+ A[2:end,:,:])


"""
    avy(array)

Compute the average of adjacent elements along the y-direction.

# Arguments
- `array`: A three-dimensional array.

# Returns
- A reduced array containing the y-direction average.
"""
@views avy(A) = 0.5 .*(A[:,1:end-1,:] .+ A[:,2:end,:])


"""
    avz(array)

Compute the average of adjacent elements along the z-direction.

# Arguments
- `array`: A three-dimensional array.

# Returns
- A reduced array containing the z-direction average.
"""
@views avz(A) = 0.5 .*(A[:,:,1:end-1] .+ A[:,:,2:end])


"""
    Navier_Stokes_3D_multixpu(; do_save)

Iterative solver for the incompressible Navier-Stokes equations in a 3D domain containing a sphere, utilizing multiple xPU devices.

# Keyword Arguments
- `do_save::Bool=true`: If `true`, saves the solution arrays at periodic checkpoints.

# Functionality
- Implements Chorin's projection method with explicit time-stepping.
- Handles boundary conditions, pressure correction, and velocity advection.
- Supports visualization and storage of intermediate results.

# Returns
- None (performs computations and writes output as needed).
"""
@views function Navier_Stokes_3D_multixpu(;  do_save=true)
    
    # physics
    ## dimensionally independent
    lz        = 1.0 # [m]
    ρ         = 1.0 # [kg/m^3]
    vin       = 1.0 # [m/s]
    ## scale
    psc       = ρ*vin^2
    ## nondimensional parameters
    Re        = 1e6    # rho*vsc*ly/μ
    Fr        = Inf   # vsc/sqrt(g*ly)
    lx_lz     = 0.5    # lx/ly
    ly_lz     = 0.5
    r_lz      = 0.05
    ox_lz     = 0.0
    oy_lz     = 0.0
    oz_lz     = -0.4
    ## dimensionally dependent
    lx        = lx_lz*lz
    ly        = ly_lz*lz
    ox        = ox_lz*lz
    oy        = oy_lz*lz
    oz        = oz_lz*lz
    r2         = (r_lz*lz)^2
    μ         = 1/Re*ρ*vin*lz
    g         = 1/Fr^2*vin^2/lz

    # numerics
    nz        = 63
    nx        = floor(Int,nz*lx_lz)
    ny        = floor(Int,nz*ly_lz) 
    me, dims,nprocs,coords  = init_global_grid(nx, ny, nz;init_MPI=false)
    coords    = Data.Array(coords)

    εit       = 1e-3
    niter     = 50*nx_g()
    nchk      = 1*(nx_g()-1)
    nt        = 10
    nsave     = 10
    CFLτ      = 0.9/sqrt(3)
    CFL_visc  = 1/5.1
    CFL_adv   = 1.0
    dx,dy,dz  = lx/nx_g(),ly/ny_g(),lz/nz_g()
    dt        = min(CFL_visc*dz^2*ρ/μ,CFL_adv*dz/vin)
    damp      = 2/nz_g()
    dτ        = CFLτ*dz
    
    # Allocations
    Pr        = @zeros(nx  ,ny  ,nz  )
    dPrdτ     = @zeros(nx-2,ny-2,nz-2)
    C         = @zeros(nx  ,ny  ,nz  )
    C_o       = @zeros(nx  ,ny  ,nz  )
    τxx       = @zeros(nx  ,ny  ,nz  )
    τyy       = @zeros(nx  ,ny  ,nz  )
    τzz       = @zeros(nx  ,ny  ,nz  )
    τxy       = @zeros(nx-1,ny-1,nz-2  )
    τxz       = @zeros(nx-1,ny-2  ,nz-1)
    τyz       = @zeros(nx-2  ,ny-1,nz-1)
    Vx        = @zeros(nx+1,ny,nz)
    Vy        = @zeros(nx  ,ny+1,nz)
    Vz        = @zeros(nx  ,ny, nz+1)
    Vx_o      = @zeros(nx+1,ny ,nz)
    Vy_o      = @zeros(nx  ,ny+1, nz)
    Vz_o      = @zeros(nx  ,ny, nz+1)
    ∇V        = @zeros(nx  ,ny  ,nz )
    Rp        = @zeros(nx-2,ny-2,nz-2)
    x = LinRange(0.5dx, lx-0.5dx,nx)
    y = LinRange(0.5dy, ly-0.5dy,ny)
   
    # Initial conditions
    Vprof = @zeros(nx,ny)
    Vprof .= Data.Array([4*vin*(x_g(ix,dx,Vprof)+0.5*dx)/lx*(1.0-(x_g(ix,dx,Vprof)+0.5dx)/lx) + 4*vin*(y_g(iy,dy,Vprof)+0.5dy)/ly*(1.0-(y_g(iy,dy,Vprof)+0.5dy)/ly) for  ix = 1:size(Vprof, 1), iy = 1:size(Vprof, 2) ]) 
    Vz[:,:,1] .= Vprof
    update_halo!(Vz)
    Pr          = Data.Array([-z_g(iz,dz,Pr)*ρ*g   for ix=1:size(Pr,1),iy=1:size(Pr,2),iz=1:size(Pr,3)])
    update_halo!(Pr)
    
    if do_save
        nx_v, ny_v, nz_v = (nx - 2) * dims[1], (ny - 2) * dims[2], (nz - 2) * dims[3]
        (nx_v * ny_v * nz_v * sizeof(Data.Number) > 0.8 * Sys.free_memory()) && error("Not enough memory for saving.")
        Pr_v   = zeros(nx_v, ny_v, nz_v) 
        Vx_v   = zeros(nx_v, ny_v, nz_v) 
        Vy_v   = zeros(nx_v, ny_v, nz_v) 
        Vz_v   = zeros(nx_v, ny_v, nz_v) 
        Pr_inn = zeros(nx - 2, ny - 2, nz - 2) 
        Vx_inn = zeros(nx - 2, ny - 2, nz - 2) 
        Vy_inn = zeros(nx - 2, ny - 2, nz - 2) 
        Vz_inn = zeros(nx - 2, ny - 2, nz - 2) 
    end


    for it = 1:nt
        err_evo = Float64[]; iter_evo = Float64[]
        @parallel update_τ!(τxx,τyy,τzz,τxy,τxz,τyz,Vx,Vy,Vz,μ,dx,dy,dz)
        @parallel predict_V!(Vx,Vy,Vz,τxx,τyy,τzz,τxy,τxz,τyz,ρ,g,dt,dx,dy,dz)
        @parallel set_sphere!(C,Vx,Vy,Vz,ox,oy,oz,lx,ly,lz,dx,dy,dz,r2,nx,ny,nz,coords)
        update_halo!(Vx,Vy,Vz)
        @parallel update_∇V!(∇V,Vx,Vy,Vz,dx,dy,dz)
        if me==0
            println("#it = $it")
        end
        for iter = 1:niter
            @parallel update_dPrdτ!(Pr,dPrdτ,∇V,ρ,dt,dτ,damp,dx,dy,dz)
            @parallel update_Pr!(Pr,dPrdτ,dτ)
            set_bc_Pr!(Pr, 0.0)
            update_halo!(Pr)
            if iter % nchk == 0
                @parallel compute_res!(Rp,Pr,∇V,ρ,dt,dx,dy,dz)
                err = max_g(abs.(Rp))*lz^2/psc
                push!(err_evo, err); push!(iter_evo,iter/nz)
                if me==0
                    @printf("  #iter = %d, err = %1.3e\n", iter, err)
                end
                if err < εit || !isfinite(err) break end
            end
        end

        
        @parallel correct_V!(Vx,Vy,Vz,Pr,dt,ρ,dx,dy,dz)
        @parallel set_sphere!(C,Vx,Vy,Vz,ox,oy,oz,lx,ly,lz,dx,dy,dz,r2,nx,ny,nz,coords)
        set_bc_Vel!(Vx, Vy, Vz, Vprof)
        update_halo!(Vx,Vy,Vz)
        Vx_o .= Vx; Vy_o .= Vy; Vz_o .= Vz; C_o .= C
        @parallel advect!(Vx,Vx_o,Vy,Vy_o,Vz,Vz_o,C,C_o,dt,dx,dy,dz)
        update_halo!(Vx,Vy,Vz)
        
        # Saving the arrays
        if do_save && it % nsave == 0 && me==0
            Pr_inn .= Array(Pr[2:end-1, 2:end-1, 2:end-1]); gather!(Pr_inn, Pr_v)
            Vx_inn .= Array(avx(Vx)[2:end-1, 2:end-1, 2:end-1]); gather!(Vx_inn, Vx_v)
            Vy_inn .= Array(avy(Vy)[2:end-1, 2:end-1, 2:end-1]); gather!(Vy_inn, Vy_v)
            Vz_inn .= Array(avz(Vz)[2:end-1, 2:end-1, 2:end-1]); gather!(Vz_inn, Vz_v)
        end
        
        if do_save && it % nsave == 0 && me==0
            save_array("./out/out_Vx_$it", Vx_v)
            save_array("./out/out_Vy_$it", Vy_v)
            save_array("./out/out_Vz_$it", Vz_v)
            save_array("./out/out_Pr_$it", Pr_v)
        end
        
        # Testing according to the ground truth
        if it==10 && me==0
            Pr_truth   = zeros(nx_v, ny_v, nz_v) 
            Vx_truth   = zeros(nx_v, ny_v, nz_v) 
            Vy_truth   = zeros(nx_v, ny_v, nz_v) 
            Vz_truth   = zeros(nx_v, ny_v, nz_v)

            load_array("./out/out_Pr_10",Pr_truth)
            load_array("./out/out_Vx_10",Vx_truth)
            load_array("./out/out_Vy_10",Vy_truth)
            load_array("./out/out_Vz_10",Vz_truth)

            @test Pr_truth==Pr_v
            @test Vx_truth==Vx_v
            @test Vy_truth==Vy_v
            @test Vz_truth==Vz_v
        end

    end
    
    finalize_global_grid(;finalize_MPI=false) 
    return
end

"""
    ∇V()

Calculate the divergence of the velocity field.

# Macro Definition
- Computes the divergence using finite-difference operations:
  `∇V = ∂Vx/∂x + ∂Vy/∂y + ∂Vz/∂z`.
"""
macro ∇V() esc(:( @d_xa(Vx)/dx + @d_ya(Vy)/dy +@d_za(Vz)/dz)) end


"""
    update_τ!(τxx, τyy, τzz, τxy, τxz, τyz, Vx, Vy, Vz, μ, dx, dy, dz)

Compute the components of the shear stress tensor based on velocity gradients.

# Arguments
- `τxx`, `τyy`, `τzz`: Normal stress components.
- `τxy`, `τxz`, `τyz`: Shear stress components.
- `Vx`, `Vy`, `Vz`: Velocity components.
- `μ`: Dynamic viscosity.
- `dx`, `dy`, `dz`: Grid spacing in the respective directions.

# Side Effects
- Updates the stress tensor components in-place.
"""
@parallel function update_τ!(τxx,τyy,τzz,τxy,τxz,τyz,Vx,Vy,Vz,μ,dx,dy,dz)
    @all(τxx) = 2μ*(@d_xa(Vx)/dx - @∇V()/3.0)  
    @all(τyy) = 2μ*(@d_ya(Vy)/dy - @∇V()/3.0)
    @all(τxy) =  μ*(@d_yi(Vx)/dy + @d_xi(Vy)/dx)
    @all(τzz) = 2μ*(@d_za(Vz)/dz - @∇V()/3.0)
    @all(τxz) =  μ*(@d_zi(Vx)/dz + @d_xi(Vz)/dx)
    @all(τyz) =  μ*(@d_zi(Vy)/dz + @d_yi(Vz)/dy)
    return
end


"""
    predict_V!(Vx, Vy, Vz, τxx, τyy, τzz, τxy, τxz, τyz, ρ, g, dt, dx, dy, dz)

Compute the first intermediate velocity using viscous and gravitational forces.

# Arguments
- `Vx`, `Vy`, `Vz`: Velocity components (updated in-place).
- `τxx`, `τyy`, `τzz`: Normal stress components.
- `τxy`, `τxz`, `τyz`: Shear stress components.
- `ρ`: Fluid density.
- `g`: Gravitational acceleration.
- `dt`: Time step size.
- `dx`, `dy`, `dz`: Grid spacing.

# Returns
- None (updates velocities in-place).
"""
@parallel function predict_V!(Vx,Vy,Vz,τxx,τyy,τzz,τxy,τxz,τyz,ρ,g,dt,dx,dy,dz)
    @inn(Vx) = @inn(Vx) + dt/ρ*(@d_xi(τxx)/dx + @d_ya(τxy)/dy + @d_za(τxz)/dz)
    @inn(Vy) = @inn(Vy) + dt/ρ*(@d_yi(τyy)/dy + @d_xa(τxy)/dx  + @d_za(τyz)/dz)
    @inn(Vz) = @inn(Vz) + dt/ρ*(@d_zi(τzz)/dz + @d_xa(τxz)/dx  + @d_ya(τyz)/dy - ρ*g)
    return
end


"""
    update_∇V!(∇V, Vx, Vy, Vz, dx, dy, dz)

Compute the divergence of the velocity field.

# Arguments
- `∇V`: Array to store the computed divergence of the velocity field.
- `Vx`, `Vy`, `Vz`: Velocity components in the x, y, and z directions.
- `dx`, `dy`, `dz`: Grid spacing in the x, y, and z directions.

# Side Effects
- Updates the `∇V` array in-place with the divergence values.
"""
@parallel function update_∇V!(∇V,Vx,Vy,Vz,dx,dy,dz)
    @all(∇V) = @∇V()
    return
end


"""
    update_dPrdτ!(Pr, dPrdτ, ∇V, ρ, dt, dτ, damp, dx, dy, dz)

Compute the pressure gradient with respect to pseudo-time using pseudo-transient methods.

# Arguments
- `Pr`: Pressure field.
- `dPrdτ`: Array to store the rate of change of pressure with respect to pseudo-time.
- `∇V`: Divergence of the velocity field.
- `ρ`: Fluid density.
- `dt`: Time step size.
- `dτ`: Pseudo-time step size.
- `damp`: Damping factor for convergence.
- `dx`, `dy`, `dz`: Grid spacing.

# Side Effects
- Updates the `dPrdτ` array in-place with the computed pseudo-time gradient.
"""
@parallel function update_dPrdτ!(Pr,dPrdτ,∇V,ρ,dt,dτ,damp,dx,dy,dz)
    @all(dPrdτ) = @all(dPrdτ)*(1.0-damp) + dτ*(@d2_xi(Pr)/dx/dx + @d2_yi(Pr)/dy/dy + @d2_zi(Pr)/dz/dz - ρ/dt*@inn(∇V))
    return
end


"""
    update_Pr!(Pr, dPrdτ, dτ)

Update the pressure field based on the pseudo-time pressure gradient.

# Arguments
- `Pr`: Pressure field (updated in-place).
- `dPrdτ`: Rate of change of pressure with respect to pseudo-time.
- `dτ`: Pseudo-time step size.

# Side Effects
- Updates the `Pr` array in-place with the new pressure values.
"""
@parallel function update_Pr!(Pr,dPrdτ,dτ)
    @inn(Pr) = @inn(Pr) + dτ*@all(dPrdτ)
    return
end


"""
    compute_res!(Rp, Pr, ∇V, ρ, dt, dx, dy, dz)

Calculate the residual of the pressure Poisson equation.

# Arguments
- `Rp`: Array to store the computed residuals.
- `Pr`: Pressure field.
- `∇V`: Divergence of the velocity field.
- `ρ`: Fluid density.
- `dt`: Time step size.
- `dx`, `dy`, `dz`: Grid spacing.

# Side Effects
- Updates the `Rp` array in-place with the computed residual values.
"""
@parallel function compute_res!(Rp,Pr,∇V,ρ,dt,dx,dy,dz)
    @all(Rp) = @d2_xi(Pr)/dx/dx + @d2_yi(Pr)/dy/dy + @d2_zi(Pr)/dz/dz - ρ/dt*@inn(∇V)
    return
end


"""
    correct_V!(Vx, Vy, Vz, Pr, dt, ρ, dx, dy, dz)

Apply pressure correction to the velocity components.

# Arguments
- `Vx`, `Vy`, `Vz`: Velocity components (updated in-place).
- `Pr`: Pressure field.
- `dt`: Time step size.
- `ρ`: Fluid density.
- `dx`, `dy`, `dz`: Grid spacing.

# Side Effects
- Updates the velocity components (`Vx`, `Vy`, `Vz`) in-place.
"""
@parallel function correct_V!(Vx,Vy,Vz,Pr,dt,ρ,dx,dy,dz)
    @inn(Vx) = @inn(Vx) - dt/ρ*@d_xi(Pr)/dx
    @inn(Vy) = @inn(Vy) - dt/ρ*@d_yi(Pr)/dy
    @inn(Vz) = @inn(Vz) - dt/ρ*@d_zi(Pr)/dz 
    return
end

"""
    bc_yz!(A)

Apply zero-flux boundary conditions in the yz-plane for a given array.

# Arguments
- `A`: Array representing a physical quantity.

# Side Effects
- Updates the boundary values of `A` in-place along the yz-plane.
"""
@parallel_indices (iy, iz) function bc_yz!(A)
    if iy <= size(A,2) && iz <= size(A,3)
        A[1  , iy,  iz] = A[2    , iy,  iz]
        A[end, iy,  iz] = A[end-1, iy,  iz]
    end
    return
end


"""
    bc_xz!(A)

Apply zero-flux boundary conditions in the xz-plane for a given array.

# Arguments
- `A`: Array representing a physical quantity.

# Side Effects
- Updates the boundary values of `A` in-place along the xz-plane.
"""
@parallel_indices (ix, iz) function bc_xz!(A)
    if ix <= size(A,1) && iz <= size(A,3)
        A[ix, 1  ,  iz] = A[ix,     2, iz]
        A[ix, end,  iz] = A[ix, end-1, iz]
    end
    return
end


"""
    bc_xy!(A)

Apply zero-flux boundary conditions in the xy-plane for a given array.

# Arguments
- `A`: Array representing a physical quantity.

# Side Effects
- Updates the boundary values of `A` in-place along the xy-plane.
"""
@parallel_indices (ix, iy) function bc_xy!(A)
    if ix <= size(A,1) && iy <= size(A,2)
        A[ix, iy,    1] = A[ix,  iy,     2]
        A[ix, iy,  end] = A[ix,  iy, end-1]
    end
    return
end


"""
    bc_zV!(A, V)

Apply a zero-flux boundary condition in one xy-plane and a Dirichlet boundary condition with value `V` in the other xy-plane.

# Arguments
- `A`: Array representing a physical quantity.
- `V`: Array specifying the Dirichlet boundary condition values.

# Side Effects
- Updates the boundary values of `A` in-place along the xy-plane.
"""
@parallel_indices (ix, iy) function bc_zV!(A, V)
    A[ix,   iy,   1] = V[    ix,    iy]
    A[ix,   iy, end] = A[ix, iy, end-1]
    return
end


"""
    bc_xyval!(A, val)

Apply a zero-flux boundary condition in one xy-plane and a Dirichlet boundary condition with value `val` in the other xy-plane.

# Arguments
- `A`: Array representing a physical quantity.
- `val`: Scalar value for the Dirichlet boundary condition.

# Side Effects
- Updates the boundary values of `A` in-place.
"""
@parallel_indices (ix, iy) function bc_xyval!(A, val)
    A[ix, iy,   1] = A[ix, iy,  2]
    A[ix, iy, end] = val
    return
end


"""
    set_bc_Vel!(Vx, Vy, Vz, Vprof)

Set boundary conditions for the velocity components throughout the domain.

# Arguments
- `Vx`, `Vy`, `Vz`: Velocity components (updated in-place).
- `Vprof`: Prescribed velocity profile for the z-direction on the boundary.

# Side Effects
- Updates the velocity components with the prescribed boundary conditions.
"""
function set_bc_Vel!(Vx, Vy, Vz, Vprof)
    @parallel bc_xy!(Vx)
    @parallel bc_xz!(Vx)
    @parallel bc_xy!(Vy)
    @parallel bc_yz!(Vy)
    @parallel bc_xz!(Vz)
    @parallel bc_yz!(Vz)
    @parallel bc_zV!(Vz, Vprof)
    return
end


"""
    set_bc_Pr!(Pr, val)

Set boundary conditions for the pressure field throughout the domain.

# Arguments
- `Pr`: Pressure field (updated in-place).
- `val`: Scalar value for the Dirichlet boundary condition.

# Side Effects
- Updates the pressure field `Pr` with the prescribed boundary conditions.
"""
function set_bc_Pr!(Pr, val)
    @parallel bc_xz!(Pr)
    @parallel bc_yz!(Pr)
    @parallel bc_xyval!(Pr, val)
    return
end


"""
    backtrack!(A, A_o, vxc, vyc, vzc, dt, dx, dy, dz, ix, iy, iz)

Perform backtracking for semi-Lagrangian advection using velocity components.

# Arguments
- `A`: Array to be updated.
- `A_o`: Array representing previous time step values.
- `vxc`, `vyc`, `vzc`: Velocity components at the grid point.
- `dt`: Time step size.
- `dx`, `dy`, `dz`: Grid spacing.
- `ix`, `iy`, `iz`: Indices of the current grid point.

# Side Effects
- Updates the value of `A` at the specified grid point.
"""
@inline function backtrack!(A,A_o,vxc,vyc,vzc,dt,dx,dy,dz,ix,iy,iz)
    δx,δy,δz    = dt*vxc/dx, dt*vyc/dy , dt*vzc/dz
    ix1      = clamp(floor(Int,ix-δx),1,size(A,1))
    iy1      = clamp(floor(Int,iy-δy),1,size(A,2))
    iz1      = clamp(floor(Int,iz-δz),1,size(A,3))
    ix2,iy2,iz2  = clamp(ix1+1,1,size(A,1)),clamp(iy1+1,1,size(A,2)),clamp(iz1+1,1,size(A,3))
    δx = (δx>0) - (δx%1); δy = (δy>0) - (δy%1) ; δz = (δz>0) - (δz%1)
    fx11      = lerp(A_o[ix1,iy1,iz1],A_o[ix2,iy1,iz1],δx)
    fx12      = lerp(A_o[ix1,iy2,iz1],A_o[ix2,iy2,iz1],δx)
    fx1 = lerp(fx11,fx12,δy)
    fx21      = lerp(A_o[ix1,iy1,iz2],A_o[ix2,iy1,iz2],δx)
    fx22      = lerp(A_o[ix1,iy2,iz2],A_o[ix2,iy2,iz2],δx)
    fx2 = lerp(fx21,fx22,δy)
    A[ix,iy,iz] = lerp(fx1,fx2,δz)
    return
end


"""
    lerp(a, b, t)

Perform linear interpolation between two values.

# Arguments
- `a`: Starting value.
- `b`: Ending value.
- `t`: Interpolation factor (0 ≤ t ≤ 1).

# Returns
- The interpolated value.
"""
@inline lerp(a,b,t) = b*t + a*(1-t)


"""
    advect!(Vx, Vx_o, Vy, Vy_o, Vz, Vz_o, dt, dx, dy, dz)

Perform semi-Lagrangian advection of the velocity components using linear interpolation.

# Arguments
- `Vx`, `Vy`, `Vz`: Velocity components (updated in-place).
- `Vx_o`, `Vy_o`, `Vz_o`: Velocity components from the previous time step.
- `dt`: Time step size.
- `dx`, `dy`, `dz`: Grid spacing.

# Side Effects
- Updates velocity components with advected values.
"""
@parallel_indices (ix,iy,iz) function advect!(Vx,Vx_o,Vy,Vy_o,Vz,Vz_o,C,C_o,dt,dx,dy,dz)
    if ix > 1 && ix < size(Vx,1) && iy <= size(Vx,2) && iz<=size(Vx,3)
        vxc      = Vx_o[ix,iy,iz]
        vyc      = 0.25*(Vy_o[ix-1,iy,iz]+Vy_o[ix-1,iy+1,iz]+Vy_o[ix,iy,iz]+Vy_o[ix,iy+1,iz])
        vzc      = 0.25*(Vz_o[ix-1,iy,iz]+Vz_o[ix-1,iy,iz+1]+Vz_o[ix,iy,iz]+Vz_o[ix,iy,iz+1])
        backtrack!(Vx,Vx_o,vxc,vyc,vzc,dt,dx,dy,dz,ix,iy,iz)
    end
    if iy > 1 && iy < size(Vy,2) && ix <= size(Vy,1) && iz<=size(Vy,3)
        vxc      = 0.25*(Vx_o[ix,iy-1,iz]+Vx_o[ix+1,iy-1,iz]+Vx_o[ix,iy,iz]+Vx_o[ix+1,iy,iz])
        vyc      = Vy_o[ix,iy,iz]
        vzc      = 0.25*(Vz_o[ix,iy-1,iz]+Vz_o[ix,iy-1,iz+1]+Vz_o[ix,iy,iz]+Vz_o[ix,iy,iz+1])
        backtrack!(Vy,Vy_o,vxc,vyc,vzc,dt,dx,dy,dz,ix,iy,iz)
    end
    if iz > 1 && iz < size(Vz,3) && ix <= size(Vz,1) && iy<=size(Vz,2)
        vxc      = 0.25*(Vx_o[ix,iy,iz-1]+Vx_o[ix+1,iy,iz-1]+Vx_o[ix,iy,iz]+Vx_o[ix+1,iy,iz])
        vyc      = 0.25*(Vy_o[ix,iy,iz-1]+Vy_o[ix,iy+1,iz-1]+Vy_o[ix,iy,iz]+Vy_o[ix,iy+1,iz])
        vzc      = Vz_o[ix,iy,iz]
        backtrack!(Vz,Vz_o,vxc,vyc,vzc,dt,dx,dy,dz,ix,iy,iz)
    end
    return
end


"""
    set_sphere!(Vx,Vy,Vz,ox,oy,oz,lx,ly,lz,dx,dy,dz,r2)

Place a no-slip spherical object in the simulation domain.

# Arguments
- `C`: Array indicating the presence of the sphere in the domain.
- `Vx`, `Vy`, `Vz`: Velocity components (updated in-place to zero inside the sphere).
- `ox`, `oy`, `oz`: Sphere center coordinates.
- `lx`, `ly`, `lz`: Domain dimensions.
- `dx`, `dy`, `dz`: Grid spacing.
- `r2`: Square of the sphere radius.

# Side Effects
- Modifies the velocity components to impose no-slip conditions inside the sphere.
"""
@parallel_indices (ix,iy,iz) function set_sphere!(C,Vx,Vy,Vz,ox,oy,oz,lx,ly,lz,dx,dy,dz,r2,nx,ny,nz,coords)
    xc,yc,zc = (coords[1]*(nx-2)+(ix-1))*dx+dx/2-lx/2,(coords[2]*(ny-2)+(iy-1))*dy+dy/2-ly/2,(coords[3]*(nz-2)+(iz-1))*dz+dz/2-lz/2
    xv,yv,zv = (coords[1]*(nx-2)+(ix-1))*dx-lx/2,(coords[2]*(ny-2)+(iy-1))*dy-ly/2,(coords[3]*(nz-2)+(iz-1))*dz-lz/2
    # Enable to set concentration for testing 
    if checkbounds(Bool,C,ix,iy,iz)
        xr = (xc-ox)
        yr = (yc-oy)
        zr = (zc-oz)
        if xr*xr + yr*yr + zr*zr < 1.0
            C[ix,iy,iz] = 0.0
        end
    end
    if checkbounds(Bool,Vx,ix,iy,iz)
        xr = (xv-ox)
        yr = (yc-oy)
        zr = (zc-oz)
        if xr*xr/r2 + yr*yr/r2 + zr*zr/r2 < 1.0
            Vx[ix,iy,iz] = 0.0
        end
    end
    if checkbounds(Bool,Vy,ix,iy,iz)
        xr = (xc-ox)
        yr = (yv-oy)
        zr = (zc-oz)
        if  xr*xr/r2 + yr*yr/r2 + zr*zr/r2 < 1.0
            Vy[ix,iy,iz] = 0.0
        end
    end
    if checkbounds(Bool,Vz,ix,iy,iz)
        xr = (xc-ox)
        yr = (yc-oy)
        zr = (zv-oz)
        if xr*xr/r2 + yr*yr/r2 + zr*zr/r2 < 1.0
            Vz[ix,iy,iz] = 0.0
        end
    end
    return
end


# Function Caller
Navier_Stokes_3D_multixpu()