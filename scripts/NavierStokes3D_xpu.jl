"""
# Navier-Stokes 3D Solver for Flow Around a Sphere on a Single xPU

This script implements a 3D solver for the incompressible Navier-Stokes equations, designed for high-performance computation
on a single xPU (GPU or multi-threaded CPU). The solver models fluid flow across a sphere in a cubic domain.

# Key Features
- **Navier-Stokes Solver**: Solves the equations using Chorin's projection method.
- **Single xPU Compatibility**: Automatically runs on a single GPU (using CUDA) or multi-threaded CPU based on configuration.
- **Spherical Obstacle**: Models a no-slip spherical object within the computational domain.
- **Boundary Conditions**: Supports zero-flux and Dirichlet boundary conditions for velocity and pressure fields.
- **Semi-Lagrangian Advection**: Includes a backtracking scheme for advection.
- **Visualization Support**: Optionally displays and saves frames of the solution (pressure, velocity) during runtime.

# Usage
- To configure the script to run on a GPU or CPU, set the constant `USE_GPU` to `true` or `false`.
- Call the main function `navier_stokes_3d_xpu(; do_vis=false, do_save=true)` to run the simulation.
- Results are optionally visualized and saved in the directory `./navier_stokes_3d_xpu_sphere_outputs`.

# Dependencies
This script requires the following Julia packages:
- `ParallelStencil` for parallel finite-difference operations.
- `CUDA` (if `USE_GPU=true`) for GPU acceleration.
- `Plots` for optional visualization.
- `MAT` for exporting results in `.mat` format.

# Notes
- Ensure all dependencies are installed and accessible in your Julia environment.
- For GPU execution, a CUDA-enabled GPU is required along with the appropriate drivers.
- The solver is not distributed and is intended for single-node, single-xPU execution.
"""

const USE_GPU = true
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3)
else
    @init_parallel_stencil(Threads, Float64, 3)
end
using LinearAlgebra, Printf
using MAT, Plots


"""
    save_array(file_name, array)

Save a given array to a binary file with the specified name.

# Arguments
- `file_name`: The desired name of the file (without extension).
- `array`: The numerical array to be saved.

# Side Effects
- Creates a `.bin` file under the directory `navier_stokes_3d_xpu_sphere_outputs`.
"""
function save_array(Aname,A)
    # Create a unique directory for this script
    base_dir = "./navier_stokes_3d_xpu_sphere_outputs"
    if !isdir(base_dir)
        mkdir(base_dir)
    end
    fname = joinpath(base_dir, string(Aname, ".bin"))
    out = open(fname, "w"); write(out, A); close(out)
end


"""
    navier_stokes_3d_xpu(; do_vis, do_save)

Iterative solver for the incompressible Navier-Stokes equation (flow across a sphere) using multiple xPU devices.

# Keyword Arguments
- `do_vis::Bool=false`: If `true`, visualizes frames of the solution (e.g., pressure, velocity magnitude) during runtime.
- `do_save::Bool=true`: If `true`, saves 3D arrays of the solution at regular intervals.

# Functionality
- Implements Chorin's projection method with explicit time-stepping.
- Computes velocity updates, divergence, and pressure corrections.
- Enforces no-slip boundary conditions for a spherical obstacle.

# Side Effects
- Saves output files to `./navier_stokes_3d_xpu_sphere_outputs` when `do_save=true`.
- Optionally displays solution frames when `do_vis=true`.

# Returns
- None (performs computations and writes output as needed).
"""
@views function navier_stokes_3d_xpu(; do_vis=false, do_save=true)
    # Physics
    ## dimensionally independent
    lz        = 1.0    # [m]
    ρ         = 1.0    # [kg/m^3]
    vin       = 1.0    # [m/s]
    ## scales
    psc       = ρ*vin^2
    ## nondimensional parameters
    Re        = 1e6    # rho*vsc*lz/μ
    Fr        = Inf    # vsc/sqrt(g*ly)
    lx_lz     = 0.5    # lx/lz
    ly_lz     = 0.5    # ly/lz
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
    μ         = 1/Re*ρ*vin*lz
    g         = 1/Fr^2*vin^2/lz
    r2        = (r_lz*lz)^2    
    
    # Numerics
    nz        = 255
    ny        = floor(Int,nz*ly_lz)
    nx        = floor(Int,nz*lx_lz)
    εit       = 1e-4
    niter     = 50*max(nx, ny, nz)
    nchk      = 1*(max(nx, ny, nz)-1)
    nvis      = 50
    nt        = 2000
    nsave     = 50
    CFLτ      = 0.9/sqrt(3) 
    CFL_visc  = 1/5.1
    CFL_adv   = 1.0 
    
    # Preprocessing
    dx,dy,dz  = lx/nx, ly/ny, lz/nz
    dt        = min(CFL_visc*dz^2*ρ/μ,CFL_adv*dz/vin)
    damp      = 2/nz
    dτ        = CFLτ*dz
    xc,yc,zc  = LinRange(-(lx-dx)/2,(lx-dx)/2,nx  ),LinRange(-(ly-dy)/2,(ly-dy)/2,ny  ), LinRange(-(lz-dz)/2,(lz-dz)/2,nz  )
    xv,yv,zv  = LinRange(-lx/2     ,lx/2     ,nx+1),LinRange(-ly/2     ,ly/2     ,ny+1), LinRange(-lz/2,     lz/2,     nz+1)
    
    # Array allocation
    Pr        = @zeros(nx  ,ny  ,nz  )
    dPrdτ     = @zeros(nx-2,ny-2,nz-2)
    τxx       = @zeros(nx  ,ny  ,nz  )
    τyy       = @zeros(nx  ,ny  ,nz  )
    τzz       = @zeros(nx  ,ny  ,nz  )
    τxy       = @zeros(nx-1,ny-1,nz-2)
    τxz       = @zeros(nx-1,ny-2,nz-1)
    τyz       = @zeros(nx-2,ny-1,nz-1)
    Vx        = @zeros(nx+1,ny  ,nz  )
    Vy        = @zeros(nx  ,ny+1,nz  )
    Vz        = @zeros(nx  ,ny  ,nz+1)
    Vx_o      = @zeros(nx+1,ny  ,nz  )
    Vy_o      = @zeros(nx  ,ny+1,nz  )
    Vz_o      = @zeros(nx  ,ny  ,nz+1)
    ∇V        = @zeros(nx  ,ny  ,nz  )
    Rp        = @zeros(nx-2,ny-2,nz-2)
    
    # Initialization
    x           = LinRange(0.5dx, lx-0.5dx,nx)
    y           = LinRange(0.5dy, ly-0.5dy,ny)
    Vprof       = Data.Array(@. 4*vin*x/lx*(1.0-x/lx) + 4*vin*y'/ly*(1.0-y'/ly))
    Vz[:,:,1]  .= Vprof
    Pr          = Data.Array([-(zc[iz] - lz/2)*ρ*g for ix=1:nx, iy=1:ny, iz=1:nz])
    # Iterative Solving
    for it = 1:nt
        err_evo = Float64[]; iter_evo = Float64[]
        # Velocity update, divergence update, sphere BC update
        @parallel update_τ!(τxx,τyy,τzz,τxy,τxz,τyz,Vx,Vy,Vz,μ,dx,dy,dz)
        @parallel predict_V!(Vx,Vy,Vz,τxx,τyy,τzz,τxy,τxz,τyz,ρ,g,dt,dx,dy,dz)
        @parallel set_sphere!(Vx,Vy,Vz,ox,oy,oz,lx,ly,lz,dx,dy,dz,r2)
        @parallel update_∇V!(∇V,Vx,Vy,Vz,dx,dy,dz)
        println("#it = $it")
        for iter = 1:niter
            # Pressure update by pseudo-transient solver
            @parallel update_dPrdτ!(Pr,dPrdτ,∇V,ρ,dt,dτ,damp,dx,dy,dz)
            @parallel update_Pr!(Pr,dPrdτ,dτ)
            set_bc_Pr!(Pr, 0.0)
            if iter % nchk == 0
                # Error computation
                @parallel compute_res!(Rp,Pr,∇V,ρ,dt,dx,dy,dz)
                err = maximum(abs.(Rp))*lz^2/psc
                push!(err_evo, err); push!(iter_evo,iter/nz)
                @printf("  #iter = %d, err = %1.3e\n", iter, err)
                if err < εit || !isfinite(err) break end
            end
        end
        # Velocity correction
        @parallel correct_V!(Vx,Vy,Vz,Pr,dt,ρ,dx,dy,dz)
        @parallel set_sphere!(Vx,Vy,Vz,ox,oy,oz,lx,ly,lz,dx,dy,dz,r2)
        set_bc_Vel!(Vx, Vy, Vz, Vprof)
        Vx_o .= Vx; Vy_o .= Vy; Vz_o .= Vz;
        # Advection scheme 
        @parallel advect!(Vx,Vx_o,Vy,Vy_o,Vz,Vz_o,dt,dx,dy,dz)
        if do_vis && it % nvis == 0
            p1=heatmap(xc,zc,Array(Pr)';aspect_ratio=1,xlims=(-lx/2,lx/2),ylims=(-ly/2,ly/2),title="Pr")
            p2=plot(iter_evo,err_evo;yscale=:log10)
            p3=heatmap(xc,zv,Array(Vz)';aspect_ratio=1,xlims=(-lx/2,lx/2),ylims=(-ly/2,ly/2),title="Vz")
            display(plot(p1,p2,p3))
        end
        if do_save && it % nsave == 0
            save_array("out_Vx_$it", convert.(Float32, Array(Vx)))
            save_array("out_Vy_$it", convert.(Float32, Array(Vy)))
            save_array("out_Vz_$it", convert.(Float32, Array(Vz)))
            save_array("out_Pr_$it", convert.(Float32, Array(Pr)))
        end
    end
    return
end


"""
    ∇V()

Macro to compute the divergence of the velocity field.

# Functionality
- Computes the divergence using finite-difference approximations:
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

# Side Effects
- Updates the velocity components with the first intermediate velocity values.
"""
@parallel function predict_V!(Vx,Vy,Vz,τxx,τyy,τzz,τxy,τxz,τyz,ρ,g,dt,dx,dy,dz)
    @inn(Vx) = @inn(Vx) + dt/ρ*(@d_xi(τxx)/dx + @d_ya(τxy)/dy  + @d_za(τxz)/dz)
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
    bc_xyV!(A, V)

Apply a zero-flux boundary condition in one xy-plane and a Dirichlet boundary condition in the other.

# Arguments
- `A`: Array representing a physical quantity.
- `V`: Array specifying the Dirichlet boundary condition values.

# Side Effects
- Updates the boundary values of `A` in-place along the xy-plane.
"""
@parallel_indices (ix, iy) function bc_xyV!(A, V)
    A[ix,   iy,   1] = V[    ix,    iy]
    A[ix,   iy, end] = A[ix, iy, end-1]
    return
end


"""
    bc_xyval!(A, val)

Apply a zero-flux boundary condition in one xy-plane and a Dirichlet boundary condition in the other.

# Arguments
- `A`: Array representing a physical quantity.
- `val`: Scalar value for the Dirichlet boundary condition.

# Side Effects
- Updates the boundary values of `A` in-place along the xy-plane.
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
    @parallel bc_yz!(Vy)
    @parallel bc_xy!(Vy)
    @parallel bc_xz!(Vz)
    @parallel bc_yz!(Vz)
    @parallel bc_xyV!(Vz, Vprof)
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
    δx,δy,δz    = dt*vxc/dx, dt*vyc/dy, dt*vzc/dz
    ix1         = clamp(floor(Int,ix-δx),1,size(A,1))
    iy1         = clamp(floor(Int,iy-δy),1,size(A,2))
    iz1         = clamp(floor(Int,iz-δz),1,size(A,3))
    ix2,iy2,iz2 = clamp(ix1+1,1,size(A,1)),clamp(iy1+1,1,size(A,2)), clamp(iz1+1,1,size(A,3))
    δx = (δx>0) - (δx%1); δy = (δy>0) - (δy%1); δz = (δz>0) - (δz%1);
    fx11        = lerp(A_o[ix1,iy1,iz1], A_o[ix2,iy1,iz1],δx)
    fx21        = lerp(A_o[ix1,iy2,iz1], A_o[ix2,iy2,iz1],δx)
    fx12        = lerp(A_o[ix1,iy1,iz2], A_o[ix2,iy1,iz2],δx)
    fx22        = lerp(A_o[ix1,iy2,iz2], A_o[ix2,iy2,iz2],δx)
    fy1         = lerp(fx11, fx21, δy)
    fy2         = lerp(fx12, fx22, δy)
    A[ix,iy,iz] = lerp(fy1,fy2,δz)
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
@parallel_indices (ix,iy,iz) function advect!(Vx,Vx_o,Vy,Vy_o,Vz,Vz_o,dt,dx,dy,dz)
    if ix > 1 && ix < size(Vx,1) && iy <= size(Vx,2) && iz <=size(Vx,3)
        vxc      = Vx_o[ix,iy,iz]
        vyc      = 0.25*(Vy_o[ix-1,iy,iz]+Vy_o[ix-1,iy+1,iz]+Vy_o[ix,iy,iz]+Vy_o[ix,iy+1,iz])
        vzc      = 0.25*(Vz_o[ix-1,iy,iz]+Vz_o[ix-1,iy,iz+1]+Vz_o[ix,iy,iz+1]+Vz_o[ix,iy,iz])
        backtrack!(Vx,Vx_o,vxc,vyc,vzc,dt,dx,dy,dz,ix,iy,iz)
    end
    if iy > 1 && iy < size(Vy,2) && ix <= size(Vy,1) && iz <= size(Vy,3)
        vxc      = 0.25*(Vx_o[ix,iy-1,iz]+Vx_o[ix+1,iy-1,iz]+Vx_o[ix,iy,iz]+Vx_o[ix+1,iy,iz])
        vyc      = Vy_o[ix,iy,iz]
        vzc      = 0.25*(Vz_o[ix,iy-1,iz]+Vz_o[ix,iy-1,iz+1]+Vz_o[ix,iy,iz]+Vz_o[ix,iy,iz+1])
        backtrack!(Vy,Vy_o,vxc,vyc,vzc,dt,dx,dy,dz,ix,iy,iz)
    end
    if iz > 1 && iz < size(Vz,3) && ix <= size(Vz,1) && iy <= size(Vz,2)
        vxc      = 0.25*(Vx_o[ix,iy,iz-1] + Vx_o[ix+1,iy,iz-1] + Vx_o[ix,iy,iz] + Vx_o[ix+1,iy,iz])
        vyc      = 0.25*(Vy_o[ix,iy,iz-1] + Vy_o[ix,iy+1,iz-1] + Vy_o[ix,iy,iz] + Vy_o[ix,iy+1,iz]) 
        vzc      = Vz_o[ix,iy,iz]
        backtrack!(Vz,Vz_o,vxc,vyc,vzc,dt,dx,dy,dz,ix,iy,iz)
    end
    return
end


"""
    set_sphere!(Vx, Vy, Vz, ox, oy, oz, lx, ly, lz, dx, dy, dz, r2)

Place a no-slip spherical object in the simulation domain.

# Arguments
- `Vx`, `Vy`, `Vz`: Velocity components (updated in-place to zero inside the sphere).
- `ox`, `oy`, `oz`: Sphere center coordinates.
- `lx`, `ly`, `lz`: Domain dimensions.
- `dx`, `dy`, `dz`: Grid spacing.
- `r2`: Square of the sphere radius.

# Side Effects
- Modifies the velocity components to impose no-slip conditions inside the sphere.
"""
@parallel_indices (ix,iy,iz) function set_sphere!(Vx,Vy,Vz,ox,oy,oz,lx,ly,lz,dx,dy,dz,r2)
    xv,yv,zv = (ix-1)*dx - lx/2, (iy-1)*dy - ly/2, (iz-1)*dz -lz/2
    xc,yc,zc = xv+dx/2, yv+dy/2 , zv+dz/2 
    if checkbounds(Bool,Vx,ix,iy,iz)
        xr = (xc-ox)
        yr = (yc-oy)
        zr = (zc-oz)
        if xr*xr/r2 + yr*yr/r2 + zr*zr/r2 < 1.0
            Vx[ix,iy,iz] = 0.0
        end
    end
    if checkbounds(Bool,Vy,ix,iy,iz)
        xr = (xc-ox)
        yr = (yc-oy)
        zr = (zc-oz)
        if  xr*xr/r2 + yr*yr/r2 + zr*zr/r2 < 1.0
            Vy[ix,iy,iz] = 0.0
        end
    end
    if checkbounds(Bool,Vz,ix,iy,iz)
        xr = (xc-ox)
        yr = (yc-oy)
        zr = (zc-oz)
        if xr*xr/r2 + yr*yr/r2 + zr*zr/r2 < 1.0
            Vz[ix,iy,iz] = 0.0
        end
    end
    return
end


# Function Caller
navier_stokes_3d_xpu()