"""
Script for 2D Heatmap Visualization and Animation of Fluid Flow Characteristics

This script performs the following operations:
1. Loads binary data representing velocity components (Vx, Vy, Vz) and pressure fields for 3D simulations.
2. Computes derived quantities such as velocity magnitude and vorticity from the loaded data.
3. Visualizes 2D heatmap slices of velocity magnitude, pressure, and vorticity at a midplane along the y-axis.
4. Saves the visualizations as PNG images for each time step.
5. Creates GIF animations for velocity magnitude, pressure, and vorticity.

Main Functions:
- `load_array`: Loads binary data into a pre-allocated array from file.
- `visualise_velocity`: Visualizes the velocity magnitude field as a 2D heatmap slice.
- `visualise_pressure`: Visualizes the pressure field as a 2D heatmap slice.
- `visualise_vorticity`: Visualizes the vorticity field as a 2D heatmap slice.

Execution:
The script loops over a specified number of time frames (`nt`), processes the data, and generates PNG images stored in respective directories (`./2D_Velocity`, `./2D_Pressure`, `./2D_Vorticity`). 
Finally, animations for each field type are created and saved as GIF files.

Dependencies:
- Plots for heatmap visualization and animation generation.
- Assumes binary input files are available in a directory with specific naming conventions.

Output:
- PNG images for each time step in structured directories (e.g., `./2D_Velocity/`, `./2D_Pressure/`, `./2D_Vorticity/`).
- GIF animations for velocity magnitude, pressure, and vorticity saved in their respective directories.

"""

using Plots, Printf

# Computes an array by averaging adjacent elements in all dimensions.
@views av1(A) = 0.5 .* (A[1:end-1] .+ A[2:end])

# Computes an array by averaging adjacent elements in the first dimension of a 3D array.
@views avx(A) = 0.5 .* (A[1:end-1, :, :] .+ A[2:end, :, :])

# Computes an array by averaging adjacent elements in the second dimension of a 3D array.
@views avy(A) = 0.5 .* (A[:, 1:end-1, :] .+ A[:, 2:end, :])

# Computes an array by averaging adjacent elements in the third dimension of a 3D array.
@views avz(A) = 0.5 .* (A[:, :, 1:end-1] .+ A[:, :, 2:end])


# Loads binary data from a file into the given array.
function load_array(Aname, A)
    fname = string(Aname, ".bin")
    fid=open(fname, "r"); read!(fid, A); close(fid)
end


# Visualizes the velocity magnitude as a heatmap slice.
function visualise_velocity(iframe = 0, vis_save = 0)
    lx, ly, lz = 0.5, 0.5, 1.0
    nz      = 506
    nx      = ny = 250
    Vx      = zeros(Float32, nx, ny, nz)
    Vy      = zeros(Float32, nx, ny, nz)
    Vz      = zeros(Float32, nx, ny, nz)
    Vmag    = zeros(Float32, nx, ny, nz)
    
    load_array("../navier_stokes_3d_multixpu_sphere_outputs/out_Vx_$(iframe)", Vx)
    load_array("../navier_stokes_3d_multixpu_sphere_outputs/out_Vy_$(iframe)", Vy)
    load_array("../navier_stokes_3d_multixpu_sphere_outputs/out_Vz_$(iframe)", Vz)
    
    Vx     .= Array(Vx)
    Vy     .= Array(Vy)
    Vz     .= Array(Vz)
    Vmag .= sqrt.(Vx .^ 2 .+ Vy .^ 2 .+ Vz .^ 2)
    xc, yc, zc = LinRange(-lx/2 ,lx/2 ,nx+1),LinRange(-ly/2,ly/2,ny+1), LinRange(-lz/2, lz/2, nz+1)
    
    p1=heatmap(xc,zc,Vmag[:, ceil(Int, ny/ 2), :]';aspect_ratio=1,xlims=(-lx/2,lx/2),ylims=(-lz/2,lz/2),title="Velocity", c=:winter, clims=(0,2.5))
    
    if !isdir("./2D_Velocity")
        mkdir("./2D_Velocity")
    end
    
    png(p1, @sprintf("./2D_Velocity/%06d.png", vis_save))
    
    return
end


# Visualizes the pressure field as a heatmap slice.
function visualise_pressure(iframe = 0, vis_save = 0)
    lx, ly, lz = 0.5, 0.5, 1.0
    nz = 506
    nx = ny = 250
    Pr  = zeros(Float32, nx, ny, nz)
    
    load_array("../navier_stokes_3d_multixpu_sphere_outputs/out_Pr_$(iframe)", Pr)
    
    xc, yc, zc = LinRange(-lx/2 ,lx/2 ,nx+1),LinRange(-ly/2,ly/2,ny+1), LinRange(-lz/2, lz/2, nz+1)
    
    p1=heatmap(xc,zc,Pr[:, ceil(Int, ny/ 2), :]';aspect_ratio=1,xlims=(-lx/2,lx/2),ylims=(-lz/2,lz/2),title="Pressure Field", c=:spring, clims=(-2,2))
    
    if !isdir("./2D_Pressure")
        mkdir("./2D_Pressure")
    end
    
    png(p1, @sprintf("./2D_Pressure/%06d.png", vis_save))
    
    return
end


# Visualizes the vorticity as a heatmap slice.
function visualise_vorticity(iframe = 0, vis_save = 0)
    lx, ly, lz = 0.5, 0.5, 1.0
    nz = 506
    nx = ny = 250
    dx,dy,dz  = lx/nx, ly/ny, lz/nz
    ω   = zeros(Float32, nx-2, ny-2, nz-2)
    ωx  = zeros(Float32, nx-2, ny-2, nz-2)
    ωy  = zeros(Float32, nx-2, ny-2, nz-2)
    ωz  = zeros(Float32, nx-2, ny-2, nz-2)
    Vx  = zeros(Float32, nx, ny, nz)
    Vy  = zeros(Float32, nx, ny, nz)
    Vz  = zeros(Float32, nx, ny, nz)
    
    load_array("../navier_stokes_3d_multixpu_sphere_outputs/out_Vx_$(iframe)", Vx)
    load_array("../navier_stokes_3d_multixpu_sphere_outputs/out_Vy_$(iframe)", Vy)
    load_array("../navier_stokes_3d_multixpu_sphere_outputs/out_Vz_$(iframe)", Vz)
    
    ωx .= avy(diff(Vz, dims = 2))[2:end-1,:,2:end-1]./dy .- avz(diff(Vy, dims = 3))[2:end-1,2:end-1,:]./dz
    ωy .= avz(diff(Vx, dims = 3))[2:end-1,2:end-1,:]./dz .- avx(diff(Vz, dims = 1))[:,2:end-1,2:end-1]./dx
    ωz .= avx(diff(Vy, dims = 1))[:,2:end-1,2:end-1]./dx .- avy(diff(Vx, dims = 2))[2:end-1,:,2:end-1]./dy
    ω  .= sqrt.(ωx .^ 2 .+ ωy .^ 2 .+ ωz .^ 2)
    xc, yc, zc = LinRange(-lx/2 + dx ,lx/2 - dx ,nx-1),LinRange(-ly/2 + dy, ly/2 - dy, ny-1), LinRange(-lz/2 + dz, lz/2 - dz, nz-1)
    p1=heatmap(xc,zc,ω[:, ceil(Int, ny/ 2), :]';aspect_ratio=1,xlims=(-lx/2,lx/2),ylims=(-lz/2,lz/2),title="Vorticity",c=:plasma, clims=(0,600))
    if !isdir("./2D_Vorticity")
        mkdir("./2D_Vorticity")
    end
    png(p1, @sprintf("./2D_Vorticity/%06d.png", vis_save))
    return
end


# Loop through frames to generate visualizations for velocity magnitude, pressure, and vorticity.
nvis    = 1
nt      = 20
frames  = 100

for it = 1:nt
    visualise_velocity(it*frames, it)
    visualise_pressure(it*frames, it)
    visualise_vorticity(it*frames, it)
end 


# Build animations from the generated images.
import Plots:Animation, buildanimation 
fnames = [@sprintf("%06d.png", k) for k in 1:nt]

## Velocity
velocity_animation = Animation("./2D_Velocity/", fnames); 
buildanimation(velocity_animation, "./2D_Velocity/2D_Velocity_Animation.gif", fps = 5, show_msg=false)  

## Pressure
pressure_animation = Animation("./2D_Pressure/", fnames); 
buildanimation(pressure_animation, "./2D_Pressure/2D_Pressure_Animation.gif", fps = 5, show_msg=false)

## Vorticity
vorticity_animation = Animation("./2D_Vorticity/", fnames); 
buildanimation(vorticity_animation, "./2D_Vorticity/2D_Vorticity_Animation.gif", fps = 5, show_msg=false)