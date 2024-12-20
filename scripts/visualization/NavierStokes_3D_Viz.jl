"""
Script for 3D Visualization and Animation of Fluid Flow Characteristics

This script performs the following tasks:
1. Loads binary data representing velocity components (Vx, Vy, Vz) and pressure fields from sorted files.
2. Computes derived quantities such as velocity magnitude and vorticity from the loaded data.
3. Generates 3D visualizations of velocity magnitude, pressure, and vorticity using GLMakie.
4. Saves the visualizations as PNG images in structured directories.
5. Creates GIF animations for each visualization type (velocity, pressure, vorticity) using Plots.

Main Functions:
- `load_array`: Loads binary data from a file into a pre-allocated array.
- `get_sorted_filenames`: Retrieves and sorts file paths based on a specified prefix and suffix.
- `visualise_velocity`: Visualizes the velocity magnitude field in 3D.
- `visualise_pressure`: Visualizes the pressure field in 3D.
- `visualise_vorticity`: Visualizes the vorticity field in 3D.

Execution:
The script processes a sequence of time frames (`nt`) and generates visualizations and animations for each time step.
Directory paths and filenames are dynamically managed for sorting and organization.

Dependencies:
- GLMakie for 3D plotting.
- Plots for animation generation.
- Assumes binary input files are located in a directory with specific naming conventions.

Output:
- PNG files for 3D visualizations, stored in corresponding directories (e.g., `./3D_Velocity/`, `./3D_Pressure/`, `./3D_Vorticity/`).
- GIF animations for velocity, pressure, and vorticity, stored in respective directories.

"""

using GLMakie, Printf

# Compute an array by averaging adjacent elements in all dimensions.
@views av1(A) = 0.5 .* (A[1:end-1] .+ A[2:end])

# Compute an array by averaging adjacent elements in the first dimension of a 3D array.
@views avx(A) = 0.5 .* (A[1:end-1, :, :] .+ A[2:end, :, :])

# Compute an array by averaging adjacent elements in the second dimension of a 3D array.
@views avy(A) = 0.5 .* (A[:, 1:end-1, :] .+ A[:, 2:end, :])

# Compute an array by averaging adjacent elements in the third dimension of a 3D array.
@views avz(A) = 0.5 .* (A[:, :, 1:end-1] .+ A[:, :, 2:end])

# Loads binary data from a file into the given array.
function load_array(fname, A)
    fid = open(fname, "r")
    read!(fid, A)
    close(fid)
end

# Sort filenames
function get_sorted_filenames(directory, prefix, suffix)
    files = readdir(directory)
    println("Files in directory: ", files)  # Debug: Print all files

    # Filter files containing the prefix and suffix
    matches = filter(file -> occursin(prefix, file) && occursin(suffix, file), files)
    println("Filtered matches: ", matches)  # Debug: Print matching files

    # Correctly interpolate the prefix and suffix into the regex
    regex = r"$prefix(\d+)$suffix"
    sorted_files = sort(
        [joinpath(directory, file) for file in matches if match(Regex("$prefix(\\d+)$suffix"), file) !== nothing],
        by = x -> parse(Int, match(Regex("$prefix(\\d+)$suffix"), basename(x)).captures[1])
    )
    println("Sorted files: ", sorted_files)  # Debug: Print sorted files
    return sorted_files
end


# Visualize the velocity
function visualise_velocity(vx_files, vy_files, vz_files, iframe, vis_save)
    lx, ly, lz = 0.5, 0.5, 1.0
    nz = 506
    nx = ny = 250
    Vx = zeros(Float32, nx, ny, nz)
    Vy = zeros(Float32, nx, ny, nz)
    Vz = zeros(Float32, nx, ny, nz)
    Vmag = zeros(Float32, nx, ny, nz)

    # Load data from sorted files
    load_array(vx_files[iframe], Vx)
    load_array(vy_files[iframe], Vy)
    load_array(vz_files[iframe], Vz)

    Vx .= Array(Vx)
    Vy .= Array(Vy)
    Vz .= Array(Vz)
    Vmag .= sqrt.(Vx .^ 2 .+ Vy .^ 2 .+ Vz .^ 2)

    # Define grid
    xc, yc, zc = LinRange(-lx/2, lx/2, nx+1), LinRange(-ly/2, ly/2, ny+1), LinRange(-lz/2, lz/2, nz+1)

    # Create 3D figure
    fig = Figure(resolution=(1600, 1000), fontsize=24)
    ax = Axis3(fig[1, 1]; aspect=(0.5, 0.5, 1.0), title="Velocity Magnitude", xlabel="lx", ylabel="ly", zlabel="lz")
    contour!(ax, xc, yc, zc, Vmag; alpha=0.05, colormap=:winter)

    # Save the visualization
    if !isdir("./3D_Velocity")
        mkdir("./3D_Velocity")
    end
    save(@sprintf("./3D_Velocity/%06d.png", vis_save), fig)
end


# Visualize the pressure field
function visualise_pressure(pressure_files, iframe, vis_save)
    lx, ly, lz = 0.5, 0.5, 1.0
    nz = 506
    nx = ny = 250
    Pr = zeros(Float32, nx, ny, nz)

    # Load data from sorted files
    load_array(pressure_files[iframe], Pr)

    # Define grid
    xc, yc, zc = LinRange(-lx/2, lx/2, nx+1), LinRange(-ly/2, ly/2, ny+1), LinRange(-lz/2, lz/2, nz+1)

    # Create 3D figure
    fig = Figure(resolution=(1600, 1000), fontsize=24)
    ax = Axis3(fig[1, 1]; aspect=(0.5, 0.5, 1.0), title="Pressure", xlabel="lx", ylabel="ly", zlabel="lz")
    contour!(ax, xc, yc, zc, Pr; alpha=0.05, colormap=:spring)

    # Save the visualization
    if !isdir("./3D_Pressure")
        mkdir("./3D_Pressure")
    end
    save(@sprintf("./3D_Pressure/%06d.png", vis_save), fig)
end


# Visualize the vorticity
function visualise_vorticity(vx_files, vy_files, vz_files, iframe, vis_save)
    lx, ly, lz = 0.5, 0.5, 1.0
    nz = 506
    nx = ny = 250
    dx, dy, dz = lx/nx, ly/ny, lz/nz
    ω = zeros(Float32, nx-2, ny-2, nz-2)
    ωx = zeros(Float32, nx-2, ny-2, nz-2)
    ωy = zeros(Float32, nx-2, ny-2, nz-2)
    ωz = zeros(Float32, nx-2, ny-2, nz-2)
    Vx = zeros(Float32, nx, ny, nz)
    Vy = zeros(Float32, nx, ny, nz)
    Vz = zeros(Float32, nx, ny, nz)

    # Load data from sorted files
    load_array(vx_files[iframe], Vx)
    load_array(vy_files[iframe], Vy)
    load_array(vz_files[iframe], Vz)

    # Compute vorticity components
    ωx .= avy(diff(Vz, dims=2))[2:end-1, :, 2:end-1] ./ dy .- avz(diff(Vy, dims=3))[2:end-1, 2:end-1, :] ./ dz
    ωy .= avz(diff(Vx, dims=3))[2:end-1, 2:end-1, :] ./ dz .- avx(diff(Vz, dims=1))[:, 2:end-1, 2:end-1] ./ dx
    ωz .= avx(diff(Vy, dims=1))[:, 2:end-1, 2:end-1] ./ dx .- avy(diff(Vx, dims=2))[2:end-1, :, 2:end-1] ./ dy
    ω .= sqrt.(ωx .^ 2 .+ ωy .^ 2 .+ ωz .^ 2)

    # Define grid
    xc, yc, zc = LinRange(-lx/2, lx/2, nx+1), LinRange(-ly/2, ly/2, ny+1), LinRange(-lz/2, lz/2, nz+1)

    # Create 3D figure
    fig = Figure(resolution=(1600, 1000), fontsize=24)
    ax = Axis3(fig[1, 1]; aspect=(0.5, 0.5, 1.0), title="Vorticity", xlabel="lx", ylabel="ly", zlabel="lz")
    contour!(ax, xc, yc, zc, ω; alpha=0.05, colormap=:plasma)

    # Save the visualization
    if !isdir("./3D_Vorticity")
        mkdir("./3D_Vorticity")
    end
    save(@sprintf("./3D_Vorticity/%06d.png", vis_save), fig)
end


# Main logic
pressure_files = get_sorted_filenames("../navier_stokes_3d_multixpu_sphere_outputs", "out_Pr_", ".bin")
vx_files = get_sorted_filenames("../navier_stokes_3d_multixpu_sphere_outputs", "out_Vx_", ".bin")
vy_files = get_sorted_filenames("../navier_stokes_3d_multixpu_sphere_outputs", "out_Vy_", ".bin")
vz_files = get_sorted_filenames("../navier_stokes_3d_multixpu_sphere_outputs", "out_Vz_", ".bin")

println("Pressure files: ", pressure_files)
println("Vx files: ", vx_files)
println("Vy files: ", vy_files)
println("Vz files: ", vz_files)

nvis = 1
nt = 20

# Generate visualizations for each time frame
for it in 1:nt
    visualise_velocity(vx_files, vy_files, vz_files, it, it)
    visualise_pressure(pressure_files, it, it)
    visualise_vorticity(vx_files, vy_files, vz_files, it, it)
end

# Build animations
import Plots: Animation, buildanimation
fnames = [@sprintf("%06d.png", k) for k in 1:nt]

## Velocity
velocity_animation = Animation("./3D_Velocity", fnames)
buildanimation(velocity_animation, "./3D_Velocity/3D_Velocity_Animation.gif", fps=5, show_msg=false)

## Pressure
pressure_animation = Animation("./3D_Pressure", fnames)
buildanimation(pressure_animation, "./3D_Pressure/3D_Pressure_Animation.gif", fps=5, show_msg=false)

## Vorticity
vorticity_animation = Animation("./3D_Vorticity", fnames)
buildanimation(vorticity_animation, "./3D_Vorticity/3D_Vorticity_Animation.gif", fps=5, show_msg=false)
