# Multi-xPU-3D-Navier-Stokes-Solver

[![Run tests](https://github.com/BadeaTayea/Multi-xPU-3D-Navier-Stokes-Solver/actions/workflows/CI.yml/badge.svg)](https://github.com/BadeaTayea/Multi-xPU-3D-Navier-Stokes-Solver/actions/workflows/CI.yml)

[![Running Literate.yml](https://github.com/BadeaTayea/Multi-xPU-3D-Navier-Stokes-Solver/actions/workflows/Literate.yml/badge.svg)](https://github.com/BadeaTayea/Multi-xPU-3D-Navier-Stokes-Solver/actions/workflows/Literate.yml)

![Release Workflow](https://github.com/BadeaTayea/Multi-xPU-3D-Navier-Stokes-Solver/actions/workflows/Release.yml/badge.svg)



## Repository Guide

```bash
.
├── deps
│   └── build.jl
├── docs
│   ├── 2D_Pressure_Animation.gif
│   ├── 2D_Velocity_Animation.gif
│   ├── 2D_Vorticity_Animation.gif
│   ├── 3D_Pressure_Animation.gif
│   ├── 3D_Velocity_Animation.gif
│   ├── 3D_Vorticity_Animation.gif
│   └── md
│       └── Literate.md
├── Project.toml
├── README.md
├── scripts
│   ├── Literate.jl
│   ├── NavierStokes3D_multixpu.jl
│   ├── NavierStokes3D_xpu.jl
│   ├── Project.toml
│   ├── run_multixpu.sh
│   ├── run_xpu.sh
│   └── visualization
│       ├── NavierStokes_2D_Viz.jl
│       └── NavierStokes_3D_Viz.jl
└── test
    ├── NavierStokes3D_multixpu_testing.jl
    ├── out
    │   ├── A_out.bin
    │   ├── out_Pr_10.bin
    │   ├── out_Vx_10.bin
    │   ├── out_Vy_10.bin
    │   └── out_Vz_10.bin
    ├── out_ground_truth
    │   ├── out_Pr_10.bin
    │   ├── out_Vx_10.bin
    │   ├── out_Vy_10.bin
    │   └── out_Vz_10.bin
    ├── Project.toml
    └── runtests.jl
```

## Setups
All computations for this project were conducted on the supercomputer **Piz Daint** at CSCS. Piz Daint comprises approximately 5,700 compute nodes, each equipped with an Nvidia P100 GPU (16GB PCIe). To ensure a seamless workflow for computation and visualization, a few preparatory measures were implemented. Details are outlined below:

### Computational Setup
To run Julia interactively on Piz Daint, the following steps were followed:

1. Access Daint using the termina
```bash
$ ssh daint-xc
```

2. Allocate the desired configuration (xPU, nodes, processes, time) via SLURM:
```bash
$ salloc -C'gpu' -Aclass04 -N1 -n1 --time=01:00:00
```

3. Access Compute node:
```bash
$ . $SCRATCH/../julia/daint-gpu-nocudaaware/activate
```


4. Activate a previously prepared Julia configuration:
```bash
$ juliaup
```

### Visualization Setup

Visualization scripts rely on GLMakie. GLMakie requires OpenGL, which is unavailable in headless environments such as Piz Daint. As a walk-through, we set up a virtual display for headless rendering on Piz Daint, and then added the packages and ran the scripts. Here's a quick summary of the scheme used to create run visualization scripts:

1. Set up a virtual display for headless rendering on Piz Daint:
```bash
$ Xvfb :1 -screen 0 1024x768x24 &
$ export DISPLAY=:1
```

2. Confirm that Xvfb is running using:
```bash
$ ps aux | grep Xvfb
class203 12856  0.0  0.0 2389004 43516 pts/0   Sl   10:00   0:00 Xvfb :1 -screen 0 1024x768x24
```

3. Enter julia REPL and activate a local project:
```julia
julia> using Pkg
julia> Pkg.activate()
julia> Pkg.instantiate()
```

4. Re-install the GLMakie package within the environment:
  ```julia
  Pkg.add("GLMakie")
  ```
  
5. Run the script relying on GLMakie (e.g. `./scripts/visualization/NavierStokes_3D_Viz.jl`):
```julia
julia> include("NavierStokes_3D_Viz.jl")
```

## Physical Problem 

### Governing System of Partial Differential Equations (PDEs)

The flow of an incompressible fluid around a spherical obstacle is governed by the Navier-Stokes equations, which describe the conservation of momentum and enforce incompressibility:

$$
\rho \left( \frac{\partial \mathbf{V}}{\partial t} + (\mathbf{V} \cdot \nabla) \mathbf{V} \right) = -\nabla p + \mu \nabla^2 \mathbf{V},
$$

$$
\nabla \cdot \mathbf{V} = 0,
$$

where:
- $\mathbf{V} = [u, v, w]^T$ is the velocity vector field in the $x$, $y$, and $z$ directions,
- $p$ is the pressure field,
- $\rho$ is the fluid density,
- $\mu$ is the dynamic viscosity.

The velocity field $\mathbf{V}$ must satisfy a variety of physical constraints. At the sphere's surface, the velocity is set to zero, enforcing the no-slip condition and ensuring that fluid adheres to the obstacle. Away from the sphere, the velocity evolves dynamically according to the Navier-Stokes equations. The inlet velocity is imposed with a parabolic profile, setting the initial conditions for flow entering the domain. This profile aligns with the assumption of laminar inflow and ensures smooth flow entry.

Pressure $p$, on the other hand, plays a crucial role in maintaining the incompressibility condition, $\nabla \cdot \mathbf{V} = 0$. The pressure field acts as a correction mechanism, dynamically adjusting to counteract any divergence in the velocity field. This adjustment is achieved by solving the pressure Poisson equation:

$$
\nabla^2 p = \frac{\rho}{\Delta t} \nabla \cdot \mathbf{V},
$$

where $\Delta t$ is the simulation time step.

The flow characteristics are largely determined by the Reynolds number, a dimensionless parameter defined as:

$$
Re = \frac{\rho V_{in} L}{\mu},
$$

Here, $V_{in}$ represents the characteristic inlet velocity, and $L$ is the characteristic length (domain size or obstacle diameter). For $Re = 10^6$, the flow regime is dominated by inertial forces, with viscous effects localized near the obstacle's boundary.

Rotational aspects of the flow are characterized by the vorticity field, which can be derived from the velocity field as the curl:

$$
\boldsymbol{\omega} = \nabla \times \mathbf{V},
$$

with components:

$$
\omega_x = \frac{\partial w}{\partial y} - \frac{\partial v}{\partial z}, \quad \omega_y = \frac{\partial u}{\partial z} - \frac{\partial w}{\partial x}, \quad \omega_z = \frac{\partial v}{\partial x} - \frac{\partial u}{\partial y}.
$$

The magnitude of vorticity is expressed as:

$$
|\omega| = \sqrt{\omega_x^2 + \omega_y^2 + \omega_z^2}.
$$

The vorticity magnitude is essential for visualizing and analyzing vortex structures, such as those formed in the wake of the sphere.


### Boundary Conditions

Boundary conditions define the flow behavior at the domain boundaries and the surface of the sphere. At the sphere's surface, the no-slip condition is enforced:

$$
\mathbf{V} = 0, \quad \forall \mathbf{x} \, \text{s.t.} \, \frac{(x - x_s)^2}{r^2} + \frac{(y - y_s)^2}{r^2} + \frac{(z - z_s)^2}{r^2} \leq 1,
$$

where $(x_s, y_s, z_s)$ is the sphere's center, and $r$ is its radius. This ensures that the fluid adheres to the sphere's surface, creating shear layers and wake dynamics.

At the inlet boundary ($z = 0$), a parabolic velocity profile is imposed for $w$:

$$
w(x, y, z=0) = 4 V_{in} \frac{x}{L_x} \left(1 - \frac{x}{L_x} \right) \frac{y}{L_y} \left(1 - \frac{y}{L_y} \right),
$$

where $L_x$ and $L_y$ are the domain dimensions in the $x$ and $y$ directions. This smooth inflow profile simulates laminar flow entering the domain.

At the outlet boundary ($z = L_z$), a zero-gradient Neumann condition is applied to pressure:

$$
\frac{\partial p}{\partial z} = 0,
$$

allowing the flow to exit without resistance. On the side walls, free-slip boundary conditions are applied to minimize interactions with the domain boundaries. For the tangential components, these conditions ensure:

$$
\frac{\partial u}{\partial y} = 0, \, v = 0, \, \frac{\partial w}{\partial x} = 0,
$$

on the YZ plane, and similarly for the XZ plane. These conditions prevent the generation of artificial shear forces along the walls.





## Numerical Methods and Implementation
The numerical solution of the incompressible Navier-Stokes equations is performed using Chorin's projection method, an operator-splitting approach. This method separates the velocity update into components corresponding to the physical terms in the equations—viscous, gravitational, convective, and pressure-driven effects. The code integrates multiple numerical strategies, including explicit Euler time-stepping, semi-Lagrangian advection, and a pseudo-transient pressure solver, to ensure efficient and stable computations.

#### Intermediate Velocity Update

The intermediate velocity $\mathbf{V}^*$ is calculated by incorporating viscous and gravitational forces using explicit Euler time-stepping:

$$
\frac{\mathbf{V}^* - \mathbf{V}^n}{\Delta t} = \mu \nabla^2 \mathbf{V}^n - \rho \mathbf{g},
$$

where $\mathbf{g}$ is the gravitational acceleration. This update is implemented in the `predict_V!` function, which updates the velocity components `Vx`, `Vy`, and `Vz` in place. The function utilizes stress tensor components (`τxx`, `τyy`, `τzz`, etc.), computed in `update_τ!`, to represent the viscous forces. These components are calculated using finite-difference operators (`@d_xa`, `@d_ya`, `@d_za`) over the velocity arrays, and the gravitational term is added explicitly.

##### Convective Term and Semi-Lagrangian Advection

The convective term, accounting for momentum transport due to the flow, is handled using a semi-Lagrangian approach. This method tracks fluid parcels backward in time along streamlines and interpolates their values. The backtracking step is described by:

$$
\mathbf{X}_{\text{backtracked}} = \mathbf{X} - \mathbf{V} \Delta t,
$$

where $\mathbf{X}_{\text{backtracked}}$ is the estimated position of a parcel. The `advect!` function performs this operation on the velocity fields using the `backtrack!` function, which calculates source positions and interpolates values using the `lerp` function. For example, `Vx`, `Vy`, and `Vz` are updated by iterating over all grid points and applying this interpolation.

##### Pressure Correction and Velocity Divergence-Free Constraint

To ensure incompressibility, the velocity field is corrected using the pressure gradient:

$$
\frac{\mathbf{V}^{n+1} - \mathbf{V}^*}{\Delta t} = -\frac{1}{\rho} \nabla p^{n+1}.
$$

The correction step is implemented in the `correct_V!` function, which updates the velocity fields `Vx`, `Vy`, and `Vz` using finite-difference operators (`@d_xi`, `@d_yi`, `@d_zi`) applied to the pressure array `Pr`. The pressure gradient enforces the divergence-free condition, ensuring mass conservation.

The pressure at iteration $(n+1)$ is obtained by solving the Poisson equation:

$$
\nabla^2 p^{n+1} = \frac{\rho}{\Delta t} \nabla \cdot \mathbf{V}^*.
$$

This equation is solved iteratively using a pseudo-transient approach. The divergence of the velocity field is first calculated in `update_∇V!`, and the pseudo-time derivative of pressure is computed in `update_dPrdτ!`. The pressure is then updated in `update_Pr!` until convergence.

##### Boundary Conditions

Boundary conditions are enforced for both velocity and pressure fields at each timestep. The function `set_bc_Vel!` applies boundary conditions for `Vx`, `Vy`, and `Vz`. For example:
- The inlet boundary (`z = 0`) is set with a parabolic velocity profile stored in `Vprof` (implemented via `bc_zV!`):

  $$ 
  w(x, y, z = 0) = 4 V_{in} \frac{x}{L_x} \left(1 - \frac{x}{L_x} \right) \frac{y}{L_y} \left(1 - \frac{y}{L_y} \right).
  $$
  
- No-slip conditions at the sphere surface are applied in `set_sphere_multixpu!`, which zeroes the velocity components `Vx`, `Vy`, and `Vz` inside and on the surface of the sphere.
- Free-slip boundary conditions for velocity and zero-gradient Neumann conditions for pressure are enforced on side walls using functions like `bc_xz!`, `bc_yz!`, and `bc_xy!`.

Pressure boundary conditions are imposed via `set_bc_Pr!`. For instance, a Dirichlet condition sets `Pr = 0` at the outlet, while zero-gradient Neumann conditions are applied elsewhere.

##### Stability and Numerical Resolution

Stability is maintained by constraining the timestep $\Delta t$ based on viscous and advective criteria:

$$
\Delta t = \min \left( \frac{\rho \Delta z^2}{\mu}, \frac{\Delta z}{V_{in}} \right),
$$

where $\Delta z$ is the grid spacing. This ensures accurate resolution of both small-scale viscous interactions and large-scale advection dynamics. 

##### Additional Numerical Considerations

- **Halo Updates**: Functions such as `update_halo!` manage data communication between grid partitions, ensuring boundary values are updated in multi-node setups.
- **Visualization**: Arrays for velocity (`Vx_v`, `Vy_v`, `Vz_v`) and pressure (`Pr_v`) are gathered and saved periodically for post-processing.
- **Divergence Computation**: The function `update_∇V!` computes the velocity divergence `∇V`, which is critical for pressure correction and diagnosing flow incompressibility.






## xPU Computing
### xPU Implementation 
### Multi-xPU Implementation

















## Visualization: Velocity, Vorticity, and Pressure Fields

### Cross-Sectional Evolution in 2D 

<table>
  <tr>
    <td align="center">
      <strong>Velocity Field</strong><br>
      <img src="docs/2D_Velocity_Animation.gif" alt="2D Velocity Animation" width="300">
    </td>
    <td align="center">
      <strong>Vorticity Field</strong><br>
      <img src="docs/2D_Vorticity_Animation.gif" alt="2D Vorticity Animation" width="300">
    </td>
    <td align="center">
      <strong>Pressure Field</strong><br>
      <img src="docs/2D_Pressure_Animation.gif" alt="2D Pressure Animation" width="300">
    </td>
  </tr>
  <tr>
    <td align="center">
      <strong>Fig. 1:</strong> 2D Velocity Field - Evolution of velocity magnitude.
    </td>
    <td align="center">
      <strong>Fig. 2:</strong> 2D Vorticity Field - Evolution of rotational flow.
    </td>
    <td align="center">
      <strong>Fig. 3:</strong> 2D Pressure Field - Evolution of pressure distribution.
    </td>
  </tr>
</table>

### Evolution in 3D 

<table>
  <tr>
    <td align="center">
      <strong>Velocity Field</strong><br>
      <img src="docs/3D_Velocity_Animation.gif" alt="3D Velocity Animation" width="300">
    </td>
    <td align="center">
      <strong>Vorticity Field</strong><br>
      <img src="docs/3D_Vorticity_Animation.gif" alt="3D Vorticity Animation" width="300">
    </td>
    <td align="center">
      <strong>Pressure Field</strong><br>
      <img src="docs/3D_Pressure_Animation.gif" alt="3D Pressure Animation" width="300">
    </td>
  </tr>
  <tr>
    <td align="center">
      <strong>Fig. 4:</strong> 3D Velocity Field - Evolution of velocity magnitude.
    </td>
    <td align="center">
      <strong>Fig. 5:</strong> 3D Vorticity Field - Evolution of rotational flow structures.
    </td>
    <td align="center">
      <strong>Fig. 6:</strong> 3D Pressure Field - Evolution of pressure distribution in 3D.
    </td>
  </tr>
</table>

## Resources
