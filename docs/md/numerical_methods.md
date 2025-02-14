### Numerical Methods and Implementation
The numerical solution of the incompressible Navier-Stokes equations is performed using **Chorin's projection method**, which splits the velocity update into separate steps handling **viscous, gravitational, convective, and pressure-driven effects**. The solver integrates:
- **Explicit Euler time-stepping** for the intermediate velocity update.
- **Semi-Lagrangian advection** for convective transport.
- **A pseudo-transient pressure solver** to enforce incompressibility.

#### Intermediate Velocity Update

The intermediate velocity $\mathbf{V}^*$ is calculated by incorporating viscous and gravitational forces using explicit Euler time-stepping:

$$
\frac{\mathbf{V}^* - \mathbf{V}^n}{\Delta t} = \mu \nabla^2 \mathbf{V}^n - \rho \mathbf{g},
$$

where $\mathbf{g}$ is the gravitational acceleration. This update is implemented in the `predict_V!` function, which updates the velocity components `Vx`, `Vy`, and `Vz` in place:

```julia
@parallel function predict_V!(Vx, Vy, Vz, τxx, τyy, τzz, τxy, τxz, τyz, ρ, g, dt, dx, dy, dz)
    @inn(Vx) = @inn(Vx) + dt/ρ * (@d_xi(τxx)/dx + @d_ya(τxy)/dy + @d_za(τxz)/dz)
    @inn(Vy) = @inn(Vy) + dt/ρ * (@d_yi(τyy)/dy + @d_xa(τxy)/dx + @d_za(τyz)/dz)
    @inn(Vz) = @inn(Vz) + dt/ρ * (@d_zi(τzz)/dz + @d_xa(τxz)/dx + @d_ya(τyz)/dy - ρ*g)
end
```

The function utilizes stress tensor components (`τxx`, `τyy`, `τzz`, etc.), computed in `update_τ!`, to represent the viscous forces:

$$
\tau_{ij} = \mu \left( \frac{\partial u_i}{\partial x_j} + \frac{\partial u_j}{\partial x_i} \right) - \frac{2}{3} \mu (\nabla \cdot \mathbf{V}) \delta_{ij}.
$$

These components are calculated using finite-difference operators (`@d_xa`, `@d_ya`, `@d_za`) over the velocity arrays, and the gravitational term is added explicitly:

```julia
@parallel function update_τ!(τxx, τyy, τzz, τxy, τxz, τyz, Vx, Vy, Vz, μ, dx, dy, dz)
    @all(τxx) = 2μ * (@d_xa(Vx)/dx - @∇V()/3.0)  
    @all(τyy) = 2μ * (@d_ya(Vy)/dy - @∇V()/3.0)
    @all(τzz) = 2μ * (@d_za(Vz)/dz - @∇V()/3.0)
    @all(τxy) =  μ * (@d_yi(Vx)/dy + @d_xi(Vy)/dx)
    @all(τxz) =  μ * (@d_zi(Vx)/dz + @d_xi(Vz)/dx)
    @all(τyz) =  μ * (@d_zi(Vy)/dz + @d_yi(Vz)/dy)
end
```


### Divergence Calculation and Incompressibility Enforcement
A key requirement for solving incompressible flows is enforcing the **zero divergence constraint**:

$$
\nabla \cdot \mathbf{V} = 0.
$$

Before solving the pressure Poisson equation, the solver first computes the **velocity divergence** $\nabla \cdot \mathbf{V}^*$, which is stored in the array `∇V`. This step is implemented in `update_∇V!`:

```julia
@parallel function update_∇V!(∇V, Vx, Vy, Vz, dx, dy, dz)
    @all(∇V) = @d_xa(Vx)/dx + @d_ya(Vy)/dy + @d_za(Vz)/dz
end
```
- **Finite-difference operators** (`@d_xa`, `@d_ya`, `@d_za`) compute the divergence in the respective spatial directions.
- The divergence is stored in `∇V`, which is later used in the **pressure correction step**.

By evaluating `∇V` before solving for pressure, the solver ensures that corrections applied to velocity eliminate divergence, enforcing incompressibility.


#### Convective Term and Semi-Lagrangian Advection

The convective term, responsible for momentum transport due to the flow, is handled using a **semi-Lagrangian approach**. Instead of solving the advection term explicitly, this method traces fluid parcels backward in time and interpolates their values:

$$
\mathbf{X}_{\text{backtracked}} = \mathbf{X} - \mathbf{V} \Delta t.
$$

where $\mathbf{X}_{\text{backtracked}}$ is the estimated position of a parcel. The backtracking and interpolation process is implemented in `advect!` by calling `backtrack!`, which:
1. Computes the source position** using velocity at the grid point.
2. Interpolates values from the nearest neighbors using `lerp`.

```julia
@inline function backtrack!(A, A_o, vxc, vyc, vzc, dt, dx, dy, dz, ix, iy, iz)
    δx, δy, δz = dt * vxc / dx, dt * vyc / dy, dt * vzc / dz
    ix1 = clamp(floor(Int, ix - δx), 1, size(A,1))
    iy1 = clamp(floor(Int, iy - δy), 1, size(A,2))
    iz1 = clamp(floor(Int, iz - δz), 1, size(A,3))
    A[ix, iy, iz] = lerp(A_o[ix1, iy1, iz1], A_o[ix1+1, iy1+1, iz1+1], δx)
end
```

For each velocity component (`Vx`, `Vy`, `Vz`), this interpolation is applied **along the flow direction**:

```julia
@parallel_indices (ix, iy, iz) function advect!(Vx, Vx_o, Vy, Vy_o, Vz, Vz_o, C, C_o, dt, dx, dy, dz)
    if ix > 1 && ix < size(Vx,1)
        vxc = Vx_o[ix, iy, iz]
        vyc = 0.25 * (Vy_o[ix-1, iy, iz] + Vy_o[ix-1, iy+1, iz] + Vy_o[ix, iy, iz] + Vy_o[ix, iy+1, iz])
        vzc = 0.25 * (Vz_o[ix-1, iy, iz] + Vz_o[ix-1, iy, iz+1] + Vz_o[ix, iy, iz] + Vz_o[ix, iy, iz+1])
        backtrack!(Vx, Vx_o, vxc, vyc, vzc, dt, dx, dy, dz, ix, iy, iz)
    end
end
```

#### Pressure Correction and Velocity Divergence-Free Constraint

To ensure incompressibility, the velocity field is corrected using the pressure gradient:

$$
\frac{\mathbf{V}^{n+1} - \mathbf{V}^*}{\Delta t} = -\frac{1}{\rho} \nabla p^{n+1}.
$$

The correction step is implemented in the `correct_V!` function, which updates the velocity fields `Vx`, `Vy`, and `Vz` using finite-difference operators (`@d_xi`, `@d_yi`, `@d_zi`) applied to the pressure array `Pr`: 

```julia
@parallel function correct_V!(Vx, Vy, Vz, Pr, dt, ρ, dx, dy, dz)
    @inn(Vx) = @inn(Vx) - dt/ρ * @d_xi(Pr)/dx
    @inn(Vy) = @inn(Vy) - dt/ρ * @d_yi(Pr)/dy
    @inn(Vz) = @inn(Vz) - dt/ρ * @d_zi(Pr)/dz
end
```
The pressure gradient enforces the divergence-free condition, ensuring mass conservation.

The pressure at iteration $(n+1)$ is obtained by solving the Poisson equation:

$$
\nabla^2 p^{n+1} = \frac{\rho}{\Delta t} \nabla \cdot \mathbf{V}^*.
$$

This equation is solved iteratively using a pseudo-transient approach. The divergence of the velocity field is first calculated in `update_∇V!`, and the pseudo-time derivative of pressure is computed in `update_dPrdτ!`. The pressure is then updated in `update_Pr!` until convergence:

```julia
@parallel function update_Pr!(Pr, dPrdτ, dτ)
    @inn(Pr) = @inn(Pr) + dτ * @all(dPrdτ)
end
```

#### Boundary Conditions

Boundary conditions are enforced for both velocity and pressure fields at each timestep. The function `set_bc_Vel!` applies boundary conditions for `Vx`, `Vy`, and `Vz`. For example:
- The inlet boundary (`z = 0`) is set with a parabolic velocity profile stored in `Vprof` (implemented via `bc_zV!`):

$$ 
w(x, y, z = 0) = 4 V_{in} \frac{x}{L_x} \left(1 - \frac{x}{L_x} \right) \frac{y}{L_y} \left(1 - \frac{y}{L_y} \right).
$$

Implemented in `bc_zV!`:

```julia
@parallel_indices (ix, iy) function bc_zV!(A, V)
    A[ix, iy, 1] = V[ix, iy]
end
```
  
- No-slip conditions at the sphere surface are applied in `set_sphere_multixpu!`, which zeroes the velocity components `Vx`, `Vy`, and `Vz` inside and on the surface of the sphere:

```julia
@parallel_indices (ix, iy, iz) function set_sphere_multixpu!(C, Vx, Vy, Vz, ox, oy, oz, lx, ly, lz, dx, dy, dz, r2)
    if (xc - ox)^2 + (yc - oy)^2 + (zc - oz)^2 < r2
        Vx[ix, iy, iz] = 0.0
        Vy[ix, iy, iz] = 0.0
        Vz[ix, iy, iz] = 0.0
    end
end
```
- Free-slip boundary conditions for velocity and zero-gradient Neumann conditions for pressure are enforced on side walls using functions like `bc_xz!`, `bc_yz!`, and `bc_xy!`.

Pressure boundary conditions are imposed via `set_bc_Pr!`. For instance, a Dirichlet condition sets `Pr = 0` at the outlet, while zero-gradient Neumann conditions are applied elsewhere.

#### Stability and Numerical Resolution

Stability is maintained by constraining the timestep $\Delta t$ based on viscous and advective criteria:

$$
\Delta t = \min \left( \frac{\rho \Delta z^2}{\mu}, \frac{\Delta z}{V_{in}} \right),
$$

where $\Delta z$ is the grid spacing. This ensures accurate resolution of both small-scale viscous interactions and large-scale advection dynamics. 
