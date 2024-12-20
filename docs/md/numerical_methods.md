### Numerical Methods and Implementation
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

