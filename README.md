# DiscreteElement

[![Build Status](https://github.com/albers-btu/DiscreteElement.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/albers-btu/DiscreteElement.jl/actions/workflows/CI.yml?query=branch%3Amain)

Settle a layer of spherical powder and write it as an STL. The intended use is the starting powder bed of a laser powder bed fusion model.

Particles are drawn from a truncated lognormal (D10, D50, D90), dropped in a dilute cloud, and packed by gravity onto a flat floor. The cell is periodic in x and y. After the bed is quiet, Marching Cubes extracts the surface of the sphere union clipped to that cell, and a CSV lists every center and radius.

## Packages

The contact loop lives in this package. There is no maintained Julia DEM that already does a 3D frictional powder bed, and the force kernel has to be the same code on CPU and GPU.

| Piece | Package |
| --- | --- |
| SI units on inputs and reported scales | [Unitful.jl](https://github.com/JuliaPhysics/Unitful.jl) |
| Particle size distribution | [Distributions.jl](https://github.com/JuliaStats/Distributions.jl) |
| Force, binning, integration, signed-distance grid | [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) with [Atomix.jl](https://github.com/JuliaConcurrent/Atomix.jl) cell-list slots |
| Marching Cubes | [Meshing.jl](https://github.com/JuliaGeometry/Meshing.jl) |
| Mesh type and STL | [GeometryBasics.jl](https://github.com/JuliaGeometry/GeometryBasics.jl), [MeshIO.jl](https://github.com/JuliaIO/MeshIO.jl), [FileIO.jl](https://github.com/JuliaIO/FileIO.jl) |

Neighbor search is a fixed-capacity cell list written against KernelAbstractions, not CellListMap.jl, so the CPU run and a later GPU run share one implementation.

## Run

```julia
using DiscreteElement

simulate_powder_layer(;
    d10 = 18u"μm",
    d50 = 30u"μm",
    d90 = 45u"μm",
    dmin = 15u"μm",
    dmax = 53u"μm",
    substrate_thickness = 200u"μm",
    material = Material(; density = 7.95u"g/cm^3", gravity = 9.81u"m/s^2"),
)
```

Bare numbers are SI base units, so `30e-6` is 30 µm and `7950` is `7950 kg/m^3`. `result.height`, `result.time`, `result.kn`, and `result.dt` come back as Unitful quantities. The CSV and the STL stay in meters.

Or from a shell, with an optional particle count:

```bash
julia --project=. examples/powder_layer.jl
julia --project=. examples/powder_layer.jl 800
```

Outputs are in meters. `particles.csv` has columns `id,x,y,z,radius` and a header comment with the periodic side lengths and the substrate thickness. The STL is a binary file: the sphere union clipped to the cell, fused to a solid build plate below `z = 0`. The default plate is three median diameters thick. Pass `substrate_thickness = 0` to omit it, or a depth in meters to set it. Flat caps close spheres that cross a periodic side.

`powder_bed.h5` is the same bed for LatticeBoltzmann.jl. `phi` is a `Float32` array indexed `phi[x, y, z]`: the metal volume fraction in each cell. A cell that is only partly covered keeps that fraction, and overlapping spheres are summed and then clamped to 1. The build plate is a separate `UInt8` mask named `substrate` and is not written into `phi`. Cell edges and the minimum corner of cell `(1,1,1)` are file attributes, in metres. Pass `h5 = nothing` to skip that file.

A GPU run uses the same kernels. After `using CUDA`:

```julia
using CUDA
simulate_powder_layer(; backend = CUDABackend())
```

## Contact model

Normal force is a linear spring-dashpot whose damping matches the coefficient of restitution. Sliding friction is a tangential dashpot capped by `μ |F_n|`. Rolling and spin resistance is a viscous torque capped by `μ_r R_eff |F_n|`. Neither friction nor rolling resistance stores a tangential spring, so a step does not need a contact history.

The spring is deliberately soft. Its stiffness is chosen so a wall impact at the rain's fall speed overlaps the smallest particle by `overlap_fraction` (default 8%) of its radius. Once the bed is at rest, overlaps shrink to the tiny value that balances a particle's weight, and the exported spheres use the nominal radii at the settled centers.

The floor is `z = 0`. The default solid is 316L (`density = 7950`), with `restitution = 0.2`, `mu_slide = 0.3`, and `mu_roll = 0.05`.
