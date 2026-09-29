"""
    DiscreteElement

Settle a powder layer with a soft-sphere discrete element method and write
the union of the settled spheres as an STL.

Particles are spheres. They are sampled from a truncated lognormal size
distribution, dropped into a periodic box (ballistic rain), and packed on a
flat floor by gravity. Sliding friction and rolling resistance are
velocity-regularized and carry no contact-history spring, so the force loop
is a pure function of the current state and runs unchanged on a
KernelAbstractions backend (CPU, or a GPU backend such as `CUDABackend()`).

The STL is the Marching Cubes surface of the sphere union clipped to the
periodic cell, fused to a solid build plate below `z = 0`. A CSV lists every
center and radius in meters. The same bed is also written as an HDF5 volume
fraction (`phi[x, y, z]`, plate in a separate `substrate` mask) for
LatticeBoltzmann.jl.

Bare numbers are SI base units. Unitful quantities are accepted anywhere a
scale is set (`18u"μm"`, `7950u"kg/m^3"`, `5u"mm/s"`). Particle arrays stored
on the device stay in those SI units so the force kernels remain plain floats.
"""
module DiscreteElement

using Atomix: @atomic
import Distributions
using HDF5
using FileIO
using GeometryBasics
using KernelAbstractions
using KernelAbstractions: @Const, @index, @kernel, CPU, allocate, get_backend, synchronize
using MeshIO
using Meshing
using Printf
using Random
using StaticArrays
using Statistics: quantile

include("units.jl")
include("types.jl")
include("psd.jl")
include("contact.jl")
include("kernels.jl")
include("place.jl")
include("simulate.jl")
include("export.jl")
include("hdf5bed.jl")

export Domain, Material, SimParams, Particles, PowderBed
export sample_lognormal_radii, suggest_lateral_size
export place_rain, contact_parameters, make_params, make_grid, assemble_bed
export run_dem!, simulate_powder_layer
export export_powder_bed, export_powder_hdf5, write_powder_bed, packing_summary
export @u_str

end
