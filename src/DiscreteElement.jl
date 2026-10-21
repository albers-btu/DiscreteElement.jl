module DiscreteElement

using Atomix: @atomic
import Distributions
using KernelAbstractions
using KernelAbstractions: @Const, @index, @kernel, CPU, allocate, get_backend, synchronize
using Printf
using Random
using StaticArrays
using Statistics: quantile, median
using Unitful
using CUDA
import HDF5
using Unitful
import Meshing


#using FileIO
#using GeometryBasics
#using MeshIO


include("units.jl")
include("model.jl")
include("particle.jl")
include("grid.jl")
include("kernels.jl")
include("force.jl")
include("velocity.jl")
include("place.jl")
include("psd.jl")
include("simulate.jl")
include("export/hdf5.jl")
include("export/stl.jl")
include("export/mp4.jl")
include("misc.jl")

export LinearSpringDashpot
export ParticleArrays, sample_lognormal_radii, @u_str
export place_particles!
export Force, compute_forces!
export half_kick!, kick_drift!, keep_in_box!, maximum_speed
export CellGrid
export simulate_n_particles
export hdf5, stl, mp4
export cut_powder_layer, estimate_n_particles

end
