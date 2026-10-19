# Rain a few thousand spheres into one powder layer and write an STL.
#
#   julia --project=. examples/powder_layer.jl
#   julia --project=. examples/powder_layer.jl 800
#
# x stays the length of that scaled 2:1 plate. y is half of that width,
# so the bed is a 4:1 strip and the powder sits in a thicker layer.
# Lengths below use Unitful. Bare numbers would be SI (meters, seconds, kg).
# The CSV, STL, and HDF5 file are written in meters.

using DiscreteElement

n = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 3000
outdir = joinpath(@__DIR__, "output")
mkpath(outdir)

d10 = 18u"μm"
d50 = 30u"μm"
d90 = 45u"μm"
dmin = 15u"μm"
dmax = 53u"μm"
layer_height = 50u"μm"
aspect = 12.8 / 6.4
radii = sample_lognormal_radii(n, d10, d50, d90; dmin, dmax)
side = suggest_lateral_size(radii, layer_height; packing = 0.5)
Lx = side * sqrt(aspect)
Ly = Lx / 4

simulate_powder_layer(;
    n,
    d10, d50, d90, dmin, dmax,
    layer_height,
    Lx, Ly,
    substrate_thickness = 200u"μm",
    voxels_per_diameter=6,
    stl = joinpath(outdir, "powder_bed.stl"),
    csv = joinpath(outdir, "particles.csv"),
    h5 = joinpath(outdir, "powder_bed.h5"),
)
