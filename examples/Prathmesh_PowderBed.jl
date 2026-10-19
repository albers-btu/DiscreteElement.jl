# Single-track LPBF powder strip.
#
#   julia --project=. examples/Prathmesh_PowderBed.jl
#   julia --project=. examples/Prathmesh_PowderBed.jl 800
#
# x is the track direction, 1.5 mm. y is 0.5 mm, a few melt-pool
# widths, so one track has powder on both sides. The particle count is
# chosen so the settled free surface is about 50 µm. Pass a count to override.
#
# D10 = 25 µm and D90 = 35 µm. D50 is their geometric mean, so the lognormal
# is symmetric. Cutoffs are 20 µm and 45 µm, about three standard deviations
# out. The build plate is 200 µm, as in examples/powder_layer.jl.
# Lengths use Unitful. The CSV, STL, and HDF5 file are written in meters.

using DiscreteElement
using Unitful: ustrip

Lx = 1.5u"mm"
Ly = 0.5u"mm"

d10 = 25u"μm"
d90 = 35u"μm"
d50 = sqrt(d10 * d90)
dmin = 20u"μm"
dmax = 45u"μm"
layer_height = 50u"μm"
# Solid fraction of a 50 µm slab. At 0.50 the free surface settles near 54 µm.
# 0.421 settles that surface at about 51 µm, the experimental layer thickness.
packing = 0.421

if length(ARGS) >= 1
    n = parse(Int, ARGS[1])
else
    probe_n = 4000
    probe = sample_lognormal_radii(probe_n, d10, d50, d90; dmin, dmax)
    side = suggest_lateral_size(probe, layer_height; packing)
    n = round(Int, probe_n * ustrip(u"m^2", Lx * Ly) / ustrip(u"m^2", side * side))
end

outdir = joinpath(@__DIR__, "output", "Prathmesh_PowderBed")
mkpath(outdir)

simulate_powder_layer(;
    n,
    d10, d50, d90, dmin, dmax,
    layer_height,
    Lx, Ly,
    packing_guess = packing,
    substrate_thickness = 200u"μm",
    voxels_per_diameter = 6,
    stl = joinpath(outdir, "powder_bed.stl"),
    csv = joinpath(outdir, "particles.csv"),
    h5 = joinpath(outdir, "powder_bed.h5"),
)
