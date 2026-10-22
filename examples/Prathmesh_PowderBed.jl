using DiscreteElement
using Unitful: ustrip


using DiscreteElement
using Dates
using Random
using Unitful: ustrip
using StaticArrays
using CUDA

using CairoMakie
using GLMakie

timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
outdir = joinpath(
    dirname(@__DIR__), 
    "output", 
    "Pratmesh_PowderBed_" * timestamp)
mkpath(outdir)

Lx = 1.5u"mm"
Ly = 0.5u"mm"
Lz = 0.5u"mm"
domain = SVector(
    ustrip(u"m", Lx),
    ustrip(u"m", Ly),
    ustrip(u"m", Lz))

d10 = 25u"μm"
d90 = 35u"μm"
d50 = sqrt(d10 * d90)
dmin = 20u"μm"
dmax = 45u"μm"
layer_height = 50u"μm"

n_particles = 30_000

radii = sample_lognormal_radii(n_particles, d10, d50, d90; dmin, dmax, rng=Xoshiro(1))

n_particles = estimate_n_particles(
    radii,
    domain,
    layer_height;
    packing = 0.58,
)

radii = sample_lognormal_radii(n_particles, d10, d50, d90; dmin, dmax, rng=Xoshiro(1))

CairoMakie.activate!()
fig_hist = Figure();
ax = Axis(fig_hist[1, 1], xlabel="Diameter (µm)", ylabel="Frequency (%)", title="Histogram")
hist!(ax, ustrip.(u"µm", 2*radii), normalization=:probability)
ax.xtickformat = values -> ["$(v) µm" for v in values]
ax.ytickformat = values -> ["$(round(100 * v, digits=1))%" for v in values]
save(joinpath(outdir, "powder_distribution.png"), fig_hist)


result = mp4(
    Float32.(ustrip.(u"m", radii)),
    Float32.(domain); 
    name = joinpath(outdir, "prathmesh.mp4"),
    frame_every = 500,
    fps = 24,
    n_steps = 80_000,
    min_speed = 1e-2,
    check_every = 1_000,
    min_speed_checks = 10,
    min_steps = 5_000,
    initial_height = 250u"µm",
    backend = CUDABackend(),
)
particles = result.particles

#particles = simulate_n_particles(
#    n_particles, 
#    Float32.(ustrip.(u"m", radii)),
#    Float32.(domain); 
#    backend = CUDABackend(),
#    n_steps = 80_000,
#    min_speed = 1e-2,
#    check_every = 1_000,
#    min_speed_checks = 10,
#    min_steps = 5_000
#)

particles = cut_powder_layer(
    particles,
    layer_height
)

hdf5_result = hdf5(
    particles, 
    domain; 
    name = joinpath(outdir, "prathmesh.h5"),
    cell_size = 5u"µm"
)

@show hdf5_result.dx
@show hdf5_result.dy
@show hdf5_result.dz

stl(
    particles,
    domain;
    name = joinpath(outdir, "prathmesh.stl"),
    cell_size = 5u"µm"
)

x = Array(particles.x)
y = Array(particles.y)
z = Array(particles.z)

GLMakie.activate!()
pts = [Point3f(x[i], y[i], z[i]) for i in eachindex(particles)]
fig3d = Figure()
ax = Axis3(fig3d[1, 1], aspect = :data)
meshscatter!(ax, pts, markersize=ustrip.(u"m", radii))

wait(display(fig3d))