using DiscreteElement
using FileIO
using HDF5
using GeometryBasics
using KernelAbstractions: CPU
using MeshIO
using Random
using Test
using Unitful: dimension, ustrip

function test_bed(xs, ys, zs, rs, masses; material = Material(; gravity = 0.0), kn = 1000.0, dt = 1e-4, z1 = 1.0, L = 1.0)
    backend = CPU()
    p = Particles(backend, xs, ys, zs, rs, 1.0)
    p.mass .= masses
    p.inertia .= 0.4 .* masses .* rs .^ 2
    domain = Domain(L, L)
    return assemble_bed(p, domain, material; kn, dt, z0 = 0.0, z1, max_per_cell = 32)
end

@testset "lognormal diameters" begin
    d50 = 30e-6
    σ = 0.25
    z10 = -1.2815515655446004
    z90 = -z10
    d10 = exp(log(d50) + σ * z10)
    d90 = exp(log(d50) + σ * z90)
    radii = sample_lognormal_radii(20_000, d10, d50, d90; dmin = 5e-6, dmax = 80e-6, rng = Xoshiro(1))
    diameters = sort(2 .* ustrip.(u"m", radii))
    n = length(diameters)
    @test diameters[clamp(round(Int, 0.1n), 1, n)] ≈ d10 rtol = 0.05
    @test diameters[clamp(round(Int, 0.5n), 1, n)] ≈ d50 rtol = 0.03
    @test diameters[clamp(round(Int, 0.9n), 1, n)] ≈ d90 rtol = 0.05
    @test all(5e-6 .<= diameters .<= 80e-6)
end

@testset "lateral size clears two diameters" begin
    radii = fill(15e-6, 100)
    L = ustrip(u"m", suggest_lateral_size(radii, 60e-6; packing = 0.6))
    @test L > 4 * maximum(radii)
    @test L > 0
end

@testset "rain has no overlaps" begin
    radii = fill(50e-6, 25)
    domain = Domain(suggest_lateral_size(radii, 200e-6), suggest_lateral_size(radii, 200e-6))
    placed = place_rain(radii, domain; rng = Xoshiro(2), gap = 0.02)
    n = length(placed.radii)
    for i in 1:n
        @test placed.z[i] >= placed.radii[i]
        @test 0 <= placed.x[i] < domain.Lx
        @test 0 <= placed.y[i] < domain.Ly
        for j in (i + 1):n
            dx = DiscreteElement.minimum_image(placed.x[j] - placed.x[i], domain.Lx)
            dy = DiscreteElement.minimum_image(placed.y[j] - placed.y[i], domain.Ly)
            dz = placed.z[j] - placed.z[i]
            reach = (placed.radii[i] + placed.radii[j]) * 1.02
            @test dx^2 + dy^2 + dz^2 >= reach^2 * (1 - 1e-8)
        end
    end
end

@testset "repulsion and periodic contact" begin
    r = 0.05
    bed = test_bed([0.50, 0.58], [0.50, 0.50], [0.10, 0.10], [r, r], [1.0, 1.0]; kn = 1000.0, z1 = 1.0)
    DiscreteElement.compute_forces!(bed.grid, bed.particles, bed.params)
    # overlap 0.02, fn = 20, particle 1 is pushed toward -x. Gravity is off.
    @test bed.particles.fx[1] ≈ -20 rtol = 1e-6
    @test bed.particles.fx[2] ≈ 20 rtol = 1e-6
    @test bed.particles.fy[1] ≈ 0 atol = 1e-8
    @test bed.particles.fz[1] ≈ 0 atol = 1e-8

    L = 1.0
    bedp = test_bed([0.01, 0.99], [0.5, 0.5], [0.1, 0.1], [0.02, 0.02], [1.0, 1.0]; kn = 1000.0, L = L)
    DiscreteElement.compute_forces!(bedp.grid, bedp.particles, bedp.params)
    @test bedp.particles.fx[1] > 0
    @test bedp.particles.fx[2] < 0
    @test bedp.particles.fx[1] ≈ -bedp.particles.fx[2] rtol = 1e-6
end

@testset "floor holds a resting sphere" begin
    r = 1e-3
    material = Material(; gravity = 9.81, restitution = 0.2, mu_slide = 0.3, mu_roll = 0.05)
    m = DiscreteElement.sphere_mass(r, material.density)
    kn, dt = contact_parameters([r], material; drop_height = 1e-4, overlap_fraction = 0.05, dt_factor = 0.2)
    bed = test_bed([5e-3], [5e-3], [r + 1e-4], [r], [m]; material, kn, dt, z1 = 5e-3, L = 2e-2)
    # test_bed overwrites mass after Particles() computed it from density 1.
    # The value passed in is already the 316L mass, and inertia was reset. Good.
    run = run_dem!(bed; max_time = 0.05, min_time = 0.005, stop_speed = 1e-3, check_every = 20, max_steps = 20_000)
    z = bed.particles.z[1]
    @test z > 0
    @test abs(z - r) < 0.05 * r
    @test abs(bed.particles.vz[1]) < 1e-2
    @test run.steps > 1
end

@testset "head-on restitution" begin
    r = 1e-3
    m = 1e-6
    kn = 100.0
    dt = 0.05 / sqrt(kn / m)
    gap = 1e-5
    v = 0.05
    material = Material(; gravity = 0.0, restitution = 0.5, mu_slide = 0.0, mu_roll = 0.0, zeta_roll = 0.0)
    x1 = 0.01
    x2 = x1 + 2r + gap
    bed = test_bed([x1, x2], [0.01, 0.01], [0.01, 0.01], [r, r], [m, m]; material, kn, dt, z1 = 0.05, L = 0.05)
    bed.particles.vx[1] = v
    bed.particles.vx[2] = -v
    run_dem!(bed; max_time = 5e-3, stop_speed = 0.0, check_every = 50, max_steps = 50_000)
    @test bed.particles.vx[1] < 0
    @test bed.particles.vx[2] > 0
    @test bed.particles.vx[1] + bed.particles.vx[2] ≈ 0 atol = 1e-4
    vout = bed.particles.vx[2] - bed.particles.vx[1]
    @test vout / (2v) ≈ 0.5 rtol = 0.2
end

@testset "one settled layer" begin
    r = 50e-6
    n = 36
    radii = fill(r, n)
    H = 4 * 2r
    L = suggest_lateral_size(radii, H; packing = 0.55)
    domain = Domain(L, L)
    placed = place_rain(radii, domain; rng = Xoshiro(3), volume_fraction = 0.2, gap = 0.02)
    material = Material()
    particles = Particles(CPU(), placed.x, placed.y, placed.z, placed.radii, material.density)
    kn, dt = contact_parameters(placed.radii, material; drop_height = placed.z_top, overlap_fraction = 0.1, dt_factor = 0.3)
    bed = assemble_bed(particles, domain, material; kn, dt, z0 = 0.0, z1 = placed.z_cloud + r, max_per_cell = 32)
    t_fall = sqrt(2 * placed.z_top / material.gravity)
    run = run_dem!(
        bed;
        max_time = 4 * t_fall,
        min_time = 1.1 * t_fall,
        stop_speed = 5e-3,
        check_every = max(1, floor(Int, t_fall / ustrip(u"s", dt) / 5)),
        max_steps = 80_000,
        energy_cap = 50 * sum(particles.mass) * material.gravity * placed.z_top,
    )
    host = DiscreteElement.cpu_particles(bed.particles)
    @test all(isfinite, host.x)
    @test all(isfinite, host.z)
    @test all(z -> z > 0, host.z)
    @test all(host.x .>= 0)
    @test all(host.x .< domain.Lx)
    @test maximum(host.z .+ host.radius) < placed.z_cloud
    @test ustrip(u"J", run.kinetic_energy) < 1e-10 || run.settled
    summary = packing_summary(bed.particles, domain)
    @test 0 < summary.apparent_packing < 1
    @test summary.mean_coordination > 0
end

@testset "stl and particle table" begin
    r = 80e-6
    L = 20 * r
    domain = Domain(L, L)
    material = Material(; gravity = 0.0)
    particles = Particles(CPU(), [L / 2], [L / 2], [2r], [r], material.density)
    bed = assemble_bed(particles, domain, material; kn = 1.0, dt = 1e-6, z0 = 0.0, z1 = 6r)
    dir = mktempdir()
    stl = joinpath(dir, "sphere.stl")
    csv = joinpath(dir, "particles.csv")
    out = export_powder_bed(bed; stl, csv, voxels_per_diameter = 8, substrate_thickness = 0)
    @test out.triangles > 20
    @test isfile(stl)
    @test filesize(stl) > 80
    rows = readlines(csv)
    data = [row for row in rows if !startswith(row, "#") && !startswith(row, "id")]
    @test length(data) == 1
    parts = split(data[1], ",")
    @test parse(Float64, parts[2]) ≈ L / 2
    @test parse(Float64, parts[5]) ≈ r

    loaded = load(stl)
    pts = GeometryBasics.coordinates(loaded)
    cx, cy, cz = L / 2, L / 2, 2r
    radii = map(pts) do p
        dx = Float64(p[1]) - cx
        dy = Float64(p[2]) - cy
        dz = Float64(p[3]) - cz
        sqrt(dx * dx + dy * dy + dz * dz)
    end
    @test DiscreteElement.median_value(radii) ≈ r rtol = 0.2
    @test ustrip(u"m", out.substrate_thickness) == 0
end

@testset "stl includes the build plate" begin
    r = 80e-6
    L = 16 * r
    thickness = 4 * r
    domain = Domain(L, L)
    material = Material(; gravity = 0.0)
    particles = Particles(CPU(), [L / 2], [L / 2], [r], [r], material.density)
    bed = assemble_bed(particles, domain, material; kn = 1.0, dt = 1e-6, z0 = 0.0, z1 = 4r)
    dir = mktempdir()
    stl = joinpath(dir, "bed.stl")
    csv = joinpath(dir, "particles.csv")
    out = export_powder_bed(
        bed; stl, csv, voxels_per_diameter = 6, substrate_thickness = thickness
    )
    @test ustrip(u"m", out.substrate_thickness) ≈ thickness
    pts = GeometryBasics.coordinates(load(stl))
    zs = [Float64(p[3]) for p in pts]
    xs = [Float64(p[1]) for p in pts]
    voxel = 2r / 6
    @test minimum(zs) ≈ -thickness atol = 2voxel
    @test minimum(zs) < -0.5 * thickness
    @test maximum(zs) > r
    @test minimum(xs) < 2voxel
    @test maximum(xs) > L - 2voxel
end

@testset "unitful inputs match SI numbers" begin
    @test Domain(1u"mm", 2u"cm").Lx ≈ 0.001
    @test Domain(1u"mm", 2u"cm").Ly ≈ 0.02
    @test_throws ArgumentError Domain(1u"s", 1u"m")

    material = Material(; density = 7.95u"g/cm^3", gravity = 981u"cm/s^2", restitution = 20u"percent")
    @test material.density ≈ 7950 rtol = 1e-12
    @test material.gravity ≈ 9.81 rtol = 1e-12
    @test material.restitution ≈ 0.2

    bare = sample_lognormal_radii(
        40, 18e-6, 30e-6, 45e-6; dmin = 15e-6, dmax = 53e-6, rng = Xoshiro(4)
    )
    unitful = sample_lognormal_radii(
        40, 18u"μm", 30u"μm", 45u"μm"; dmin = 15u"μm", dmax = 53u"μm", rng = Xoshiro(4)
    )
    @test ustrip.(u"m", bare) ≈ ustrip.(u"m", unitful)

    L = suggest_lateral_size((20u"μm", 30u"μm"), 100u"μm")
    @test dimension(L) == dimension(u"m")
    @test ustrip(u"m", L) > 0

    kn, dt = contact_parameters(
        [15u"μm"], Material(); drop_height = 0.5u"mm", overlap_fraction = 0.08, dt_factor = 0.25
    )
    @test dimension(kn) == dimension(u"N/m")
    @test dimension(dt) == dimension(u"s")
    @test ustrip(u"N/m", kn) > 0
    @test ustrip(u"s", dt) > 0
    @test_throws ArgumentError contact_parameters(
        [15u"μm"], Material(); drop_height = 1u"s"
    )
end

@testset "sphere-cell volume" begin
    vol = DiscreteElement.sphere_aabb_volume
    @test vol(0, 0, 0, 1, -2, 2, -2, 2, -2, 2) ≈ 4 / 3 * π rtol = 1e-6
    @test vol(0, 0, 0, 1, 0, 2, 0, 2, 0, 2) ≈ π / 6 rtol = 1e-4
    @test vol(0, 0, 0, 2, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5) ≈ 1 rtol = 1e-6
    @test vol(0, 0, 0, 1, 3, 4, 0, 1, 0, 1) == 0
    @test DiscreteElement.disk_rect_area(1, -2, 2, -2, 2) ≈ π rtol = 1e-8
    @test DiscreteElement.disk_rect_area(1, -2, 2, 0, 2) ≈ π / 2 rtol = 1e-8
end

@testset "hdf5 powder bed" begin
    r = 50e-6
    L = 20 * r
    domain = Domain(L, L)
    # Centre of the particle sits on a cell centre once dx divides L.
    particles = Particles(CPU(), [L / 2], [L / 2], [2r], [r], 1000.0)
    bed = assemble_bed(particles, domain, Material(; gravity = 0.0); kn = 1.0, dt = 1e-6, z0 = 0.0, z1 = 6r)
    grid = DiscreteElement.metal_fraction_grid(
        particles, domain; voxels_per_diameter = 6, substrate_thickness = 4r
    )
    @test grid.dx ≈ grid.dz
    @test grid.origin[1] == 0
    @test grid.origin[3] < 0
    Nx, Ny, Nz = size(grid.phi)
    x = floor(Int, (L / 2 - grid.origin[1]) / grid.dx) + 1
    y = floor(Int, (L / 2 - grid.origin[2]) / grid.dy) + 1
    z = floor(Int, (2r - grid.origin[3]) / grid.dz) + 1
    @test 1 <= x <= Nx && 1 <= y <= Ny && 1 <= z <= Nz
    @test grid.phi[x, y, z] > 0
    @test all(0 .<= grid.phi .<= 1)
    @test grid.substrate !== nothing
    @test all(grid.phi[grid.substrate .== 1] .== 0)
    @test any(grid.substrate .== 1)
    cell = grid.dx * grid.dy * grid.dz
    @test sum(grid.phi) * cell ≈ (4 / 3) * π * r^3 rtol = 1e-3

    dir = mktempdir()
    path = joinpath(dir, "bed.h5")
    out = export_powder_hdf5(
        bed; h5 = path, voxels_per_diameter = 6, substrate_thickness = 4r
    )
    h5open(path, "r") do file
        phi = file["phi"][:, :, :]
        @test eltype(phi) == Float32
        @test ndims(phi) == 3
        @test size(phi) == out.size
        @test all(0 .<= phi .<= 1)
        attrs = attributes(file)
        @test read(attrs["index_order"]) == "x_fastest"
        @test read(attrs["cell_center"]) == "integer"
        @test read(attrs["quantity"]) == "metal_volume_fraction"
        @test read(attrs["dx"]) ≈ grid.dx
        origin = read(attrs["origin"])
        @test origin ≈ collect(grid.origin)
        substrate = file["substrate"][:, :, :]
        @test eltype(substrate) == UInt8
        @test size(substrate) == size(phi)
        ix = floor(Int, (L / 2 - origin[1]) / read(attrs["dx"])) + 1
        iy = floor(Int, (L / 2 - origin[2]) / read(attrs["dy"])) + 1
        iz = floor(Int, (2r - origin[3]) / read(attrs["dz"])) + 1
        @test phi[ix, iy, iz] > 0
    end

    bare = joinpath(dir, "spheres.h5")
    export_powder_hdf5(bed; h5 = bare, voxels_per_diameter = 6, substrate_thickness = 0)
    h5open(bare, "r") do file
        @test !haskey(file, "substrate")
        @test haskey(file, "phi")
    end
end
