"""
    make_grid(backend, domain, z0, z1, rmax; max_per_cell=64) -> CellGrid

Cell list covering `z ∈ [z0, z1]`. Cell edges are at least one maximum diameter,
which is the largest center-to-center contact distance.
"""
function make_grid(
    backend,
    domain::Domain{T},
    z0,
    z1,
    rmax;
    max_per_cell::Int = 64,
) where {T}
    z0t = T(as_si(z0, u"m"))
    z1t = T(as_si(z1, u"m"))
    rmax = T(as_si(rmax, u"m"))
    z1t > z0t || throw(ArgumentError("z1 must exceed z0"))
    rmax > 0 || throw(ArgumentError("rmax must be positive"))
    max_per_cell >= 4 || throw(ArgumentError("max_per_cell must be at least 4"))
    cutoff = 2 * T(rmax)
    nx = max(1, floor(Int, domain.Lx / cutoff))
    ny = max(1, floor(Int, domain.Ly / cutoff))
    nz = max(1, floor(Int, (z1t - z0t) / cutoff))
    ncells = nx * ny * nz
    return CellGrid(
        zeros_on(backend, Int32, ncells),
        zeros_on(backend, Int32, max_per_cell * ncells),
        zeros_on(backend, Int32, 1),
        nx, ny, nz, max_per_cell, ncells,
    )
end

"""
    assemble_bed(particles, domain, material; kn, dt, z0, z1, max_per_cell=64) -> PowderBed
"""
function assemble_bed(
    particles::Particles{T},
    domain::Domain{T},
    material::Material{T};
    kn,
    dt,
    z0,
    z1,
    max_per_cell::Int = 64,
) where {T}
    kn = T(as_si(kn, u"N/m"))
    dt = T(as_si(dt, u"s"))
    z0 = T(as_si(z0, u"m"))
    z1 = T(as_si(z1, u"m"))
    nparticles(particles) > 0 || throw(ArgumentError("particles must not be empty"))
    rmax = maximum(particles.radius)
    check_periodic_size(domain.Lx, domain.Ly, rmax)
    kn > 0 || throw(ArgumentError("kn must be positive"))
    dt > 0 || throw(ArgumentError("dt must be positive"))
    grid = make_grid(
        get_backend(particles.x), domain, z0, z1, rmax; max_per_cell
    )
    params = make_params(
        material, domain, grid;
        kn, dt, n = nparticles(particles), z0, z1,
    )
    return PowderBed(particles, domain, params, grid)
end

"""
    run_dem!(bed; max_time, min_time=0, max_steps, stop_speed=0, check_every=100,
             energy_cap=Inf, verbose=false)

Semi-implicit Euler steps until `max_time`, `max_steps`, or the
mass-weighted root-mean-square speed (translation plus rotation) drops below
`stop_speed` after `min_time`. `stop_speed <= 0` disables the early exit.
Kinetic energy above `energy_cap` aborts the run as unstable.

Times are seconds, speeds are `m/s`, and `energy_cap` is joules. Unitful
quantities are accepted (`10u"ms"`, `5u"mm/s"`, `1u"μJ"`). `Inf` leaves the
energy check off. The returned time is seconds and the kinetic energy is joules.
"""
function run_dem!(
    bed::PowderBed{T};
    max_time,
    min_time = 0,
    max_steps::Int = 10^8,
    stop_speed = 0,
    check_every::Int = 100,
    energy_cap = Inf,
    verbose::Bool = false,
) where {T}
    max_time = T(as_si(max_time, u"s"))
    min_time = T(as_si(min_time, u"s"))
    stop_speed = T(as_si(stop_speed, u"m/s"))
    energy_cap = as_energy(energy_cap)
    max_time > 0 || throw(ArgumentError("max_time must be positive"))
    check_every >= 1 || throw(ArgumentError("check_every must be positive"))
    p = bed.particles
    params = bed.params
    grid = bed.grid
    ke = zeros_on(get_backend(p.x), T, nparticles(p))
    t = zero(T)
    step = 0
    settled = false
    KE = zero(T)
    mass_sum = sum(p.mass)
    while t < T(max_time) && step < max_steps
        compute_forces!(grid, p, params)
        integrate!(p, params)
        step += 1
        t += params.dt
        due = step % check_every == 0 || step == max_steps || t >= T(max_time)
        if due
            KE = kinetic_energy!(ke, p)
            zmax = maximum(p.z)
            speed = sqrt(2 * max(KE, zero(T)) / mass_sum)
            if verbose
                @printf(
                    "step %7d   t = %.4e s   speed = %.4e m/s   zmax = %.4e m\n",
                    step, t, speed, zmax,
                )
            end
            if !isfinite(KE) || !isfinite(zmax)
                error(
                    "non-finite particle state at step $step. " *
                    "Lower dt_factor or overlap_fraction.",
                )
            end
            if zmax > params.z1
                error(
                    "a particle rose above the cell grid (z=$zmax, grid top $(params.z1)). " *
                    "Increase the vertical grid extent.",
                )
            end
            if KE > energy_cap
                error(
                    "kinetic energy $KE exceeded the stability cap $energy_cap at step $step. " *
                    "Lower dt_factor.",
                )
            end
            if stop_speed > 0 && t >= T(min_time) && speed < T(stop_speed)
                settled = true
                break
            end
        end
    end
    return (; steps = step, time = t * u"s", kinetic_energy = KE * u"J", settled)
end

"""
    simulate_powder_layer(; n=3000, d10, d50, d90, dmin, dmax, kwargs...)

Sample a truncated lognormal powder, rain it onto `z = 0` inside a periodic
cell, settle it, and write an STL plus a particle CSV. Bare lengths are
meters. Unitful lengths such as `18u"μm"` are converted to meters.
The default powder is a 15–53 µm cut with D10/D50/D90 = 18/30/45 µm and the
density of 316L. `backend` is any KernelAbstractions backend; `CPU()` is the
default, and a CUDA run is `backend = CUDABackend()` after `using CUDA`.
`L` is the side of a square cell. Set `Ly` alone for a strip along x: the
length is chosen so the same powder volume still fills `layer_height`.
`Lx` and `Ly` together set the rectangle directly. Each period must be longer
than two largest diameters. Dimensional fields of the result (`time`,
`height`, `Lx`, `Ly`, `kn`, `dt`, `kinetic_energy`, `substrate_thickness`)
carry units.
"""
function simulate_powder_layer(;
    n::Int = 3000,
    d10 = 18u"μm",
    d50 = 30u"μm",
    d90 = 45u"μm",
    dmin = 15u"μm",
    dmax = 53u"μm",
    L = nothing,
    Lx = nothing,
    Ly = nothing,
    layer_height = nothing,
    packing_guess = 0.58,
    material::Material{Float64} = Material(),
    overlap_fraction = 0.08,
    dt_factor = 0.25,
    volume_fraction = 0.12,
    gap = 0.02,
    seed::Int = 1,
    backend = CPU(),
    stl::AbstractString = "powder_bed.stl",
    csv::AbstractString = "particles.csv",
    h5::Union{Nothing,AbstractString} = "powder_bed.h5",
    voxels_per_diameter = 6,
    substrate_thickness = nothing,
    stop_speed = 5u"mm/s",
    max_time = nothing,
    max_steps::Int = 500_000,
    max_per_cell::Int = 64,
    verbose::Bool = true,
)
    d10 = as_si(d10, u"m")
    d50 = as_si(d50, u"m")
    d90 = as_si(d90, u"m")
    dmin = as_si(dmin, u"m")
    dmax = as_si(dmax, u"m")
    packing_guess = as_dimensionless(packing_guess)
    overlap_fraction = as_dimensionless(overlap_fraction)
    dt_factor = as_dimensionless(dt_factor)
    volume_fraction = as_dimensionless(volume_fraction)
    gap = as_dimensionless(gap)
    voxels_per_diameter = as_dimensionless(voxels_per_diameter)
    stop_speed = as_si(stop_speed, u"m/s")
    layer_height = maybe_length(layer_height)
    L = maybe_length(L)
    Lx = maybe_length(Lx)
    Ly = maybe_length(Ly)
    substrate_thickness = maybe_length(substrate_thickness)
    max_time = maybe_time(max_time)
    rng = Xoshiro(seed)
    radii = sample_lognormal_radii(n, d10, d50, d90; dmin, dmax, rng)
    height = layer_height === nothing ? 4 * d50 : layer_height
    square = L === nothing ? as_si(suggest_lateral_size(radii, height; packing = packing_guess), u"m") : L
    if Lx === nothing && Ly === nothing
        Lx = square
        Ly = square
    elseif Lx === nothing
        Lx = square^2 / Ly
    elseif Ly === nothing
        Ly = square^2 / Lx
    end
    domain = Domain(Lx, Ly)
    placed = place_rain(radii, domain; rng, volume_fraction, gap)
    particles = Particles(
        backend, placed.x, placed.y, placed.z, placed.radii, material.density
    )
    kn, dt = contact_parameters(
        placed.radii, material;
        drop_height = placed.z_top, overlap_fraction, dt_factor,
    )
    z1 = placed.z_cloud + maximum(placed.radii)
    bed = assemble_bed(
        particles, domain, material;
        kn, dt, z0 = 0.0, z1, max_per_cell,
    )
    t_fall = sqrt(2 * placed.z_top / material.gravity)
    t_end = max_time === nothing ? 4 * t_fall : float(max_time)
    t_min = max_time === nothing ? 1.15 * t_fall : 0.0
    cap = 50 * sum(placed.radii .^ 3) * (4 / 3 * π) * material.density *
        material.gravity * placed.z_top + 1e-18
    dt_s = as_si(dt, u"s")
    checks = max(1, floor(Int, t_end / dt_s / 20))
    if verbose
        @printf("Powder layer   n = %d   Lx = %.4e m   Ly = %.4e m   cloud top = %.4e m\n",
            n, domain.Lx, domain.Ly, placed.z_cloud)
        @printf("Contact        kn = %.4e N/m   dt = %.4e s   steps to %.4e s ≤ %d\n",
            as_si(kn, u"N/m"), dt_s, t_end, min(max_steps, ceil(Int, t_end / dt_s)))
    end
    run = run_dem!(
        bed;
        max_time = t_end,
        min_time = t_min,
        max_steps,
        stop_speed,
        check_every = checks,
        energy_cap = cap,
        verbose,
    )
    summary = packing_summary(bed.particles, domain)
    exported = export_powder_bed(
        bed; stl, csv, voxels_per_diameter, substrate_thickness
    )
    hdf5 = h5 === nothing ? nothing : export_powder_hdf5(
        bed; h5, voxels_per_diameter, substrate_thickness
    )
    if verbose
        @printf(
            "Settled        steps = %d   t = %.4e s   speed = %.4e m/s   height = %.4e m   packing = %.3f\n",
            run.steps,
            as_si(run.time, u"s"),
            sqrt(2 * as_si(run.kinetic_energy, u"J") / sum(bed.particles.mass)),
            as_si(summary.height, u"m"),
            summary.apparent_packing,
        )
        @printf("Substrate      thickness = %.4e m\n", as_si(exported.substrate_thickness, u"m"))
        println("Wrote $stl")
        println("Wrote $csv")
        hdf5 === nothing || println("Wrote $(hdf5.h5)")
    end
    return (;
        bed,
        n,
        steps = run.steps,
        time = run.time,
        kinetic_energy = run.kinetic_energy,
        settled = run.settled,
        height = summary.height,
        apparent_packing = summary.apparent_packing,
        mean_coordination = summary.mean_coordination,
        stl = String(stl),
        csv = String(csv),
        h5 = hdf5 === nothing ? nothing : hdf5.h5,
        substrate_thickness = exported.substrate_thickness,
        Lx = domain.Lx,
        Ly = domain.Ly,
        kn,
        dt,
    )
end
