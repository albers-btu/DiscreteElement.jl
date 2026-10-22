import Base: summary

function summary(
    particles::ParticleArrays,
    domain::AbstractVector;
    name::AbstractString = "summary.txt",
    layer_height = nothing,
    cell_size,
    substrate_thickness = nothing,
    substrate_top = 0.0,
    include_particles::Bool = true,
    max_particles::Union{Nothing,Int} = nothing,
)
    n = length(particles)
    n > 0 || throw(ArgumentError("particles must not be empty"))
    length(domain) == 3 ||
        throw(ArgumentError("domain must have three components"))

    L = [Float64(as_si(d, u"m")) for d in domain]
    Lx, Ly, Lz = L
    all(>(0), L) || throw(ArgumentError("domain sides must be positive"))

    x = Array(particles.x)
    y = Array(particles.y)
    z = Array(particles.z)
    radius = Array(particles.radius)
    mass = Array(particles.mass)

    diameters = sort!(2 .* Float64.(radius))
    d10 = quantile(diameters, 0.10)
    d50 = quantile(diameters, 0.50)
    d90 = quantile(diameters, 0.90)
    dmin = first(diameters)
    dmax = last(diameters)
    dmean = sum(diameters) / n

    z_lo = minimum(z .- radius)
    z_hi = maximum(z .+ radius)

    h = if layer_height === nothing
        z_hi
    else
        Float64(as_si(layer_height, u"m"))
    end
    isfinite(h) && h >= 0 ||
        throw(ArgumentError("layer_height must be finite and non-negative"))
    h_source = layer_height === nothing ?
        "estimated from maximum particle top" : "user supplied"

    target = Float64(as_si(cell_size, u"m"))
    cell_source = "user supplied"
    target > 0 || throw(ArgumentError("cell_size must be positive"))

    Nx = max(1, ceil(Int, Lx / target - 1e-12))
    dx = Lx / Nx
    Ny = max(1, ceil(Int, Ly / dx - 1e-12))
    dy = Ly / Ny
    dz = dx

    substrate_top_m = Float64(as_si(substrate_top, u"m"))
    isfinite(substrate_top_m) ||
        throw(ArgumentError("substrate_top must be finite"))

    if substrate_thickness === nothing
        thickness_nominal = 0.0
        substrate_source = "none (not provided)"
    else
        thickness_nominal = Float64(as_si(substrate_thickness, u"m"))
        substrate_source = "user supplied"
    end
    isfinite(thickness_nominal) && thickness_nominal >= 0 ||
        throw(ArgumentError(
            "substrate_thickness must be finite and non-negative",
        ))

    n_sub = thickness_nominal > 0 ?
        max(1, round(Int, thickness_nominal / dz)) : 0
    substrate_thickness_actual = n_sub * dz
    substrate_bottom = substrate_top_m - substrate_thickness_actual
    grid_bottom = n_sub > 0 ? substrate_bottom : min(substrate_top_m, z_lo)
    n_powder = max(1, ceil(Int, (z_hi - grid_bottom) / dz - 1e-12))
    Nz = n_sub + n_powder
    grid_top = grid_bottom + Nz * dz

    total_volume = sum((4 / 3) * pi * Float64(r)^3 for r in radius)
    layer_volume = Lx * Ly * h
    domain_volume = Lx * Ly * Lz
    solid_fraction_layer =
        layer_volume > 0 ? total_volume / layer_volume : NaN
    solid_fraction_domain = total_volume / domain_volume
    mean_mass = sum(Float64.(mass)) / n
    density_est = sum(Float64.(mass)) / total_volume

    above_layer = count(z .+ radius .> h)

    max_particles === nothing || max_particles >= 0 ||
        throw(ArgumentError("max_particles must be non-negative"))
    nprint = if include_particles
        max_particles === nothing ? n : min(n, max_particles)
    else
        0
    end

    path = String(name)
    directory = dirname(path)
    isempty(directory) || mkpath(directory)

    open(path, "w") do io
        println(io, "# DiscreteElement powder-bed summary")
        println(io, "# Generated: ", Dates.now())
        println(io)

        println(io, "[Domain]")
        @printf(io, "Lx = %.9e m (%.6f um)\n", Lx, Lx * 1e6)
        @printf(io, "Ly = %.9e m (%.6f um)\n", Ly, Ly * 1e6)
        @printf(io, "Lz = %.9e m (%.6f um)\n", Lz, Lz * 1e6)
        @printf(io, "volume = %.9e m^3\n", domain_volume)
        println(io)

        println(io, "[Particles]")
        @printf(io, "count = %d\n", n)
        @printf(io, "total_solid_volume = %.9e m^3\n", total_volume)
        @printf(io, "solid_fraction_domain = %.9f\n", solid_fraction_domain)
        @printf(io, "solid_fraction_layer = %.9f\n", solid_fraction_layer)
        @printf(io, "mean_mass = %.9e kg\n", mean_mass)
        @printf(io, "estimated_density = %.9e kg/m^3\n", density_est)
        @printf(io, "z_bottom = %.9e m\n", z_lo)
        @printf(io, "z_top = %.9e m\n", z_hi)
        println(io)

        println(io, "[Particle size distribution]")
        @printf(io, "D10 = %.6f um (%.9e m)\n", d10 * 1e6, d10)
        @printf(io, "D50 = %.6f um (%.9e m)\n", d50 * 1e6, d50)
        @printf(io, "D90 = %.6f um (%.9e m)\n", d90 * 1e6, d90)
        @printf(io, "Dmin = %.6f um (%.9e m)\n", dmin * 1e6, dmin)
        @printf(io, "Dmax = %.6f um (%.9e m)\n", dmax * 1e6, dmax)
        @printf(io, "Dmean = %.6f um (%.9e m)\n", dmean * 1e6, dmean)
        println(io)

        println(io, "[Layer]")
        @printf(io, "layer_height = %.9e m (%.6f um)\n", h, h * 1e6)
        println(io, "layer_height_source = ", h_source)
        @printf(io, "layer_volume = %.9e m^3\n", layer_volume)
        @printf(io, "particles_above_layer_height = %d\n", above_layer)
        println(io)

        println(io, "[Grid]")
        @printf(io, "requested_cell_size = %.9e m (%.6f um)\n",
                target, target * 1e6)
        println(io, "cell_size_source = ", cell_source)
        @printf(io, "dx = %.9e m (%.6f um)\n", dx, dx * 1e6)
        @printf(io, "dy = %.9e m (%.6f um)\n", dy, dy * 1e6)
        @printf(io, "dz = %.9e m (%.6f um)\n", dz, dz * 1e6)
        @printf(io, "Nx = %d\n", Nx)
        @printf(io, "Ny = %d\n", Ny)
        @printf(io, "Nz = %d\n", Nz)
        @printf(io, "origin = (0, 0, %.9e) m\n", grid_bottom)
        @printf(io, "grid_top = %.9e m\n", grid_top)
        println(io)

        println(io, "[Substrate]")
        println(io, "enabled = ", n_sub > 0)
        println(io, "thickness_source = ", substrate_source)
        @printf(io, "thickness_nominal = %.9e m (%.6f um)\n",
                thickness_nominal, thickness_nominal * 1e6)
        @printf(io, "thickness_actual = %.9e m (%.6f um)\n",
                substrate_thickness_actual, substrate_thickness_actual * 1e6)
        @printf(io, "top_z = %.9e m\n", substrate_top_m)
        @printf(io, "bottom_z = %.9e m\n", substrate_bottom)
        @printf(io, "cells = %d\n", Nx * Ny * n_sub)
        @printf(io, "volume = %.9e m^3\n",
                Lx * Ly * substrate_thickness_actual)
        println(io)

        println(io, "[Particle list]")
        if include_particles
            @printf(io,
                "# %6s %14s %14s %14s %14s %14s %14s\n",
                "id", "x [m]", "y [m]", "z [m]", "r [m]", "d [m]", "mass [kg]")
            for i in 1:nprint
                @printf(io,
                    "  %6d %14.7e %14.7e %14.7e %14.7e %14.7e %14.7e\n",
                    i, x[i], y[i], z[i], radius[i], 2 * radius[i], mass[i])
            end
            if nprint < n
                @printf(io,
                    "# truncated: wrote %d of %d particles\n", nprint, n)
            end
        else
            println(io, "particle table disabled")
        end
    end

    return (;
        name = path,
        n,
        d10 = d10 * u"m",
        d50 = d50 * u"m",
        d90 = d90 * u"m",
        dmin = dmin * u"m",
        dmax = dmax * u"m",
        layer_height = h * u"m",
        cell_size = target * u"m",
        dx = dx * u"m",
        dy = dy * u"m",
        dz = dz * u"m",
        grid_size = (Nx, Ny, Nz),
        substrate_thickness = substrate_thickness_actual * u"m",
        substrate_top = substrate_top_m * u"m",
        substrate_bottom = substrate_bottom * u"m",
    )
end