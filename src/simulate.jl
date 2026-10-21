function simulate_n_particles(
    n_particles::Int,
    radii_input::AbstractVector,
    domain::SVector{3,T};
    model::LinearSpringDashpot{T} = LinearSpringDashpot(
        T(0.5), T(0.5);
        mu = T(0.3), mu_roll = T(0.05), zeta_roll = T(0.3),
    ),
    dt::Union{Nothing,Real} = nothing,
    n_steps::Int = 80_000,
    density = 7890.0,
    gravity::SVector{3,T} = SVector{3,T}(0, 0, -T(9.81)),
    initial_height = nothing,
    seed_volume_fraction::Real = 0.5,
    rng::AbstractRNG = Xoshiro(1),
    min_speed = zero(T),
    check_every::Int = 500,      # host check interval; use 1000+ on GPU
    min_speed_checks::Int = 10,  # consecutive checks below min_speed
    min_steps::Int = 5_000,
    backend = CPU(),
    max_per_cell::Union{Nothing,Int} = nothing,
    snapshot_every::Int = 0,
    snapshot_callback = nothing,
) where {T<:AbstractFloat}
    length(radii_input) == n_particles ||
        throw(ArgumentError("n_particles must equal length(radii_input)"))
    n_particles > 0 || throw(ArgumentError("n_particles must be positive"))
    check_every >= 1 || throw(ArgumentError("check_every must be positive"))
    min_speed_checks >= 1 ||
        throw(ArgumentError("min_speed_checks must be positive"))
    snapshot_every >= 0 ||
        throw(ArgumentError("snapshot_every must be non-negative"))
    if snapshot_callback !== nothing && snapshot_every <= 0
        throw(ArgumentError(
            "snapshot_every must be positive when snapshot_callback is given",
        ))
    end

    radii = T[T(as_si(r, u"m")) for r in radii_input]
    v_stop = T(as_si(min_speed, u"m/s"))
    rmax = maximum(radii)
    domain[3] > T(2) * rmax ||
        throw(ArgumentError("domain[3] must exceed twice the largest radius"))

    if initial_height === nothing
        zero(T) < T(seed_volume_fraction) < one(T) ||
            throw(ArgumentError("seed_volume_fraction must lie in (0, 1)"))
        volume = sum(T(4) / T(3) * T(π) * r^3 for r in radii)
        z_cloud = volume / (T(seed_volume_fraction) * domain[1] * domain[2])
    else
        z_cloud = T(as_si(initial_height, u"m"))
    end
    z_cloud = clamp(z_cloud, T(2) * rmax, domain[3])

    # Dense random seed for FBA. This is not place_rain: the seed is placed at
    # a target volume fraction and place_particles! removes the overlaps.
    positions = Vector{SVector{3,T}}(undef, n_particles)
    for i in 1:n_particles
        positions[i] = SVector{3,T}(
            rand(rng, T) * domain[1],
            rand(rng, T) * domain[2],
            radii[i] + rand(rng, T) * max(z_cloud - T(2) * radii[i], zero(T)),
        )
    end

    ok = place_particles!(positions, radii, domain)
    ok || @warn "initial FBA placement did not converge; starting with overlaps"

    ρ = T(as_si(density, u"kg/m^3"))
    particles = ParticleArrays(backend, positions, radii, ρ)

    # Stable step from the stiffest contact.
    m_min = minimum(particles.mass)
    dt_t = dt === nothing ? T(0.25) / sqrt(model.kn / m_min) : T(dt)

    force = Force(particles, model, domain; max_per_cell)
    compute_forces!(force, particles; gravity)
    sum(force.grid.overflow) == 0 || throw(ArgumentError(
        "cell-list overflow: increase max_per_cell",
    ))

    snapshot_callback === nothing || snapshot_callback(particles, 0)

    below = 0
    for step in 1:n_steps
        kick_drift!(particles, dt_t, domain)
        compute_forces!(force, particles; gravity)
        half_kick!(particles, dt_t)

        if snapshot_callback !== nothing && step % snapshot_every == 0
            snapshot_callback(particles, step)
        end

        if step % check_every == 0
            sum(force.grid.overflow) == 0 || error(
                "cell-list overflow at step $step: increase max_per_cell",
            )
            vmax = maximum_speed(particles)

            if v_stop > zero(T) && step > min_steps
                if vmax < v_stop
                    below += 1
                    if below >= min_speed_checks
                        @printf(
                            "stopped early at step %d: vmax = %.3e m/s < %.3e m/s\n",
                            step, vmax, v_stop,
                        )
                        break
                    end
                else
                    below = 0
                end
            end

            if step % 5000 == 0
                zmax = maximum(particles.z)
                @printf(
                    "step %6d / %d   Δt = %.3e s   vmax = %.3e m/s   zmax = %.3e m\n",
                    step, n_steps, dt_t, vmax, zmax,
                )
            end
        end
    end

    return particles
end

function maximum_speed(particles::ParticleArrays{T}) where {T}
    v2 = particles.vx .^ 2 .+ particles.vy .^ 2 .+ particles.vz .^ 2
    return sqrt(maximum(v2))
end