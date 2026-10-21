struct Force{T,G,P}
    grid::G
    params::P
end

function Force(
    particles::ParticleArrays{T},
    model::LinearSpringDashpot{T},
    domain::SVector{3,T};
    max_per_cell::Union{Nothing,Int} = nothing,
) where {T<:AbstractFloat}
    backend = get_backend(particles.x)
    rmax = maximum(particles.radius)
    grid = make_cell_grid(
        backend, domain, rmax, length(particles); max_per_cell,
    )

    e = model.restitution
    log_e = e >= one(T) ? zero(T) : log(max(e, T(1e-8)))
    dedupe = (grid.nx < 3 || grid.ny < 3) ? 1 : 0

    params = ContactParams{T}(
        model.kn,
        e,
        log_e,
        model.mu,
        model.mu_roll,
        model.zeta_roll,
        domain[1],
        domain[2],
        zero(T),
        domain[3],
        dedupe,
    )

    return Force{T,typeof(grid),typeof(params)}(grid, params)
end

function compute_forces!(
    force::Force{T},
    particles::ParticleArrays{T};
    gravity::SVector{3,T} = SVector(0, 0, -T(9.81)),
) where {T<:AbstractFloat}
    backend = get_backend(particles.x)
    launch_clear_cells!(force.grid, backend)
    launch_bin!(force.grid, particles, force.params, backend)
    launch_force!(force.grid, particles, force.params, backend, gravity)
    return particles
end