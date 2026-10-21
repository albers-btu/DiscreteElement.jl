struct ParticleArrays{T<:AbstractFloat, A<:AbstractVector{T}}
    x::A; y::A; z::A
    vx::A; vy::A; vz::A
    wx::A; wy::A; wz::A
    fx::A; fy::A; fz::A
    tx::A; ty::A; tz::A
    radius::A; mass::A; inertia::A
end

Base.length(p::ParticleArrays) = length(p.x)
Base.eachindex(p::ParticleArrays) = eachindex(p.x)
nparticles(p::ParticleArrays) = length(p.x)

function ParticleArrays(
    backend,
    positions::AbstractVector{SVector{3, T}},
    radii::AbstractVector{<:Real},
    density::Real
) where {T<:AbstractFloat}
    n = length(positions)
    length(radii) == n ||
        throw(ArgumentError("positions and radii must have the same length"))

    ρ = T(density)
    x = Vector{T}(undef, n); y = Vector{T}(undef, n); z = Vector{T}(undef, n)
    radius = similar(x); mass = similar(x); inertia = similar(x)

    @inbounds for i in 1:n
        x[i], y[i], z[i] = positions[i]
        r = T(radii[i])
        r > zero(T) || throw(ArgumentError("radii must be positive"))
        radius[i] = r
        mass[i] = ρ * T(4) / T(3) * T(π) * r^3
        inertia[i] = T(2) / T(5) * mass[i] * r^2
    end

    z0 = zeros_on(backend, T, n)
    A = typeof(z0)

    return ParticleArrays{T,A}(
        upload_on(backend, x),
        upload_on(backend, y),
        upload_on(backend, z),

        z0,
        zeros_on(backend, T, n),
        zeros_on(backend, T, n),

        zeros_on(backend, T, n),
        zeros_on(backend, T, n),
        zeros_on(backend, T, n),

        zeros_on(backend, T, n),
        zeros_on(backend, T, n),
        zeros_on(backend, T, n),

        zeros_on(backend, T, n),
        zeros_on(backend, T, n),
        zeros_on(backend, T, n),

        upload_on(backend, radius),
        upload_on(backend, mass),
        upload_on(backend, inertia),
    )
end

ParticleArrays(positions, radii, density) =
    ParticleArrays(CPU(), positions, radii, density)

function zeros_on(backend, ::Type{T}, n::Int) where {T}
    a = allocate(backend, T, (n,))
    fill!(a, zero(T))
    return a
end

function upload_on(backend, values::AbstractVector{T}) where {T}
    a = allocate(backend, T, (length(values),))
    copyto!(a, values)
    return a
end