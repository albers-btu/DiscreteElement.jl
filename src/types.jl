"""
Domain of the powder cell. The floor is the plane `z = 0`. `x` and `y` are periodic.

`Lx` and `Ly` are stored in meters. Pass a length (`1u"mm"`) or a bare number
of meters.
"""
struct Domain{T<:AbstractFloat}
    Lx::T
    Ly::T
end

function Domain(Lx::Real, Ly::Real)
    T = promote_type(typeof(float(Lx)), typeof(float(Ly)))
    return Domain{T}(T(Lx), T(Ly))
end

function Domain(Lx, Ly)
    return Domain(as_si(Lx, u"m"), as_si(Ly, u"m"))
end

function Base.show(io::IO, domain::Domain)
    print(io, "Domain(", domain.Lx * u"m", " × ", domain.Ly * u"m", ")")
end

"""
Bulk density and contact coefficients for a soft-sphere powder.

`restitution` is the normal coefficient of restitution of the linear
spring-dashpot. `mu_slide` caps tangential friction. `mu_roll` caps the
rolling and spin resistance torque at `mu_roll * R_eff * |F_n|`.
`zeta_roll` sets the viscous scale of that torque relative to the contact
frequency.

`density` is `kg/m^3` and `gravity` is `m/s^2` when given as bare numbers.
`7950u"kg/m^3"` and `9.81u"m/s^2"` are the same values.
"""
struct Material{T<:AbstractFloat}
    density::T
    restitution::T
    mu_slide::T
    mu_roll::T
    zeta_roll::T
    gravity::T
end

function Material{T}(;
    density = 7950u"kg/m^3",
    restitution = 0.2,
    mu_slide = 0.3,
    mu_roll = 0.05,
    zeta_roll = 0.3,
    gravity = 9.81u"m/s^2",
) where {T<:AbstractFloat}
    ρ = T(as_si(density, u"kg/m^3"))
    e = T(as_dimensionless(restitution))
    μ = T(as_dimensionless(mu_slide))
    μr = T(as_dimensionless(mu_roll))
    ζ = T(as_dimensionless(zeta_roll))
    g = T(as_si(gravity, u"m/s^2"))
    e > 0 || throw(ArgumentError("restitution must be positive"))
    e <= 1 || throw(ArgumentError("restitution must be at most 1"))
    μ >= 0 || throw(ArgumentError("mu_slide must be non-negative"))
    μr >= 0 || throw(ArgumentError("mu_roll must be non-negative"))
    ζ >= 0 || throw(ArgumentError("zeta_roll must be non-negative"))
    ρ > 0 || throw(ArgumentError("density must be positive"))
    return Material{T}(ρ, e, μ, μr, ζ, g)
end

Material(; kwargs...) = Material{Float64}(; kwargs...)

function Base.show(io::IO, material::Material)
    print(
        io,
        "Material(",
        material.density * u"kg/m^3",
        ", g = ",
        material.gravity * u"m/s^2",
        ", e = ",
        material.restitution,
        ")",
    )
end

"""
Isbits view of one DEM step. Passed by value into the force and integration kernels.
"""
struct SimParams{T<:AbstractFloat}
    kn::T
    restitution::T
    log_restitution::T
    mu_slide::T
    mu_roll::T
    zeta_roll::T
    gravity::T
    Lx::T
    Ly::T
    z0::T
    z1::T
    dt::T
    nx::Int
    ny::Int
    nz::Int
    n::Int
    max_per_cell::Int
    ncells::Int
    # 1 when a periodic neighbor stencil can land on the same cell twice (nx < 3 or ny < 3).
    dedupe::Int
end

"""
Struct-of-arrays particle state. Every column lives on the same KernelAbstractions backend.
"""
struct Particles{T<:AbstractFloat,A<:AbstractVector{T}}
    x::A
    y::A
    z::A
    vx::A
    vy::A
    vz::A
    wx::A
    wy::A
    wz::A
    fx::A
    fy::A
    fz::A
    tx::A
    ty::A
    tz::A
    radius::A
    mass::A
    inertia::A
    function Particles(
        x::A, y::A, z::A,
        vx::A, vy::A, vz::A,
        wx::A, wy::A, wz::A,
        fx::A, fy::A, fz::A,
        tx::A, ty::A, tz::A,
        radius::A, mass::A, inertia::A,
    ) where {T<:AbstractFloat,A<:AbstractVector{T}}
        n = length(x)
        for column in (
            y, z, vx, vy, vz, wx, wy, wz, fx, fy, fz, tx, ty, tz, radius, mass, inertia
        )
            length(column) == n ||
                throw(ArgumentError("particle columns must have equal length"))
        end
        return new{T,A}(
            x, y, z, vx, vy, vz, wx, wy, wz, fx, fy, fz, tx, ty, tz, radius, mass, inertia
        )
    end
end

nparticles(p::Particles) = length(p.x)

function Base.show(io::IO, p::Particles)
    print(io, "Particles(n=$(nparticles(p)), $(typeof(p.x)))")
end

"""Fixed-capacity cell list. `ids` is addressed as `(slot - 1) * ncells + cell`."""
struct CellGrid{I<:AbstractVector{Int32}}
    counts::I
    ids::I
    overflow::I
    nx::Int
    ny::Int
    nz::Int
    max_per_cell::Int
    ncells::Int
end

struct PowderBed{T<:AbstractFloat,A<:AbstractVector{T}}
    particles::Particles{T,A}
    domain::Domain{T}
    params::SimParams{T}
    grid::CellGrid
end

@inline function sphere_mass(radius::T, density::T) where {T<:AbstractFloat}
    return (T(4) / T(3)) * T(π) * radius^3 * density
end

@inline function sphere_inertia(mass::T, radius::T) where {T<:AbstractFloat}
    return T(2) / T(5) * mass * radius^2
end

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

"""
    Particles(backend, x, y, z, radius, density) -> Particles

Upload a host placement to `backend` and fill velocities, forces, mass, and inertia.
"""
function Particles(backend, x, y, z, radius, density)
    x = _si_column(x, u"m")
    y = _si_column(y, u"m")
    z = _si_column(z, u"m")
    radius = _si_column(radius, u"m")
    T = eltype(x)
    length(x) == length(y) == length(z) == length(radius) ||
        throw(ArgumentError("x, y, z, and radius must have equal length"))
    n = length(x)
    ρ = T(as_si(density, u"kg/m^3"))
    mass = Vector{T}(undef, n)
    inertia = Vector{T}(undef, n)
    @inbounds for i in 1:n
        radius[i] > 0 || throw(ArgumentError("radii must be positive"))
        mass[i] = sphere_mass(T(radius[i]), ρ)
        inertia[i] = sphere_inertia(mass[i], T(radius[i]))
    end
    zcol() = zeros_on(backend, T, n)
    return Particles(
        upload_on(backend, Vector{T}(x)),
        upload_on(backend, Vector{T}(y)),
        upload_on(backend, Vector{T}(z)),
        zcol(), zcol(), zcol(),
        zcol(), zcol(), zcol(),
        zcol(), zcol(), zcol(),
        zcol(), zcol(), zcol(),
        upload_on(backend, Vector{T}(radius)),
        upload_on(backend, mass),
        upload_on(backend, inertia),
    )
end

function cpu_particles(p::Particles{T,A}) where {T,A<:Array{T}}
    return p
end

function cpu_particles(p::Particles{T}) where {T}
    n = nparticles(p)
    cols = ntuple(nfields(p)) do i
        src = getfield(p, i)
        dst = Vector{T}(undef, n)
        copyto!(dst, src)
        dst
    end
    return Particles(cols...)
end

function on_backend(backend, p::Particles{T}) where {T}
    if get_backend(p.x) === backend || (get_backend(p.x) isa CPU && backend isa CPU)
        return p
    end
    host = cpu_particles(p)
    return Particles(
        ntuple(nfields(host)) do i
            upload_on(backend, getfield(host, i))
        end...,
    )
end

"""
Lateral period must exceed two largest diameters so each contact has one periodic image.
"""
function check_periodic_size(Lx, Ly, rmax)
    limit = 4 * rmax
    if Lx <= limit || Ly <= limit
        throw(ArgumentError(
            "Lx and Ly must exceed two largest diameters ($(limit) m) so a particle " *
            "cannot touch two images of the same neighbor. Got Lx=$Lx, Ly=$Ly.",
        ))
    end
    return nothing
end
