abstract type ContactModel end

struct LinearSpringDashpot{T} <: ContactModel
    kn::T
    restitution::T
    mu::T
    mu_roll::T
    zeta_roll::T
end

function LinearSpringDashpot(
    kn::Real,
    restitution::Real;
    mu::Real = 0.3,
    mu_roll::Real = 0.05,
    zeta_roll::Real = 0.3,
)
    T = promote_type(
        typeof(float(kn)), typeof(float(restitution)),
        typeof(float(mu)), typeof(float(mu_roll)), typeof(float(zeta_roll)),
    )
    return LinearSpringDashpot{T}(
        T(kn), T(restitution), T(mu), T(mu_roll), T(zeta_roll)
    )
end

struct HertzMindlinJKR{T<:AbstractFloat} <: ContactModel
    young_modulus::T
    poisson_ratio::T
    surface_energy::T
end

function HertzMindlinJKR(young_modulus, poisson_ratio, surface_energy)
    T = promote_type(
        typeof(float(young_modulus)),
        typeof(float(poisson_ratio)),
        typeof(float(surface_energy)),
    )
    return HertzMindlinJKR{T}(
        T(young_modulus), T(poisson_ratio), T(surface_energy),
    )
end