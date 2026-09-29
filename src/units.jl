# Unit conversion at the public boundary.
#
# Bare numbers are SI base units (m, kg, s, and derived units such as N/m).
# Unitful quantities are converted to those same units and stripped, so the
# kernels keep plain floating-point state.

import Unitful
import Unitful: @u_str

function as_si(x::Real, ::Unitful.Units)
    return float(x)
end

function as_si(x::Unitful.Quantity, unit::Unitful.Units)
    if Unitful.dimension(x) != Unitful.dimension(unit)
        throw(ArgumentError(
            "expected dimensions of $(unit), got $(x). A bare number is taken as $(unit).",
        ))
    end
    return float(Unitful.ustrip(unit, x))
end

function as_si(x, ::Unitful.Units)
    throw(ArgumentError("expected a number or a Unitful quantity, got $(typeof(x))"))
end

"""Bare numbers are already dimensionless. `2u"percent"` is accepted."""
function as_dimensionless(x::Real)
    return float(x)
end

function as_dimensionless(x::Unitful.Quantity)
    if Unitful.dimension(x) != Unitful.dimension(Unitful.NoUnits)
        throw(ArgumentError("expected a dimensionless quantity, got $(x)"))
    end
    return float(Unitful.ustrip(Unitful.NoUnits, x))
end

function as_dimensionless(x)
    throw(ArgumentError("expected a number or a dimensionless quantity, got $(typeof(x))"))
end

"""`Inf` stays a unitless sentinel. Any finite energy is converted to joules."""
function as_energy(x::Real)
    isinf(x) && return oftype(float(x), Inf)
    return float(x)
end

function as_energy(x::Unitful.Quantity)
    return as_si(x, u"J")
end

function maybe_length(x::Nothing)
    return nothing
end

function maybe_length(x)
    return as_si(x, u"m")
end

function maybe_time(x::Nothing)
    return nothing
end

function maybe_time(x)
    return as_si(x, u"s")
end

function _si_column(x::AbstractVector{<:Real}, ::Unitful.Units)
    return x
end

function _si_column(x::AbstractVector{<:Unitful.Quantity}, unit::Unitful.Units)
    return as_si.(x, Ref(unit))
end

function _si_column(x, unit::Unitful.Units)
    return [as_si(value, unit) for value in x]
end
