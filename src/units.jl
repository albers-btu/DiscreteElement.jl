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