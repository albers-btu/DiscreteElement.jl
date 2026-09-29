"""
    sample_lognormal_radii(n, d10, d50, d90; dmin, dmax, rng) -> Vector

Draw `n` radii from the lognormal whose median diameter is `d50`. The width is
the average of the widths implied by `d10` and `d90`. Samples outside
`[dmin, dmax]` are rejected. Diameters are meters, or lengths such as
`30u"μm"`. The returned radii are lengths in meters.
"""
function sample_lognormal_radii(
    n::Int,
    d10,
    d50,
    d90;
    dmin,
    dmax,
    rng::AbstractRNG = Xoshiro(1),
)
    d10 = as_si(d10, u"m")
    d50 = as_si(d50, u"m")
    d90 = as_si(d90, u"m")
    dmin = as_si(dmin, u"m")
    dmax = as_si(dmax, u"m")
    n > 0 || throw(ArgumentError("n must be positive"))
    dmin > 0 || throw(ArgumentError("dmin must be positive"))
    dmin < dmax || throw(ArgumentError("dmin must be below dmax"))
    0 < d10 < d50 < d90 || throw(ArgumentError("require 0 < D10 < D50 < D90"))

    μ = log(float(d50))
    z10 = quantile(Distributions.Normal(), 0.1)
    z90 = quantile(Distributions.Normal(), 0.9)
    σ10 = (log(float(d10)) - μ) / z10
    σ90 = (log(float(d90)) - μ) / z90
    σ10 > 0 && σ90 > 0 || throw(ArgumentError("D10 and D90 must lie on opposite sides of D50"))
    σ = (σ10 + σ90) / 2
    if abs(σ10 - σ90) > 0.25 * σ
        @warn "D10 and D90 imply different lognormal widths; using the average" σ10 σ90 σ
    end

    dist = Distributions.LogNormal(μ, σ)
    radii = Vector{Float64}(undef, n)
    filled = 0
    tries = 0
    limit = 1000 * n
    while filled < n
        tries += 1
        tries <= limit || throw(ArgumentError(
            "could not draw $n diameters inside [$dmin, $dmax]; widen the truncation",
        ))
        d = rand(rng, dist)
        if dmin <= d <= dmax
            filled += 1
            radii[filled] = 0.5 * d
        end
    end
    return radii .* u"m"
end

"""
    suggest_lateral_size(radii, layer_height; packing=0.58) -> L

Square period of a cell whose height `layer_height` would hold `radii` at the
given solid fraction. The result is never smaller than two largest diameters.
Lengths may be bare meters or unitful (`120u"μm"`). The result is a length.
"""
function suggest_lateral_size(radii, layer_height; packing = 0.58)
    packing = as_dimensionless(packing)
    layer_height = as_si(layer_height, u"m")
    packing > 0 || throw(ArgumentError("packing must be positive"))
    layer_height > 0 || throw(ArgumentError("layer_height must be positive"))
    isempty(radii) && throw(ArgumentError("radii must not be empty"))
    volume = 0.0
    rmax = 0.0
    for r in radii
        rr = as_si(r, u"m")
        volume += (4 / 3) * π * rr^3
        rmax = max(rmax, rr)
    end
    L = sqrt(volume / (packing * layer_height))
    return max(L, 4.2 * rmax) * u"m"
end
