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
