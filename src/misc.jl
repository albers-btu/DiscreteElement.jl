function cut_powder_layer(
    particles::ParticleArrays,
    layer_height;
    criterion::Symbol = :top,
    allow_empty::Bool = false,
)
    criterion in (:top, :center) || throw(ArgumentError(
        "criterion must be :top or :center",
    ))

    h = eltype(particles.x)(as_si(layer_height, u"m"))
    isfinite(h) || throw(ArgumentError("layer_height must be finite"))
    h >= zero(h) ||
        throw(ArgumentError("layer_height must be non-negative"))

    zhost = Array(particles.z)
    rhost = Array(particles.radius)

    keep = if criterion === :top
        findall(i -> zhost[i] + rhost[i] <= h, eachindex(zhost))
    else
        findall(i -> zhost[i] <= h, eachindex(zhost))
    end

    if isempty(keep) && !allow_empty
        throw(ArgumentError(
            "layer_height = $layer_height leaves no particles; " *
            "lower the cut height or pass allow_empty=true",
        ))
    end

    T = eltype(particles.x)
    cols = ntuple(
        i -> _cut_column(getfield(particles, i), keep),
        nfields(particles),
    )
    return ParticleArrays{T, typeof(cols[1])}(cols...)
end

function _cut_column(col, keep)
    host = Array(col)
    out = similar(col, length(keep))
    copyto!(out, host[keep])
    return out
end

function estimate_n_particles(
    radii,
    domain,
    layer_height;
    packing::Real = 0.58,
)
    isempty(radii) && throw(ArgumentError("radii must not be empty"))
    length(domain) >= 2 ||
        throw(ArgumentError("domain must contain Lx and Ly"))

    Lx = Float64(as_si(domain[1], u"m"))
    Ly = Float64(as_si(domain[2], u"m"))
    h = Float64(as_si(layer_height, u"m"))
    φ = Float64(packing)

    Lx > 0 || throw(ArgumentError("domain[1] must be positive"))
    Ly > 0 || throw(ArgumentError("domain[2] must be positive"))
    h > 0 || throw(ArgumentError("layer_height must be positive"))
    isfinite(Lx) && isfinite(Ly) && isfinite(h) ||
        throw(ArgumentError("domain and layer_height must be finite"))
    0 < φ <= 1 || throw(ArgumentError("packing must lie in (0, 1]"))

    mean_volume = 0.0
    for r in radii
        rr = Float64(as_si(r, u"m"))
        rr > 0 || throw(ArgumentError("radii must be positive"))
        mean_volume += (4 / 3) * π * rr^3
    end
    mean_volume /= length(radii)

    n_est = φ * Lx * Ly * h / mean_volume
    isfinite(n_est) ||
        throw(ArgumentError("estimated particle count is not finite"))

    return max(1, round(Int, n_est))
end