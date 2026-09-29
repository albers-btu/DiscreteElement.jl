"""
    place_rain(radii, domain; rng, volume_fraction=0.12, gap=0.02) -> named tuple

Park spheres in a dilute cloud above the floor, largest first, with no overlaps.
`x` and `y` are periodic. Radii are meters or lengths. Returns host columns
`x, y, z, radii` in meters, plus `z_top` (highest center) and `z_cloud`
(top of the insertion window).
"""
function place_rain(
    radii,
    domain::Domain{T};
    rng::AbstractRNG = Xoshiro(1),
    volume_fraction = 0.12,
    gap = 0.02,
    max_rounds::Int = 8,
) where {T}
    volume_fraction = as_dimensionless(volume_fraction)
    gap = as_dimensionless(gap)
    0 < volume_fraction < 0.4 ||
        throw(ArgumentError("volume_fraction must lie in (0, 0.4)"))
    0 <= gap < 0.5 || throw(ArgumentError("gap must lie in [0, 0.5)"))
    radii_t = T[T(as_si(r, u"m")) for r in radii]
    isempty(radii_t) && throw(ArgumentError("radii must not be empty"))
    all(>(zero(T)), radii_t) || throw(ArgumentError("radii must be positive"))
    rmax = maximum(radii_t)
    check_periodic_size(domain.Lx, domain.Ly, rmax)

    volume = zero(T)
    for r in radii_t
        volume += (T(4) / T(3)) * T(π) * r^3
    end
    height = volume / (T(volume_fraction) * domain.Lx * domain.Ly)
    height = max(height, 6 * rmax)
    order = sortperm(radii_t; rev = true)
    gap_t = T(gap)

    for _ in 1:max_rounds
        z_cloud = rmax * (1 + gap_t) + height
        placed = _try_place(radii_t, order, domain, z_cloud, gap_t, rng)
        if placed !== nothing
            x, y, z = placed
            return (; x, y, z, radii = radii_t, z_top = maximum(z), z_cloud)
        end
        height *= T(1.7)
    end
    throw(ArgumentError(
        "could not place $(length(radii_t)) spheres in $(domain.Lx) × $(domain.Ly) m " *
        "without overlap; enlarge the cell or lower the inserted volume fraction",
    ))
end

function _try_place(radii, order, domain::Domain{T}, z_cloud, gap, rng) where {T}
    n = length(radii)
    rmax = maximum(radii)
    cell = 2 * rmax * (1 + gap)
    nx = max(1, ceil(Int, domain.Lx / cell))
    ny = max(1, ceil(Int, domain.Ly / cell))
    nz = max(1, ceil(Int, z_cloud / cell))
    buckets = [Int[] for _ in 1:(nx * ny * nz)]
    xs = Vector{T}(undef, n)
    ys = Vector{T}(undef, n)
    zs = Vector{T}(undef, n)

    lin(ix, iy, iz) = ix + (iy - 1) * nx + (iz - 1) * nx * ny
    function bucket_index(x, y, z)
        ix = clamp(floor(Int, x / cell) + 1, 1, nx)
        iy = clamp(floor(Int, y / cell) + 1, 1, ny)
        iz = clamp(floor(Int, z / cell) + 1, 1, nz)
        return ix, iy, iz
    end
    function fits(x, y, z, ri)
        ix, iy, iz = bucket_index(x, y, z)
        for oz in -1:1, oy in -1:1, ox in -1:1
            jz = iz + oz
            1 <= jz <= nz || continue
            jx = mod1(ix + ox, nx)
            jy = mod1(iy + oy, ny)
            for j in buckets[lin(jx, jy, jz)]
                dx = minimum_image(xs[j] - x, domain.Lx)
                dy = minimum_image(ys[j] - y, domain.Ly)
                dz = zs[j] - z
                min_d = (ri + radii[j]) * (1 + gap)
                if dx * dx + dy * dy + dz * dz < min_d * min_d
                    return false
                end
            end
        end
        return true
    end

    attempts = 100
    for idx in order
        ri = radii[idx]
        zlo = ri * (1 + gap)
        span = z_cloud - zlo
        span > 0 || return nothing
        placed = false
        for _ in 1:attempts
            x = rand(rng, T) * domain.Lx
            y = rand(rng, T) * domain.Ly
            z = zlo + rand(rng, T) * span
            if fits(x, y, z, ri)
                xs[idx] = x
                ys[idx] = y
                zs[idx] = z
                ix, iy, iz = bucket_index(x, y, z)
                push!(buckets[lin(ix, iy, iz)], idx)
                placed = true
                break
            end
        end
        placed || return nothing
    end
    return xs, ys, zs
end
