function place_particles!(
    positions::Vector{SVector{3,T}},
    radii::AbstractVector{T},
    domain::SVector{3,T};
    max_iter::Int = 1000,
    tol::Real = 1e-4,
    relaxation::Real = 0.2,
) where {T<:AbstractFloat}
    n = length(positions)
    n == length(radii) ||
        throw(ArgumentError("positions and radii must have the same length"))
    n == 0 && return true
    all(r -> r > zero(T), radii) ||
        throw(ArgumentError("radii must be positive"))
    max_iter >= 1 || throw(ArgumentError("max_iter must be positive"))
    zero(T) < T(relaxation) <= one(T) ||
        throw(ArgumentError("relaxation must lie in (0, 1]"))

    Lx, Ly, Lz = domain
    Lx > zero(T) && Ly > zero(T) && Lz > zero(T) ||
        throw(ArgumentError("domain sides must be positive"))

    rmax = maximum(radii)
    rmin = minimum(radii)
    cutoff = T(2) * rmax
    nx = max(1, floor(Int, Lx / cutoff))
    ny = max(1, floor(Int, Ly / cutoff))
    nz = max(1, floor(Int, Lz / cutoff))
    ncells = nx * ny * nz

    xs = Vector{T}(undef, n)
    ys = Vector{T}(undef, n)
    zs = Vector{T}(undef, n)
    @inbounds for i in 1:n
        r = radii[i]
        Lz > T(2) * r || throw(ArgumentError(
            "domain[3] must exceed twice the largest radius",
        ))
        p = positions[i]
        xs[i] = mod(p[1], Lx)
        ys[i] = mod(p[2], Ly)
        zs[i] = clamp(p[3], r, Lz - r)
    end

    # Cell -> particle list. Rebuilt once per FBA iteration.
    buckets = [Int[] for _ in 1:ncells]

    # Unique neighbour cells for every cell. `d == c` is kept for intra-cell
    # pairs; `d > c` is kept so every cross-cell pair is visited once.
    neighbours = [Int[] for _ in 1:ncells]
    seen = Vector{Int}(undef, 27)
    @inbounds for iz in 1:nz, iy in 1:ny, ix in 1:nx
        c = cell_linear(ix, iy, iz, nx, ny)
        nseen = 0
        for dz in -1:1, dy in -1:1, dx in -1:1
            jz = iz + dz
            1 <= jz <= nz || continue
            d = cell_linear(
                wrap_cell(ix + dx, nx),
                wrap_cell(iy + dy, ny),
                jz, nx, ny,
            )
            d >= c || continue
            duplicate = false
            for s in 1:nseen
                if seen[s] == d
                    duplicate = true
                    break
                end
            end
            duplicate && continue
            nseen += 1
            seen[nseen] = d
            push!(neighbours[c], d)
        end
    end

    disp = [zero(SVector{3,T}) for _ in 1:n]
    target_tol = T(tol) * rmin
    relax = T(relaxation)
    xlo = zero(T)
    max_overlap = zero(T)

    for _ in 1:max_iter
        for b in buckets
            empty!(b)
        end
        @inbounds for i in 1:n
            c = cell_linear(
                cell_index_1d(xs[i], Lx, nx),
                cell_index_1d(ys[i], Ly, ny),
                cell_index_z(zs[i], xlo, Lz, nz),
                nx, ny,
            )
            push!(buckets[c], i)
        end
        fill!(disp, zero(SVector{3,T}))
        max_overlap = zero(T)

        @inbounds for c in 1:ncells
            bi = buckets[c]
            isempty(bi) && continue
            for d in neighbours[c]
                bj = buckets[d]
                isempty(bj) && continue
                if d == c
                    for a in 1:(length(bi) - 1)
                        i = bi[a]
                        ri = radii[i]
                        xi = xs[i]; yi = ys[i]; zi = zs[i]
                        for b in (a + 1):length(bi)
                            j = bi[b]
                            dx = minimum_image(xs[j] - xi, Lx)
                            dy = minimum_image(ys[j] - yi, Ly)
                            dz = zs[j] - zi
                            rad = ri + radii[j]
                            dist2 = dx * dx + dy * dy + dz * dz
                            dist2 < rad * rad || continue
                            dist2 > zero(T) || continue
                            dist = sqrt(dist2)
                            overlap = rad - dist
                            max_overlap = max(max_overlap, overlap)
                            s = overlap / dist
                            disp[i] -= s * SVector{3,T}(dx, dy, dz)
                            disp[j] += s * SVector{3,T}(dx, dy, dz)
                        end
                    end
                else
                    for i in bi
                        ri = radii[i]
                        xi = xs[i]; yi = ys[i]; zi = zs[i]
                        for j in bj
                            dx = minimum_image(xs[j] - xi, Lx)
                            dy = minimum_image(ys[j] - yi, Ly)
                            dz = zs[j] - zi
                            rad = ri + radii[j]
                            dist2 = dx * dx + dy * dy + dz * dz
                            dist2 < rad * rad || continue
                            dist2 > zero(T) || continue
                            dist = sqrt(dist2)
                            overlap = rad - dist
                            max_overlap = max(max_overlap, overlap)
                            s = overlap / dist
                            disp[i] -= s * SVector{3,T}(dx, dy, dz)
                            disp[j] += s * SVector{3,T}(dx, dy, dz)
                        end
                    end
                end
            end
        end

        if max_overlap <= target_tol
            write_positions!(positions, xs, ys, zs)
            return true
        end

        @inbounds for i in 1:n
            r = radii[i]
            xs[i] = mod(xs[i] + relax * disp[i][1], Lx)
            ys[i] = mod(ys[i] + relax * disp[i][2], Ly)
            zs[i] = clamp(zs[i] + relax * disp[i][3], r, Lz - r)
        end
    end

    write_positions!(positions, xs, ys, zs)
    @warn "force-biased placement did not converge" max_overlap = max_overlap
    return false
end

function write_positions!(positions, xs, ys, zs)
    @inbounds for i in eachindex(positions)
        positions[i] = SVector{3,eltype(xs)}(xs[i], ys[i], zs[i])
    end
    return positions
end