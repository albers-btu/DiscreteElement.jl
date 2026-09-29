@inline function accumulate_cell(
    acc::ForceTorque{T}, c, i,
    xi, yi, zi, vxi, vyi, vzi, wxi, wyi, wzi, ri, mi, Ii,
    x, y, z, vx, vy, vz, wx, wy, wz, radius, mass, inertia,
    counts, ids, p::SimParams{T},
) where {T}
    cnt = Int(@inbounds counts[c])
    if cnt > p.max_per_cell
        cnt = p.max_per_cell
    end
    @inbounds for slot in 1:cnt
        j = Int(ids[(slot - 1) * p.ncells + c])
        if j != i && j >= 1
            acc = add_force(acc, pair_force(
                xi, yi, zi, vxi, vyi, vzi, wxi, wyi, wzi, ri, mi, Ii,
                x[j], y[j], z[j], vx[j], vy[j], vz[j], wx[j], wy[j], wz[j],
                radius[j], mass[j], inertia[j],
                p,
            ))
        end
    end
    return acc
end

@kernel function clear_cells!(counts, overflow)
    i = @index(Global, Linear)
    if i <= length(counts)
        @inbounds counts[i] = Int32(0)
    end
    if i == 1
        @inbounds overflow[1] = Int32(0)
    end
end

@kernel function bin_kernel!(
    ids, counts, overflow,
    @Const(x), @Const(y), @Const(z),
    p::SimParams{T},
) where {T}
    i = @index(Global, Linear)
    if i <= p.n
        @inbounds begin
            ix = cell_index_1d(x[i], p.Lx, p.nx)
            iy = cell_index_1d(y[i], p.Ly, p.ny)
            iz = cell_index_z(z[i], p.z0, p.z1, p.nz)
            c = cell_linear(ix, iy, iz, p.nx, p.ny)
            slot = @atomic counts[c] += Int32(1)
            if Int(slot) <= p.max_per_cell
                ids[(Int(slot) - 1) * p.ncells + c] = Int32(i)
            else
                @atomic overflow[1] += Int32(1)
            end
        end
    end
end

@kernel function force_kernel!(
    fx, fy, fz, tx, ty, tz,
    @Const(x), @Const(y), @Const(z),
    @Const(vx), @Const(vy), @Const(vz),
    @Const(wx), @Const(wy), @Const(wz),
    @Const(radius), @Const(mass), @Const(inertia),
    @Const(counts), @Const(ids),
    p::SimParams{T},
) where {T}
    i = @index(Global, Linear)
    if i <= p.n
    @inbounds begin
        xi = x[i]
        yi = y[i]
        zi = z[i]
        vxi = vx[i]
        vyi = vy[i]
        vzi = vz[i]
        wxi = wx[i]
        wyi = wy[i]
        wzi = wz[i]
        ri = radius[i]
        mi = mass[i]
        Ii = inertia[i]

        acc = floor_force(zi, vxi, vyi, vzi, wxi, wyi, wzi, ri, mi, Ii, p)

        ix = cell_index_1d(xi, p.Lx, p.nx)
        iy = cell_index_1d(yi, p.Ly, p.ny)
        iz = cell_index_z(zi, p.z0, p.z1, p.nz)
        if p.dedupe == 0
            for dz in -1:1, dy in -1:1, dx in -1:1
                jz = iz + dz
                if jz < 1 || jz > p.nz
                    continue
                end
                c = cell_linear(wrap_cell(ix + dx, p.nx), wrap_cell(iy + dy, p.ny), jz, p.nx, p.ny)
                acc = accumulate_cell(acc, c, i, xi, yi, zi, vxi, vyi, vzi, wxi, wyi, wzi, ri, mi, Ii, x, y, z, vx, vy, vz, wx, wy, wz, radius, mass, inertia, counts, ids, p)
            end
        else
            seen = MVector{27,Int}(undef)
            nseen = 0
            for dz in -1:1, dy in -1:1, dx in -1:1
                jz = iz + dz
                if jz < 1 || jz > p.nz
                    continue
                end
                c = cell_linear(wrap_cell(ix + dx, p.nx), wrap_cell(iy + dy, p.ny), jz, p.nx, p.ny)
                dup = false
                for s in 1:nseen
                    if seen[s] == c
                        dup = true
                        break
                    end
                end
                if dup
                    continue
                end
                nseen += 1
                seen[nseen] = c
                acc = accumulate_cell(acc, c, i, xi, yi, zi, vxi, vyi, vzi, wxi, wyi, wzi, ri, mi, Ii, x, y, z, vx, vy, vz, wx, wy, wz, radius, mass, inertia, counts, ids, p)
            end
        end

        fx[i] = acc.fx
        fy[i] = acc.fy
        fz[i] = acc.fz - mi * p.gravity
        tx[i] = acc.tx
        ty[i] = acc.ty
        tz[i] = acc.tz
    end
    end
end

@kernel function integrate_kernel!(
    x, y, z, vx, vy, vz, wx, wy, wz,
    @Const(fx), @Const(fy), @Const(fz),
    @Const(tx), @Const(ty), @Const(tz),
    @Const(mass), @Const(inertia),
    p::SimParams{T},
) where {T}
    i = @index(Global, Linear)
    if i <= p.n
    @inbounds begin
        vx[i] = vx[i] + fx[i] / mass[i] * p.dt
        vy[i] = vy[i] + fy[i] / mass[i] * p.dt
        vz[i] = vz[i] + fz[i] / mass[i] * p.dt
        wx[i] = wx[i] + tx[i] / inertia[i] * p.dt
        wy[i] = wy[i] + ty[i] / inertia[i] * p.dt
        wz[i] = wz[i] + tz[i] / inertia[i] * p.dt
        x[i] = wrap_position(x[i] + vx[i] * p.dt, p.Lx)
        y[i] = wrap_position(y[i] + vy[i] * p.dt, p.Ly)
        z[i] = z[i] + vz[i] * p.dt
    end
    end
end

@kernel function kinetic_kernel!(
    ke, @Const(vx), @Const(vy), @Const(vz),
    @Const(wx), @Const(wy), @Const(wz),
    @Const(mass), @Const(inertia), n::Int,
)
    i = @index(Global, Linear)
    if i <= n
        @inbounds begin
            half = oftype(mass[i], 0.5)
            ke[i] = half * mass[i] * (vx[i]^2 + vy[i]^2 + vz[i]^2) +
                half * inertia[i] * (wx[i]^2 + wy[i]^2 + wz[i]^2)
        end
    end
end

@kernel function sdf_kernel!(
    sdf,
    @Const(x), @Const(y), @Const(z), @Const(radius),
    @Const(counts), @Const(ids),
    p::SimParams{T},
    x0, dx, y0, dy, z0s, dz,
    box_z0, box_z1,
) where {T}
    i, j, k = @index(Global, NTuple)
    px = x0 + (i - 1) * dx
    py = y0 + (j - 1) * dy
    pz = z0s + (k - 1) * dz
    pxw = wrap_position(px, p.Lx)
    pyw = wrap_position(py, p.Ly)
    ix = cell_index_1d(pxw, p.Lx, p.nx)
    iy = cell_index_1d(pyw, p.Ly, p.ny)
    iz = cell_index_z(pz, p.z0, p.z1, p.nz)

    nearest = T(1e6)
    @inbounds for oz in -1:1, oy in -1:1, ox in -1:1
        jz = iz + oz
        if jz < 1 || jz > p.nz
            continue
        end
        jx = wrap_cell(ix + ox, p.nx)
        jy = wrap_cell(iy + oy, p.ny)
        c = cell_linear(jx, jy, jz, p.nx, p.ny)
        cnt = Int(counts[c])
        if cnt > p.max_per_cell
            cnt = p.max_per_cell
        end
        for slot in 1:cnt
            q = Int(ids[(slot - 1) * p.ncells + c])
            if q < 1
                continue
            end
            ddx = minimum_image(x[q] - px, p.Lx)
            ddy = minimum_image(y[q] - py, p.Ly)
            ddz = z[q] - pz
            dist = sqrt(ddx * ddx + ddy * ddy + ddz * ddz) - radius[q]
            if dist < nearest
                nearest = dist
            end
        end
    end

    qx = max(-px, px - p.Lx)
    qy = max(-py, py - p.Ly)
    qz = max(box_z0 - pz, pz - box_z1)
    dbox = max(qx, max(qy, qz))
    @inbounds sdf[i, j, k] = max(nearest, dbox)
end

function _launch(kernel, backend, ndrange, args...)
    k = kernel(backend)
    k(args...; ndrange = ndrange)
    synchronize(backend)
    return nothing
end

function clear_cells!(grid::CellGrid)
    backend = get_backend(grid.counts)
    n = max(grid.ncells, 1)
    _launch(clear_cells!, backend, n, grid.counts, grid.overflow)
    return nothing
end

function bin_particles!(grid::CellGrid, p::Particles, params::SimParams)
    clear_cells!(grid)
    backend = get_backend(p.x)
    _launch(
        bin_kernel!, backend, params.n,
        grid.ids, grid.counts, grid.overflow, p.x, p.y, p.z, params,
    )
    return nothing
end

function cell_overflow(grid::CellGrid)
    host = Vector{Int32}(undef, 1)
    copyto!(host, grid.overflow)
    return Int(host[1])
end

function compute_forces!(grid::CellGrid, p::Particles, params::SimParams)
    bin_particles!(grid, p, params)
    spilled = cell_overflow(grid)
    if spilled > 0
        error(
            "cell list overflowed ($spilled particles did not fit). " *
            "Increase max_per_cell above $(grid.max_per_cell).",
        )
    end
    backend = get_backend(p.x)
    _launch(
        force_kernel!, backend, params.n,
        p.fx, p.fy, p.fz, p.tx, p.ty, p.tz,
        p.x, p.y, p.z, p.vx, p.vy, p.vz, p.wx, p.wy, p.wz,
        p.radius, p.mass, p.inertia,
        grid.counts, grid.ids, params,
    )
    return nothing
end

function integrate!(p::Particles, params::SimParams)
    backend = get_backend(p.x)
    _launch(
        integrate_kernel!, backend, params.n,
        p.x, p.y, p.z, p.vx, p.vy, p.vz, p.wx, p.wy, p.wz,
        p.fx, p.fy, p.fz, p.tx, p.ty, p.tz,
        p.mass, p.inertia, params,
    )
    return nothing
end

function kinetic_energy!(ke, p::Particles)
    backend = get_backend(p.x)
    _launch(
        kinetic_kernel!, backend, nparticles(p),
        ke, p.vx, p.vy, p.vz, p.wx, p.wy, p.wz, p.mass, p.inertia, nparticles(p),
    )
    return sum(ke)
end
