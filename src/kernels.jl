struct ForceTorque{T<:AbstractFloat}
    fx::T
    fy::T
    fz::T
    tx::T
    ty::T
    tz::T
end

@inline function zero_force(::Type{T}) where {T<:AbstractFloat}
    z = zero(T)
    return ForceTorque{T}(z, z, z, z, z, z)
end

@inline function add_force(a::ForceTorque{T}, b::ForceTorque{T}) where {T<:AbstractFloat}
    return ForceTorque{T}(
        a.fx + b.fx, a.fy + b.fy, a.fz + b.fz,
        a.tx + b.tx, a.ty + b.ty, a.tz + b.tz,
    )
end

@inline function cross3(ax, ay, az, bx, by, bz)
    return (
        ay * bz - az * by,
        az * bx - ax * bz,
        ax * by - ay * bx,
    )
end

struct ContactParams{T<:AbstractFloat}
    kn::T
    restitution::T
    log_e::T
    mu::T
    mu_roll::T
    zeta_roll::T
    Lx::T
    Ly::T
    z0::T
    z1::T
    dedupe::Int
end

@inline function normal_gamma(kn, mstar, restitution, log_e)
    T = typeof(kn)
    if restitution >= one(T)
        return zero(T)
    elseif restitution < T(1e-3)
        return 2 * sqrt(kn * mstar)
    else
        return -2 * log_e * sqrt(kn * mstar) /
            sqrt(T(π)^2 + log_e * log_e)
    end
end

@inline function resist_rotation(wx, wy, wz, fn, Reff, mstar, Istar,
                                p::ContactParams{T}) where {T}
    ω2 = wx * wx + wy * wy + wz * wz
    if fn <= zero(T) || ω2 == zero(T) || p.mu_roll == zero(T) || Istar <= zero(T)
        return (zero(T), zero(T), zero(T))
    end
    ω = sqrt(ω2)
    γ = p.zeta_roll * Istar * sqrt(p.kn / mstar)
    mag = min(p.mu_roll * Reff * fn, γ * ω)
    s = -mag / ω
    return (s * wx, s * wy, s * wz)
end

@inline function pair_force(
    xi, yi, zi, vxi, vyi, vzi, wxi, wyi, wzi, ri, mi, Ii,
    xj, yj, zj, vxj, vyj, vzj, wxj, wyj, wzj, rj, mj, Ij,
    p::ContactParams{T},
) where {T}
    dx = minimum_image(xj - xi, p.Lx)
    dy = minimum_image(yj - yi, p.Ly)
    dz = zj - zi
    dist2 = dx * dx + dy * dy + dz * dz
    rad = ri + rj

    if dist2 >= rad * rad || dist2 == zero(T)
        return zero_force(T)
    end

    dist = sqrt(dist2)
    inv = one(T) / dist
    nx = dx * inv
    ny = dy * inv
    nz = dz * inv
    δ = rad - dist

    vrx = vxi - vxj
    vry = vyi - vyj
    vrz = vzi - vzj
    v_approach = vrx * nx + vry * ny + vrz * nz

    mstar = mi * mj / (mi + mj)
    γn = normal_gamma(p.kn, mstar, p.restitution, p.log_e)
    fn = p.kn * δ + γn * v_approach
    if fn <= zero(T)
        return zero_force(T)
    end

    ri_c = max(ri - T(0.5) * δ, T(0.05) * ri)
    rj_c = max(rj - T(0.5) * δ, T(0.05) * rj)
    rix, riy, riz = nx * ri_c, ny * ri_c, nz * ri_c
    rjx, rjy, rjz = -nx * rj_c, -ny * rj_c, -nz * rj_c

    wix, wiy, wiz = cross3(wxi, wyi, wzi, rix, riy, riz)
    wjx, wjy, wjz = cross3(wxj, wyj, wzj, rjx, rjy, rjz)
    sx = vrx + wix - wjx
    sy = vry + wiy - wjy
    sz = vrz + wiz - wjz
    sn = sx * nx + sy * ny + sz * nz
    tx = sx - sn * nx
    ty = sy - sn * ny
    tz = sz - sn * nz
    tmag = sqrt(tx * tx + ty * ty + tz * tz)

    fx = -fn * nx
    fy = -fn * ny
    fz = -fn * nz
    tqx = zero(T)
    tqy = zero(T)
    tqz = zero(T)

    if tmag > zero(T) && p.mu > zero(T)
        ft = min(p.mu * fn, γn * tmag)
        s = -ft / tmag
        ftx = s * tx
        fty = s * ty
        ftz = s * tz
        fx += ftx
        fy += fty
        fz += ftz
        cx, cy, cz = cross3(rix, riy, riz, ftx, fty, ftz)
        tqx += cx
        tqy += cy
        tqz += cz
    end

    Istar = Ii * Ij / (Ii + Ij)
    Reff = ri * rj / rad
    rx, ry, rz = resist_rotation(
        wxi - wxj, wyi - wyj, wzi - wzj, fn, Reff, mstar, Istar, p,
    )

    return ForceTorque{T}(fx, fy, fz, tqx + rx, tqy + ry, tqz + rz)
end

@inline function floor_force(
    z, vx, vy, vz, wx, wy, wz, radius, mass, inertia,
    p::ContactParams{T},
) where {T}
    δ = radius - z
    if δ <= zero(T)
        return zero_force(T)
    end

    γn = normal_gamma(p.kn, mass, p.restitution, p.log_e)
    v_approach = -vz
    fn = p.kn * δ + γn * v_approach
    if fn <= zero(T)
        return zero_force(T)
    end

    ri_c = max(radius - T(0.5) * δ, T(0.05) * radius)

    sx = vx - wy * ri_c
    sy = vy + wx * ri_c
    tmag = sqrt(sx * sx + sy * sy)

    fx = zero(T)
    fy = zero(T)
    fz = fn
    tqx = zero(T)
    tqy = zero(T)
    tqz = zero(T)

    if tmag > zero(T) && p.mu > zero(T)
        ft = min(p.mu * fn, γn * tmag)
        ftx = -ft / tmag * sx
        fty = -ft / tmag * sy
        fx += ftx
        fy += fty
        tqx += ri_c * fty
        tqy += -ri_c * ftx
    end

    rx, ry, rz = resist_rotation(wx, wy, wz, fn, radius, mass, inertia, p)
    return ForceTorque{T}(fx, fy, fz, tqx + rx, tqy + ry, tqz + rz)
end

@kernel function clear_cells_kernel!(counts, overflow)
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
    nx::Int, ny::Int, nz::Int, ncells::Int, max_per_cell::Int,
    Lx::T, Ly::T, z0::T, z1::T, n::Int,
) where {T}
    i = @index(Global, Linear)
    if i <= n
        @inbounds begin
            ix = cell_index_1d(x[i], Lx, nx)
            iy = cell_index_1d(y[i], Ly, ny)
            iz = cell_index_z(z[i], z0, z1, nz)
            c = cell_linear(ix, iy, iz, nx, ny)
            slot = @atomic counts[c] += Int32(1)
            if Int(slot) <= max_per_cell
                ids[(Int(slot) - 1) * ncells + c] = Int32(i)
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
    p::ContactParams{T},
    nx::Int, ny::Int, nz::Int, ncells::Int, max_per_cell::Int, n::Int,
    gx::T, gy::T, gz::T,
) where {T}
    i = @index(Global, Linear)
    if i <= n
        @inbounds begin
            xi = x[i]; yi = y[i]; zi = z[i]
            vxi = vx[i]; vyi = vy[i]; vzi = vz[i]
            wxi = wx[i]; wyi = wy[i]; wzi = wz[i]
            ri = radius[i]; mi = mass[i]; Ii = inertia[i]

            acc = floor_force(
                zi, vxi, vyi, vzi, wxi, wyi, wzi, ri, mi, Ii, p,
            )

            ix = cell_index_1d(xi, p.Lx, nx)
            iy = cell_index_1d(yi, p.Ly, ny)
            iz = cell_index_z(zi, p.z0, p.z1, nz)

            if p.dedupe == 0
                for dz in -1:1, dy in -1:1, dx in -1:1
                    jz = iz + dz
                    (jz < 1 || jz > nz) && continue
                    jx = wrap_cell(ix + dx, nx)
                    jy = wrap_cell(iy + dy, ny)
                    c = cell_linear(jx, jy, jz, nx, ny)
                    cnt = min(Int(counts[c]), max_per_cell)
                    for slot in 1:cnt
                        j = Int(ids[(slot - 1) * ncells + c])
                        if j != i && j >= 1
                            acc = add_force(acc, pair_force(
                                xi, yi, zi, vxi, vyi, vzi,
                                wxi, wyi, wzi, ri, mi, Ii,
                                x[j], y[j], z[j],
                                vx[j], vy[j], vz[j],
                                wx[j], wy[j], wz[j],
                                radius[j], mass[j], inertia[j],
                                p,
                            ))
                        end
                    end
                end
            else
                seen = MVector{27,Int}(undef)
                nseen = 0
                for dz in -1:1, dy in -1:1, dx in -1:1
                    jz = iz + dz
                    (jz < 1 || jz > nz) && continue
                    c = cell_linear(
                        wrap_cell(ix + dx, nx),
                        wrap_cell(iy + dy, ny),
                        jz, nx, ny,
                    )
                    duplicate = false
                    for s in 1:nseen
                        if seen[s] == c
                            duplicate = true
                            break
                        end
                    end
                    duplicate && continue
                    nseen += 1
                    seen[nseen] = c

                    cnt = min(Int(counts[c]), max_per_cell)
                    for slot in 1:cnt
                        j = Int(ids[(slot - 1) * ncells + c])
                        if j != i && j >= 1
                            acc = add_force(acc, pair_force(
                                xi, yi, zi, vxi, vyi, vzi,
                                wxi, wyi, wzi, ri, mi, Ii,
                                x[j], y[j], z[j],
                                vx[j], vy[j], vz[j],
                                wx[j], wy[j], wz[j],
                                radius[j], mass[j], inertia[j],
                                p,
                            ))
                        end
                    end
                end
            end

            fx[i] = acc.fx + mi * gx
            fy[i] = acc.fy + mi * gy
            fz[i] = acc.fz + mi * gz
            tx[i] = acc.tx
            ty[i] = acc.ty
            tz[i] = acc.tz
        end
    end
end

function launch_clear_cells!(grid::CellGrid, backend)
    clear_cells_kernel!(backend)(
        grid.counts, grid.overflow; ndrange = max(grid.ncells, 1),
    )
    return nothing
end

function launch_bin!(
    grid::CellGrid,
    particles::ParticleArrays{T},
    params::ContactParams{T},
    backend,
) where {T}
    n = length(particles)
    bin_kernel!(backend)(
        grid.ids, grid.counts, grid.overflow,
        particles.x, particles.y, particles.z,
        grid.nx, grid.ny, grid.nz, grid.ncells, grid.max_per_cell,
        params.Lx, params.Ly, params.z0, params.z1, n;
        ndrange = n,
    )
    return nothing
end

function launch_force!(
    grid::CellGrid,
    particles::ParticleArrays{T},
    params::ContactParams{T},
    backend,
    gravity::SVector{3,T},
) where {T}
    n = length(particles)
    force_kernel!(backend)(
        particles.fx, particles.fy, particles.fz,
        particles.tx, particles.ty, particles.tz,
        particles.x, particles.y, particles.z,
        particles.vx, particles.vy, particles.vz,
        particles.wx, particles.wy, particles.wz,
        particles.radius, particles.mass, particles.inertia,
        grid.counts, grid.ids,
        params,
        grid.nx, grid.ny, grid.nz, grid.ncells, grid.max_per_cell, n,
        gravity[1], gravity[2], gravity[3];
        ndrange = n,
    )
    return nothing
end