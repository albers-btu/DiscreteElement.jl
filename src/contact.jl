struct ForceTorque{T<:AbstractFloat}
    fx::T
    fy::T
    fz::T
    tx::T
    ty::T
    tz::T
end

@inline function zero_force(::Type{T}) where {T}
    z = zero(T)
    return ForceTorque{T}(z, z, z, z, z, z)
end

@inline function add_force(a::ForceTorque{T}, b::ForceTorque{T}) where {T}
    return ForceTorque{T}(
        a.fx + b.fx, a.fy + b.fy, a.fz + b.fz,
        a.tx + b.tx, a.ty + b.ty, a.tz + b.tz,
    )
end

@inline function wrap_position(x, L)
    return x - L * floor(x / L)
end

@inline function minimum_image(dx, L)
    return dx - L * round(dx / L)
end

@inline function cross3(ax, ay, az, bx, by, bz)
    return (
        ay * bz - az * by,
        az * bx - ax * bz,
        ax * by - ay * bx,
    )
end

@inline function cell_index_1d(x, L, n::Int)
    xw = wrap_position(x, L)
    ix = unsafe_trunc(Int, xw / L * n) + 1
    return ix < 1 ? 1 : (ix > n ? n : ix)
end

@inline function cell_index_z(z, z0, z1, nz::Int)
    iz = unsafe_trunc(Int, (z - z0) / (z1 - z0) * nz) + 1
    return iz < 1 ? 1 : (iz > nz ? nz : iz)
end

@inline function cell_linear(ix::Int, iy::Int, iz::Int, nx::Int, ny::Int)
    return ix + (iy - 1) * nx + (iz - 1) * nx * ny
end

@inline function wrap_cell(i::Int, n::Int)
    r = i % n
    return r <= 0 ? r + n : r
end

"""Normal dashpot that reproduces `restitution` for a linear spring."""
@inline function normal_gamma(kn, mstar, restitution, log_restitution)
    T = typeof(kn)
    if restitution >= one(T)
        return zero(T)
    elseif restitution < T(1e-3)
        return 2 * sqrt(kn * mstar)
    else
        return -2 * log_restitution * sqrt(kn * mstar) /
            sqrt(T(π)^2 + log_restitution * log_restitution)
    end
end

@inline function resist_rotation(wx, wy, wz, fn, Reff, mstar, Istar, p::SimParams{T}) where {T}
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

"""
Force and torque on particle `i` from an overlap with particle `j`.

`n` points from `i` toward `j`. The force is repulsive. Tangential friction is
a dashpot capped by Coulomb's law, so there is no tangential spring to store.
Rolling resistance opposes the full relative angular velocity and is capped
by `mu_roll * R_eff * fn`.
"""
@inline function pair_force(
    xi, yi, zi, vxi, vyi, vzi, wxi, wyi, wzi, ri, mi, Ii,
    xj, yj, zj, vxj, vyj, vzj, wxj, wyj, wzj, rj, mj, Ij,
    p::SimParams{T},
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
    γn = normal_gamma(p.kn, mstar, p.restitution, p.log_restitution)
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
    if tmag > zero(T) && p.mu_slide > zero(T)
        ft = min(p.mu_slide * fn, γn * tmag)
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
        wxi - wxj, wyi - wyj, wzi - wzj, fn, Reff, mstar, Istar, p
    )
    return ForceTorque{T}(fx, fy, fz, tqx + rx, tqy + ry, tqz + rz)
end

"""Force and torque on a sphere from the rigid floor `z = 0`."""
@inline function floor_force(
    z, vx, vy, vz, wx, wy, wz, radius, mass, inertia, p::SimParams{T}
) where {T}
    δ = radius - z
    if δ <= zero(T)
        return zero_force(T)
    end
    γn = normal_gamma(p.kn, mass, p.restitution, p.log_restitution)
    # n points from the particle toward the floor, so n = (0, 0, -1).
    v_approach = -vz
    fn = p.kn * δ + γn * v_approach
    if fn <= zero(T)
        return zero_force(T)
    end
    ri_c = max(radius - T(0.5) * δ, T(0.05) * radius)
    # Velocity of the contact point on the sphere.
    sx = vx - wy * ri_c
    sy = vy + wx * ri_c
    tmag = sqrt(sx * sx + sy * sy)
    fx = zero(T)
    fy = zero(T)
    fz = fn
    tqx = zero(T)
    tqy = zero(T)
    tqz = zero(T)
    if tmag > zero(T) && p.mu_slide > zero(T)
        ft = min(p.mu_slide * fn, γn * tmag)
        ftx = -ft / tmag * sx
        fty = -ft / tmag * sy
        fx += ftx
        fy += fty
        # r_contact × F_t with r_contact = (0, 0, -ri_c)
        tqx += ri_c * fty
        tqy += -ri_c * ftx
    end
    rx, ry, rz = resist_rotation(wx, wy, wz, fn, radius, mass, inertia, p)
    return ForceTorque{T}(fx, fy, fz, tqx + rx, tqy + ry, tqz + rz)
end

"""
    contact_parameters(radii, material; drop_height, overlap_fraction=0.08, dt_factor=0.25)

Reduced stiffness and a stable timestep. `kn` is chosen so that a wall impact
of the smallest particle, at the speed gained by falling `drop_height`,
overlaps about `overlap_fraction` of its radius. Settled overlaps are far
smaller, because a particle's weight is tiny next to that impact force.
`dt` is `dt_factor / ω` of that wall contact. `drop_height` is meters or a
length. The returned stiffness is `N/m` and the timestep is seconds.
"""
function contact_parameters(
    radii,
    material::Material{T};
    drop_height,
    overlap_fraction = 0.08,
    dt_factor = 0.25,
) where {T}
    drop_height = T(as_si(drop_height, u"m"))
    overlap_fraction = T(as_dimensionless(overlap_fraction))
    dt_factor = T(as_dimensionless(dt_factor))
    0 < overlap_fraction < 0.5 ||
        throw(ArgumentError("overlap_fraction must lie in (0, 0.5)"))
    dt_factor > 0 || throw(ArgumentError("dt_factor must be positive"))
    drop_height > 0 || throw(ArgumentError("drop_height must be positive"))
    rmin = minimum(T(as_si(r, u"m")) for r in radii)
    rmin > 0 || throw(ArgumentError("radii must be positive"))
    mmin = sphere_mass(T(rmin), material.density)
    vchar = sqrt(2 * material.gravity * drop_height)
    vchar = max(vchar, T(1e-4))
    δ = overlap_fraction * rmin
    kn = mmin * vchar^2 / δ^2
    ω = sqrt(kn / mmin)
    dt = dt_factor / ω
    return kn * u"N/m", dt * u"s"
end

function make_params(
    material::Material{T},
    domain::Domain{T},
    grid::CellGrid;
    kn::T,
    dt::T,
    n::Int,
    z0::T,
    z1::T,
) where {T}
    e = material.restitution
    log_e = e >= one(T) ? zero(T) : log(max(e, T(1e-8)))
    dedupe = (grid.nx < 3 || grid.ny < 3) ? 1 : 0
    return SimParams{T}(
        kn,
        e,
        log_e,
        material.mu_slide,
        material.mu_roll,
        material.zeta_roll,
        material.gravity,
        domain.Lx,
        domain.Ly,
        z0,
        z1,
        dt,
        grid.nx,
        grid.ny,
        grid.nz,
        n,
        grid.max_per_cell,
        grid.ncells,
        dedupe,
    )
end
