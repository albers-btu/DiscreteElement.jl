const _GL64_X, _GL64_W = let
    n = 32
    x = zeros(n)
    w = zeros(n)
    for i in 1:((n + 1) ÷ 2)
        z = cos(π * (i - 0.25) / (n + 0.5))
        pp = 0.0
        for _ in 1:60
            p1 = 1.0
            p2 = 0.0
            for j in 1:n
                p3 = p2
                p2 = p1
                p1 = ((2j - 1) * z * p2 - (j - 1) * p3) / j
            end
            pp = n * (z * p1 - p2) / (z * z - 1)
            z1 = z
            z = z1 - p1 / pp
            abs(z - z1) < 1e-15 && break
        end
        x[i] = -z
        x[n + 1 - i] = z
        w[i] = 2 / ((1 - z * z) * pp * pp)
        w[n + 1 - i] = w[i]
    end
    x, w
end

function _circle_primitive(rad, x)
    s = sqrt(max(rad * rad - x * x, 0))
    return 0.5 * (x * s + rad * rad * asin(clamp(x / rad, -1, 1)))
end

function disk_rect_area(rad, x0, x1, y0, y1)
    rad <= 0 && return 0.0
    x0 = max(x0, -rad)
    x1 = min(x1, rad)
    y0 = max(y0, -rad)
    y1 = min(y1, rad)
    (x1 <= x0 || y1 <= y0) && return 0.0
    xs = Float64[x0, x1]
    for yb in (y0, y1)
        ay = abs(yb)
        if 0 < ay < rad
            xb = sqrt(rad * rad - ay * ay)
            for s in (xb, -xb)
                if x0 < s < x1
                    push!(xs, s)
                end
            end
        end
    end
    sort!(xs)
    nkeep = 1
    for k in 2:length(xs)
        if xs[k] > xs[nkeep] + 1e-14 * max(1.0, abs(xs[k]))
            nkeep += 1
            xs[nkeep] = xs[k]
        end
    end
    area = 0.0
    for k in 1:(nkeep - 1)
        a = xs[k]
        b = xs[k + 1]
        b <= a && continue
        xm = 0.5 * (a + b)
        sm = sqrt(max(rad * rad - xm * xm, 0))
        # Strict: equality holds only on a breakpoint, where the chord is the
        # circle itself. Using the midpoint of a full-diameter interval would
        # otherwise treat the bounding square as the disk.
        lo_is_bound = y0 > -sm
        hi_is_bound = y1 < sm
        lo = lo_is_bound ? y0 : -sm
        hi = hi_is_bound ? y1 : sm
        hi <= lo && continue
        fa = _circle_primitive(rad, a)
        fb = _circle_primitive(rad, b)
        if lo_is_bound && hi_is_bound
            area += (y1 - y0) * (b - a)
        elseif lo_is_bound && !hi_is_bound
            area += (fb - fa) - y0 * (b - a)
        elseif !lo_is_bound && hi_is_bound
            area += y1 * (b - a) + (fb - fa)
        else
            area += 2 * (fb - fa)
        end
    end
    return max(area, 0)
end

function sphere_aabb_volume(cx, cy, cz, radius, xmin, xmax, ymin, ymax, zmin, zmax)
    radius <= 0 && return 0.0
    (xmax <= xmin || ymax <= ymin || zmax <= zmin) && return 0.0
    x0 = xmin - cx
    x1 = xmax - cx
    y0 = ymin - cy
    y1 = ymax - cy
    z0 = zmin - cz
    z1 = zmax - cz
    function axis_outside(a, b)
        return b < -radius || a > radius
    end
    if axis_outside(x0, x1) || axis_outside(y0, y1) || axis_outside(z0, z1)
        return 0.0
    end
    dx = (x0 > 0 ? x0 : (x1 < 0 ? x1 : 0.0))
    dy = (y0 > 0 ? y0 : (y1 < 0 ? y1 : 0.0))
    dz = (z0 > 0 ? z0 : (z1 < 0 ? z1 : 0.0))
    dx * dx + dy * dy + dz * dz > radius * radius && return 0.0
    if x0 <= -radius && x1 >= radius && y0 <= -radius && y1 >= radius &&
            z0 <= -radius && z1 >= radius
        return (4 / 3) * π * radius^3
    end
    r2 = radius * radius
    inside = true
    for x in (x0, x1), y in (y0, y1), z in (z0, z1)
        if x * x + y * y + z * z > r2
            inside = false
            break
        end
    end
    inside && return (x1 - x0) * (y1 - y0) * (z1 - z0)

    za = max(z0, -radius)
    zb = min(z1, radius)
    zb <= za && return 0.0
    # Split where the slice radius crosses a rectangle edge. Between those
    # points the slice area is smooth, so Gauss–Legendre stays accurate.
    zs = Float64[za, zb]
    for edge in (x0, x1, y0, y1)
        d = abs(edge)
        if 0 < d < radius
            zk = sqrt(radius * radius - d * d)
            for s in (zk, -zk)
                if za < s < zb
                    push!(zs, s)
                end
            end
        end
    end
    sort!(zs)
    volume = 0.0
    @inbounds for k in 1:(length(zs) - 1)
        a = zs[k]
        b = zs[k + 1]
        b <= a + 1e-15 * radius && continue
        mid = 0.5 * (a + b)
        half = 0.5 * (b - a)
        piece = 0.0
        for i in eachindex(_GL64_X)
            z = mid + half * _GL64_X[i]
            slice = sqrt(max(radius * radius - z * z, 0))
            piece += _GL64_W[i] * disk_rect_area(slice, x0, x1, y0, y1)
        end
        volume += piece * half
    end
    box = (xmax - xmin) * (ymax - ymin) * (zmax - zmin)
    return clamp(volume, 0, min((4 / 3) * π * radius^3, box))
end

function _accumulate_sphere!(phi, cx, cy, cz, radius, origin, dx, dy, dz, Lx, Ly, cell, n_sub)
    Nx, Ny, Nz = size(phi)
    x_max = origin[1] + Nx * dx
    y_max = origin[2] + Ny * dy
    z_max = origin[3] + Nz * dz
    for oy in -1:1, ox in -1:1
        sx = cx + ox * Lx
        sy = cy + oy * Ly
        if sx + radius <= origin[1] || sx - radius >= x_max ||
                sy + radius <= origin[2] || sy - radius >= y_max ||
                cz + radius <= origin[3] || cz - radius >= z_max
            continue
        end
        i0 = max(1, floor(Int, (sx - radius - origin[1]) / dx) + 1)
        i1 = min(Nx, floor(Int, (sx + radius - origin[1]) / dx) + 1)
        j0 = max(1, floor(Int, (sy - radius - origin[2]) / dy) + 1)
        j1 = min(Ny, floor(Int, (sy + radius - origin[2]) / dy) + 1)
        k0 = max(n_sub + 1, floor(Int, (cz - radius - origin[3]) / dz) + 1)
        k1 = min(Nz, floor(Int, (cz + radius - origin[3]) / dz) + 1)
        for k in k0:k1
            z0 = origin[3] + (k - 1) * dz
            z1 = z0 + dz
            for j in j0:j1
                y0 = origin[2] + (j - 1) * dy
                y1 = y0 + dy
                for i in i0:i1
                    x0 = origin[1] + (i - 1) * dx
                    x1 = x0 + dx
                    vol = sphere_aabb_volume(sx, sy, cz, radius, x0, x1, y0, y1, z0, z1)
                    vol <= 0 && continue
                    phi[i, j, k] += vol / cell
                end
            end
        end
    end
    return phi
end

function grid_cell_size(
    domain::AbstractVector{<:Real},
    resolution,
)
    length(domain) == 3 ||
        throw(ArgumentError("domain must have three components"))
    Lx, Ly, _ = Float64.(domain)
    (Lx > 0 && Ly > 0) ||
        throw(ArgumentError("domain sides must be positive"))

    target = Float64(as_si(resolution, u"m"))
    target > 0 || throw(ArgumentError("resolution must be positive"))

    Nx = max(1, ceil(Int, Lx / target - 1e-12))
    dx = Lx / Nx

    Ny = max(1, ceil(Int, Ly / dx - 1e-12))
    dy = Ly / Ny

    return (;
        cell_size = dx * u"m",
        dx = dx * u"m",
        dy = dy * u"m",
        Nx,
        Ny,
    )
end

function metal_fraction_grid(
    particles::ParticleArrays,
    domain::AbstractVector{<:Real};
    substrate_thickness = 0,
    cell_size,
)
    length(domain) == 3 || throw(ArgumentError("domain must have three components"))
    n = length(particles)
    n > 0 || throw(ArgumentError("particles must not be empty"))
    x = Array(particles.x)
    y = Array(particles.y)
    z = Array(particles.z)
    radius = Array(particles.radius)

    Lx, Ly, Lz = Float64.(domain)
    (Lx > 0 && Ly > 0 && Lz > 0) ||
        throw(ArgumentError("domain sides must be positive"))

    target = Float64(as_si(cell_size, u"m"))
    target > 0 || throw(ArgumentError("cell_size must be positive"))

    thickness = substrate_thickness === nothing ? 0.0 :
        Float64(as_si(substrate_thickness, u"m"))
    thickness >= 0 ||
        throw(ArgumentError("substrate_thickness must be non-negative"))

    Nx = max(1, ceil(Int, Lx / target - 1e-12))
    dx_m = Lx / Nx
    Ny = max(1, ceil(Int, Ly / dx_m - 1e-12))
    dy_m = Ly / Ny
    dz_m = dx_m
    z_hi = maximum(z .+ radius)
    n_powder = max(1, ceil(Int, z_hi / dz_m - 1e-12))
    n_sub = thickness > 0 ? max(1, round(Int, thickness / dz_m)) : 0
    Nz = n_sub + n_powder
    origin = (0.0, 0.0, -n_sub * dz_m)

    phi = zeros(Float64, Nx, Ny, Nz)
    cell = dx_m * dy_m * dz_m
    for i in 1:n
        _accumulate_sphere!(
            phi, x[i], y[i], z[i], radius[i],
            origin, dx_m, dy_m, dz_m, Lx, Ly, cell, n_sub,
        )
    end
    phi .= clamp.(phi, 0, 1)

    substrate = nothing
    if n_sub > 0
        substrate = zeros(UInt8, Nx, Ny, Nz)
        substrate[:, :, 1:n_sub] .= UInt8(1)
        phi[:, :, 1:n_sub] .= 0
    end
    return (;
        phi,
        substrate,
        dx = dx_m,
        dy = dy_m,
        dz = dz_m,
        origin,
        substrate_thickness = n_sub * dz_m,
    )
end

function write_powder_bed(
    path::AbstractString,
    phi::AbstractArray{<:Real,3},
    dx::Real,
    dy::Real,
    dz::Real,
    origin::NTuple{3,Real};
    substrate::Union{Nothing,AbstractArray{UInt8,3}} = nothing,
)
    phi32 = Float32.(clamp.(phi, 0, 1))
    HDF5.h5open(path, "w") do file
        file["phi"] = phi32
        if substrate !== nothing
            size(substrate) == size(phi32) ||
                error("substrate size $(size(substrate)) != phi size $(size(phi32))")
            file["substrate"] = substrate
        end
        HDF5.attributes(file)["dx"] = Float64(dx)
        HDF5.attributes(file)["dy"] = Float64(dy)
        HDF5.attributes(file)["dz"] = Float64(dz)
        HDF5.attributes(file)["origin"] = Float64[origin...]
        HDF5.attributes(file)["index_order"] = "x_fastest"
        HDF5.attributes(file)["cell_center"] = "integer"
        HDF5.attributes(file)["quantity"] = "metal_volume_fraction"
    end
    return nothing
end

function hdf5(
    particles::ParticleArrays,
    domain::AbstractVector{<:Real};
    name::AbstractString,
    substrate_thickness = nothing,
    cell_size,
)
    n = length(particles)
    n > 0 || throw(ArgumentError("particles must not be empty"))
    thickness = substrate_thickness === nothing ? 0.0 :
        Float64(as_si(substrate_thickness, u"m"))

    grid = metal_fraction_grid(
        particles, domain;
        substrate_thickness = thickness, cell_size,
    )
    directory = dirname(name)
    isempty(directory) || mkpath(directory)
    write_powder_bed(
        name, grid.phi, grid.dx, grid.dy, grid.dz, grid.origin;
        substrate = grid.substrate,
    )
    return (;
        name = String(name),
        size = size(grid.phi),
        dx = grid.dx * u"m",
        dy = grid.dy * u"m",
        dz = grid.dz * u"m",
        origin = grid.origin .* u"m",
        substrate_thickness = grid.substrate_thickness * u"m",
    )
end