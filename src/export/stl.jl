@inline _stl_minimum_image(dx, L) = dx - L * round(dx / L)

@inline function _stl_wrap(x, L)
    return x - L * floor(x / L)
end

@inline function _stl_cell_x(x, L, n)
    xw = _stl_wrap(x, L)
    ix = unsafe_trunc(Int, xw / L * n) + 1
    return ix < 1 ? 1 : (ix > n ? n : ix)
end

@inline function _stl_cell_z(z, z0, z1, n)
    iz = unsafe_trunc(Int, (z - z0) / (z1 - z0) * n) + 1
    return iz < 1 ? 1 : (iz > n ? n : iz)
end

@inline _stl_cell_linear(ix, iy, iz, nx, ny) =
    ix + (iy - 1) * nx + (iz - 1) * nx * ny

function _stl_axis_positions(lo, hi, cell)
    n = max(2, ceil(Int, (hi - lo) / cell) + 1)
    return range(lo, hi; length = n)
end

function _stl_build_bins(x, y, z, radius, Lx, Ly, cell)
    rmax = maximum(radius)
    bin = max(2 * rmax, cell)

    nx = max(1, ceil(Int, Lx / bin))
    ny = max(1, ceil(Int, Ly / bin))
    z0 = minimum(z) - rmax
    z1 = maximum(z) + rmax
    nz = max(1, ceil(Int, (z1 - z0) / bin))
    ncells = nx * ny * nz

    counts = zeros(Int32, ncells)
    @inbounds for i in eachindex(x)
        ix = _stl_cell_x(x[i], Lx, nx)
        iy = _stl_cell_x(y[i], Ly, ny)
        iz = _stl_cell_z(z[i], z0, z1, nz)
        counts[_stl_cell_linear(ix, iy, iz, nx, ny)] += Int32(1)
    end

    starts = Vector{Int}(undef, ncells + 1)
    starts[1] = 1
    @inbounds for c in 1:ncells
        starts[c + 1] = starts[c] + Int(counts[c])
    end

    ids = Vector{Int32}(undef, length(x))
    fill = copy(starts)
    @inbounds for i in eachindex(x)
        ix = _stl_cell_x(x[i], Lx, nx)
        iy = _stl_cell_x(y[i], Ly, ny)
        iz = _stl_cell_z(z[i], z0, z1, nz)
        c = _stl_cell_linear(ix, iy, iz, nx, ny)
        ids[fill[c]] = Int32(i)
        fill[c] += 1
    end

    return (; nx, ny, nz, z0, z1, ncells, starts, ids)
end

function _stl_sdf_grid!(
    sdf, xs, ys, zs,
    x, y, z, radius,
    Lx, Ly, box_z0, box_z1,
    thickness, plate_top, bins,
)
    Nx, Ny, Nz = size(sdf)

    nx = bins.nx
    ny = bins.ny
    nz = bins.nz
    z0 = bins.z0
    z1 = bins.z1
    starts = bins.starts
    ids = bins.ids

    ixv = Vector{Int}(undef, Nx)
    iyv = Vector{Int}(undef, Ny)
    izv = Vector{Int}(undef, Nz)
    qxv = Vector{Float64}(undef, Nx)
    qyv = Vector{Float64}(undef, Ny)

    @inbounds for i in 1:Nx
        px = xs[i]
        ixv[i] = _stl_cell_x(px, Lx, nx)
        qxv[i] = max(-px, px - Lx)
    end
    @inbounds for j in 1:Ny
        py = ys[j]
        iyv[j] = _stl_cell_x(py, Ly, ny)
        qyv[j] = max(-py, py - Ly)
    end
    @inbounds for k in 1:Nz
        izv[k] = _stl_cell_z(zs[k], z0, z1, nz)
    end

    plate = thickness > 0

    Threads.@threads for k in 1:Nz
        pz = zs[k]
        iz = izv[k]
        qz = max(box_z0 - pz, pz - box_z1)
        @inbounds for j in 1:Ny
            py = ys[j]
            iy = iyv[j]
            qy = qyv[j]
            for i in 1:Nx
                px = xs[i]
                ix = ixv[i]
                nearest = Inf

                for oz in -1:1
                    jz = iz + oz
                    (jz < 1 || jz > nz) && continue
                    for oy in -1:1
                        jy = mod1(iy + oy, ny)
                        for ox in -1:1
                            jx = mod1(ix + ox, nx)
                            c = _stl_cell_linear(jx, jy, jz, nx, ny)
                            for s in starts[c]:(starts[c + 1] - 1)
                                q = Int(ids[s])
                                ddx = _stl_minimum_image(x[q] - px, Lx)
                                ddy = _stl_minimum_image(y[q] - py, Ly)
                                ddz = z[q] - pz
                                d = sqrt(ddx * ddx + ddy * ddy + ddz * ddz) -
                                    radius[q]
                                d < nearest && (nearest = d)
                            end
                        end
                    end
                end

                dbox = max(qxv[i], qy, qz)
                d = max(nearest, dbox)
                if plate
                    dplate = max(
                        qxv[i], qy,
                        max(-thickness - pz, pz - plate_top),
                    )
                    d = min(d, dplate)
                end
                sdf[i, j, k] = d
            end
        end
    end
    return sdf
end

function _stl_sdf_point(px, py, pz, x, y, z, radius, Lx, Ly, box_z0, box_z1,
                        thickness, plate_top)
    nearest = Inf
    @inbounds for q in eachindex(x)
        ddx = _stl_minimum_image(x[q] - px, Lx)
        ddy = _stl_minimum_image(y[q] - py, Ly)
        ddz = z[q] - pz
        d = sqrt(ddx * ddx + ddy * ddy + ddz * ddz) - radius[q]
        d < nearest && (nearest = d)
    end
    qx = max(-px, px - Lx)
    qy = max(-py, py - Ly)
    qz = max(box_z0 - pz, pz - box_z1)
    d = max(nearest, qx, qy, qz)
    if thickness > 0
        dplate = max(qx, qy, max(-thickness - pz, pz - plate_top))
        d = min(d, dplate)
    end
    return d
end

function _stl_orient_outward(points, faces, sdf_point, eps)
    isempty(faces) && return faces
    score = 0.0
    step = max(1, length(faces) ÷ 32)
    @inbounds for i in 1:step:length(faces)
        f = faces[i]
        p1 = points[f[1]]
        p2 = points[f[2]]
        p3 = points[f[3]]
        ax = p2[1] - p1[1]
        ay = p2[2] - p1[2]
        az = p2[3] - p1[3]
        bx = p3[1] - p1[1]
        by = p3[2] - p1[2]
        bz = p3[3] - p1[3]
        nx = ay * bz - az * by
        ny = az * bx - ax * bz
        nz = ax * by - ay * bx
        mag = sqrt(nx * nx + ny * ny + nz * nz)
        mag == 0 && continue
        mx = (p1[1] + p2[1] + p3[1]) / 3
        my = (p1[2] + p2[2] + p3[2]) / 3
        mz = (p1[3] + p2[3] + p3[3]) / 3
        ux, uy, uz = nx / mag, ny / mag, nz / mag
        outward = sdf_point(mx + eps * ux, my + eps * uy, mz + eps * uz) -
            sdf_point(mx - eps * ux, my - eps * uy, mz - eps * uz)
        score += outward
    end
    score >= 0 && return faces
    return [(f[1], f[3], f[2]) for f in faces]
end

function _stl_write_binary(path, points, faces)
    ntri = length(faces)
    ntri <= typemax(UInt32) ||
        throw(ArgumentError("too many triangles for binary STL"))

    directory = dirname(path)
    isempty(directory) || mkpath(directory)

    header = zeros(UInt8, 80)
    text = codeunits("DiscreteElement binary STL")
    copyto!(header, 1, text, 1, min(length(text), length(header)))

    chunk_triangles = 8192
    buffer = Vector{UInt8}(undef, 50 * chunk_triangles)

    open(path, "w") do io
        write(io, header)
        write(io, UInt32(ntri))

        first = 1
        while first <= ntri
            last = min(first + chunk_triangles - 1, ntri)
            count = last - first + 1

            @inbounds for t in 1:count
                f = faces[first + t - 1]
                p1 = points[f[1]]
                p2 = points[f[2]]
                p3 = points[f[3]]

                ux = p2[1] - p1[1]
                uy = p2[2] - p1[2]
                uz = p2[3] - p1[3]
                vx = p3[1] - p1[1]
                vy = p3[2] - p1[2]
                vz = p3[3] - p1[3]

                nx = uy * vz - uz * vy
                ny = uz * vx - ux * vz
                nz = ux * vy - uy * vx
                inv = sqrt(nx * nx + ny * ny + nz * nz)
                if inv > 0
                    inv = 1 / inv
                    nx *= inv
                    ny *= inv
                    nz *= inv
                else
                    nx = 0.0
                    ny = 0.0
                    nz = 0.0
                end

                off = (t - 1) * 50
                unsafe_store!(Ptr{Float32}(pointer(buffer, off + 1)), Float32(nx))
                unsafe_store!(Ptr{Float32}(pointer(buffer, off + 5)), Float32(ny))
                unsafe_store!(Ptr{Float32}(pointer(buffer, off + 9)), Float32(nz))
                unsafe_store!(Ptr{Float32}(pointer(buffer, off + 13)), Float32(p1[1]))
                unsafe_store!(Ptr{Float32}(pointer(buffer, off + 17)), Float32(p1[2]))
                unsafe_store!(Ptr{Float32}(pointer(buffer, off + 21)), Float32(p1[3]))
                unsafe_store!(Ptr{Float32}(pointer(buffer, off + 25)), Float32(p2[1]))
                unsafe_store!(Ptr{Float32}(pointer(buffer, off + 29)), Float32(p2[2]))
                unsafe_store!(Ptr{Float32}(pointer(buffer, off + 33)), Float32(p2[3]))
                unsafe_store!(Ptr{Float32}(pointer(buffer, off + 37)), Float32(p3[1]))
                unsafe_store!(Ptr{Float32}(pointer(buffer, off + 41)), Float32(p3[2]))
                unsafe_store!(Ptr{Float32}(pointer(buffer, off + 45)), Float32(p3[3]))
                unsafe_store!(Ptr{UInt16}(pointer(buffer, off + 49)), UInt16(0))
            end

            write(io, @view buffer[1:(50 * count)])
            first = last + 1
        end
    end
    return ntri
end

function stl(
    particles::ParticleArrays,
    domain::AbstractVector{<:Real};
    name::AbstractString,
    cell_size,
    substrate_thickness = nothing,
)
    length(domain) == 3 ||
        throw(ArgumentError("domain must have three components"))
    n = length(particles)
    n > 0 || throw(ArgumentError("particles must not be empty"))

    x = Array(particles.x)
    y = Array(particles.y)
    z = Array(particles.z)
    radius = Array(particles.radius)

    Lx, Ly, Lz = Float64.(domain)
    (Lx > 0 && Ly > 0 && Lz > 0) ||
        throw(ArgumentError("domain sides must be positive"))

    cell = Float64(as_si(cell_size, u"m"))
    cell > 0 || throw(ArgumentError("cell_size must be positive"))

    thickness = substrate_thickness === nothing ? 0.0 :
        Float64(as_si(substrate_thickness, u"m"))
    thickness >= 0 ||
        throw(ArgumentError("substrate_thickness must be non-negative"))

    plate_top = thickness > 0 ? cell : 0.0
    rmax = maximum(radius)
    margin = max(rmax * 0.05, cell)
    box_z0 = minimum(z .- radius) - margin
    box_z1 = maximum(z .+ radius) + margin

    pad = 2 * cell
    z_lo = thickness > 0 ? min(box_z0, -thickness) : box_z0
    xs = _stl_axis_positions(-pad, Lx + pad, cell)
    ys = _stl_axis_positions(-pad, Ly + pad, cell)
    zs = _stl_axis_positions(z_lo - pad, box_z1 + pad, cell)
    grid_cell = xs[2] - xs[1]

    sdf = Array{Float64}(undef, length(xs), length(ys), length(zs))
    bins = _stl_build_bins(x, y, z, radius, Lx, Ly, cell)
    _stl_sdf_grid!(
        sdf, xs, ys, zs,
        x, y, z, radius,
        Lx, Ly, box_z0, box_z1,
        thickness, plate_top, bins,
    )

    points, faces = Meshing.isosurface(
        sdf, Meshing.MarchingCubes(iso = 0.0), xs, ys, zs,
    )
    isempty(faces) &&
        error("Marching Cubes produced an empty mesh; check the particle positions")

    eps = cell / 10
    sdf_point = (px, py, pz) -> _stl_sdf_point(
        px, py, pz, x, y, z, radius,
        Lx, Ly, box_z0, box_z1, thickness, plate_top,
    )
    faces = _stl_orient_outward(points, faces, sdf_point, eps)
    ntri = _stl_write_binary(name, points, faces)

    return (;
        name = String(name),
        triangles = ntri,
        cell_size = grid_cell * u"m",
        substrate_thickness = thickness * u"m",
        size = size(sdf),
    )
end