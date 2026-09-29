function median_value(values)
    s = sort(values)
    n = length(s)
    n == 0 && throw(ArgumentError("median of an empty collection"))
    if isodd(n)
        return s[(n + 1) ÷ 2]
    else
        return (s[n ÷ 2] + s[n ÷ 2 + 1]) / 2
    end
end

function ensure_parent(path::AbstractString)
    directory = dirname(path)
    if !isempty(directory)
        mkpath(directory)
    end
    return path
end

"""
    packing_summary(particles, domain) -> named tuple

Bed height is the highest point on a sphere. The apparent packing fraction
divides the sphere volume by `Lx * Ly * height`, so the free surface makes it
lower than a bulk measurement. Coordination counts periodic contacts.
"""
function packing_summary(particles::Particles, domain::Domain)
    host = cpu_particles(particles)
    height = maximum(host.z .+ host.radius)
    volume = zero(eltype(host.radius))
    for r in host.radius
        volume += (4 / 3) * π * r^3
    end
    packing = volume / (domain.Lx * domain.Ly * height)
    return (;
        height = height * u"m",
        apparent_packing = packing,
        mean_coordination = mean_coordination(host, domain),
        n = nparticles(host),
    )
end

function mean_coordination(host::Particles, domain::Domain)
    n = nparticles(host)
    n == 0 && return 0.0
    rmax = maximum(host.radius)
    cell = 2 * rmax
    nx = max(1, floor(Int, domain.Lx / cell))
    ny = max(1, floor(Int, domain.Ly / cell))
    z0 = minimum(host.z) - rmax
    z1 = maximum(host.z) + rmax
    nz = max(1, floor(Int, (z1 - z0) / cell))
    buckets = [Int[] for _ in 1:(nx * ny * nz)]
    lin(ix, iy, iz) = ix + (iy - 1) * nx + (iz - 1) * nx * ny
    for i in 1:n
        ix = clamp(floor(Int, host.x[i] / domain.Lx * nx) + 1, 1, nx)
        iy = clamp(floor(Int, host.y[i] / domain.Ly * ny) + 1, 1, ny)
        iz = clamp(floor(Int, (host.z[i] - z0) / (z1 - z0) * nz) + 1, 1, nz)
        push!(buckets[lin(ix, iy, iz)], i)
    end
    contacts = 0
    seen = Vector{Int}(undef, 27)
    for i in 1:n
        ix = clamp(floor(Int, host.x[i] / domain.Lx * nx) + 1, 1, nx)
        iy = clamp(floor(Int, host.y[i] / domain.Ly * ny) + 1, 1, ny)
        iz = clamp(floor(Int, (host.z[i] - z0) / (z1 - z0) * nz) + 1, 1, nz)
        ri = host.radius[i]
        nseen = 0
        for oz in -1:1, oy in -1:1, ox in -1:1
            jz = iz + oz
            1 <= jz <= nz || continue
            jx = mod1(ix + ox, nx)
            jy = mod1(iy + oy, ny)
            c = lin(jx, jy, jz)
            dup = false
            for s in 1:nseen
                if seen[s] == c
                    dup = true
                    break
                end
            end
            dup && continue
            nseen += 1
            seen[nseen] = c
            for j in buckets[c]
                j <= i && continue
                dx = minimum_image(host.x[j] - host.x[i], domain.Lx)
                dy = minimum_image(host.y[j] - host.y[i], domain.Ly)
                dz = host.z[j] - host.z[i]
                reach = ri + host.radius[j]
                if dx * dx + dy * dy + dz * dz <= reach * reach
                    contacts += 1
                end
            end
        end
    end
    return 2 * contacts / n
end

function fill_sdf!(sdf, particles::Particles, grid::CellGrid, params::SimParams, xs, ys, zs, box_z0, box_z1)
    T = eltype(sdf)
    x0 = T(xs[1])
    y0 = T(ys[1])
    z0s = T(zs[1])
    dx = T((xs[end] - xs[1]) / (length(xs) - 1))
    dy = T((ys[end] - ys[1]) / (length(ys) - 1))
    dz = T((zs[end] - zs[1]) / (length(zs) - 1))
    _launch(
        sdf_kernel!, get_backend(sdf), size(sdf),
        sdf,
        particles.x, particles.y, particles.z, particles.radius,
        grid.counts, grid.ids,
        params,
        x0, dx, y0, dy, z0s, dz,
        T(box_z0), T(box_z1),
    )
    return sdf
end

function sample_axes(start, stop, voxel)
    n = max(2, ceil(Int, (stop - start) / voxel) + 1)
    return range(float(start), float(stop); length = n)
end

"""Signed distance of an axis-aligned box. Negative inside, zero on the faces."""
function aabb_sdf(x, y, z, x0, x1, y0, y1, z0, z1)
    return max(max(x0 - x, x - x1), max(y0 - y, y - y1), max(z0 - z, z - z1))
end

"""
Union a substrate slab into a powder signed-distance grid.

The slab is the build plate: it fills the periodic cell from `z0` up to `z1`.
`z1` sits one voxel above the floor so particles resting on `z = 0` join the plate.
"""
function union_substrate!(sdf, xs, ys, zs, Lx, Ly, z0, z1)
    @inbounds for k in eachindex(zs)
        z = zs[k]
        for j in eachindex(ys)
            y = ys[j]
            for i in eachindex(xs)
                plate = aabb_sdf(xs[i], y, z, 0.0, Lx, 0.0, Ly, z0, z1)
                sdf[i, j, k] = min(sdf[i, j, k], plate)
            end
        end
    end
    return sdf
end

function union_sdf(px, py, pz, host::Particles, domain::Domain)
    d = Inf
    @inbounds for i in 1:nparticles(host)
        dx = minimum_image(host.x[i] - px, domain.Lx)
        dy = minimum_image(host.y[i] - py, domain.Ly)
        dz = host.z[i] - pz
        d = min(d, sqrt(dx * dx + dy * dy + dz * dz) - host.radius[i])
    end
    return d
end

function orient_outward(points, faces, sdf, eps)
    isempty(faces) && return faces
    score = 0.0
    step = max(1, length(faces) ÷ 32)
    for i in 1:step:length(faces)
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
        outward = sdf(mx + eps * ux, my + eps * uy, mz + eps * uz) -
            sdf(mx - eps * ux, my - eps * uy, mz - eps * uz)
        score += outward
    end
    score >= 0 && return faces
    return [(f[1], f[3], f[2]) for f in faces]
end

function mesh_from_sdf(sdf::AbstractArray{<:AbstractFloat,3}, xs, ys, zs, sdf_point, eps)
    points, faces = isosurface(sdf, MarchingCubes(iso = eltype(sdf)(0)), xs, ys, zs)
    isempty(faces) && error("Marching Cubes produced an empty mesh; check the particle positions")
    faces = orient_outward(points, faces, sdf_point, eps)
    gb_points = Point3{Float64}[Point3{Float64}(p[1], p[2], p[3]) for p in points]
    gb_faces = TriangleFace{Int}[TriangleFace{Int}(f[1], f[2], f[3]) for f in faces]
    return Mesh(gb_points, gb_faces)
end

"""
    export_powder_bed(bed; stl, csv, voxels_per_diameter=6, substrate_thickness=nothing)

Write the Marching Cubes surface of the sphere union clipped to the periodic
cell, sitting on a solid build plate, and a CSV of `id,x,y,z,radius` in meters.
The grid is about `voxels_per_diameter` cells across the median diameter.

`substrate_thickness` is the plate depth below `z = 0`. `nothing` uses three
median diameters. `0` or `0u"m"` writes the spheres alone. A length such as
`200u"μm"` sets the depth. The returned thickness is a length.
"""
function export_powder_bed(
    bed::PowderBed;
    stl::AbstractString,
    csv::AbstractString,
    voxels_per_diameter = 6,
    substrate_thickness = nothing,
)
    voxels_per_diameter = as_dimensionless(voxels_per_diameter)
    voxels_per_diameter >= 2 ||
        throw(ArgumentError("voxels_per_diameter must be at least 2"))
    particles = bed.particles
    domain = bed.domain
    bin_particles!(bed.grid, particles, bed.params)
    spilled = cell_overflow(bed.grid)
    spilled == 0 || error("cell list overflowed during meshing ($spilled particles)")

    host = cpu_particles(particles)
    d_med = 2 * median_value(host.radius)
    voxel = d_med / float(voxels_per_diameter)
    thickness = substrate_thickness === nothing ? 3 * d_med : as_si(substrate_thickness, u"m")
    thickness >= 0 || throw(ArgumentError("substrate_thickness must be non-negative"))
    # One voxel of the plate extends above z = 0 so resting particles share
    # interior voxels with the plate and the mesh joins them.
    plate_top = thickness > 0 ? voxel : 0.0
    margin = max(maximum(host.radius) * 0.05, voxel)
    box_z0 = minimum(host.z .- host.radius) - margin
    box_z1 = maximum(host.z .+ host.radius) + margin
    pad = 2 * voxel
    z_lo = thickness > 0 ? min(box_z0, -thickness) : box_z0
    xs = sample_axes(-pad, domain.Lx + pad, voxel)
    ys = sample_axes(-pad, domain.Ly + pad, voxel)
    zs = sample_axes(z_lo - pad, box_z1 + pad, voxel)

    backend = get_backend(particles.x)
    T = eltype(particles.x)
    sdf = allocate(backend, T, (length(xs), length(ys), length(zs)))
    fill!(sdf, zero(T))
    fill_sdf!(sdf, particles, bed.grid, bed.params, xs, ys, zs, box_z0, box_z1)
    host_sdf = Array{T,3}(undef, size(sdf))
    copyto!(host_sdf, sdf)
    if thickness > 0
        union_substrate!(host_sdf, xs, ys, zs, domain.Lx, domain.Ly, -thickness, plate_top)
    end

    mesh = mesh_from_sdf(host_sdf, xs, ys, zs, (x, y, z) -> begin
        dbox = aabb_sdf(x, y, z, 0.0, domain.Lx, 0.0, domain.Ly, box_z0, box_z1)
        d = max(union_sdf(x, y, z, host, domain), dbox)
        if thickness > 0
            d = min(d, aabb_sdf(x, y, z, 0.0, domain.Lx, 0.0, domain.Ly, -thickness, plate_top))
        end
        return d
    end, voxel / 10)

    ensure_parent(stl)
    save(File{format"STL_BINARY"}(stl), mesh)
    write_particle_csv(csv, host, domain, thickness)
    return (;
        stl = String(stl),
        csv = String(csv),
        triangles = length(faces(mesh)),
        substrate_thickness = thickness * u"m",
    )
end

function write_particle_csv(path, host::Particles, domain::Domain, substrate_thickness)
    ensure_parent(path)
    open(path, "w") do io
        println(io, "# DiscreteElement powder bed")
        println(io, "# unit_m")
        @printf(io, "# Lx %.17g\n", domain.Lx)
        @printf(io, "# Ly %.17g\n", domain.Ly)
        @printf(io, "# substrate_thickness %.17g\n", substrate_thickness)
        println(io, "id,x,y,z,radius")
        for i in 1:nparticles(host)
            @printf(io, "%d,%.17g,%.17g,%.17g,%.17g\n",
                i, host.x[i], host.y[i], host.z[i], host.radius[i])
        end
    end
    return path
end
