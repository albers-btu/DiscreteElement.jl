import GLMakie
import GeometryBasics as GB

function mp4(
    radii_input::AbstractVector,
    domain::SVector{3,T};
    name::AbstractString,
    frame_every::Int = 500,
    fps::Real = 24,
    size::Tuple{Int,Int} = (800, 600),
    sphere_quality::Int = 6,
    kwargs...,
) where {T<:AbstractFloat}
    frame_every >= 1 ||
        throw(ArgumentError("frame_every must be positive"))
    fps > 0 || throw(ArgumentError("fps must be positive"))
    sphere_quality >= 2 ||
        throw(ArgumentError("sphere_quality must be at least 2"))
    isempty(radii_input) && throw(ArgumentError("radii must not be empty"))

    n = length(radii_input)
    Lx, Ly, Lz = Float64.(domain)
    directory = dirname(name)
    isempty(directory) || mkpath(directory)

    xhost = Vector{Cfloat}(undef, n)
    yhost = Vector{Cfloat}(undef, n)
    zhost = Vector{Cfloat}(undef, n)
    rhost = Vector{Cfloat}(undef, n)
    for i in eachindex(radii_input)
        rhost[i] = Float32(as_si(radii_input[i], u"m"))
    end

    # A unit sphere marker; per-particle `markersize = radius` scales it to the
    # real particle radius, exactly as in the interactive meshscatter view.
    marker = GB.normal_mesh(
        GB.Tessellation(GB.Sphere(GLMakie.Point3f(0), 1.0f0), sphere_quality),
    )

    # Keep boundary spheres whole instead of clipping them at the box faces.
    view_pad = maximum(rhost)

    GLMakie.activate!()
    fig = GLMakie.Figure(; size = size, backgroundcolor = :white)
    ax = GLMakie.Axis3(
        fig[1, 1];
        aspect = :data,
        limits = (
            (-view_pad, Lx + view_pad),
            (-view_pad, Ly + view_pad),
            (-view_pad, Lz + view_pad),
        ),
        xlabel = "x (m)", ylabel = "y (m)", zlabel = "z (m)",
    )

    # Filled by the step-0 callback.
    points = GLMakie.Observable(GLMakie.Point3f[])
    colors = GLMakie.Observable(Cfloat[])

    GLMakie.meshscatter!(
        ax, points;
        marker = marker,
        markersize = rhost,
        color = colors,
        colormap = :viridis,
        colorrange = (0, Lz),
    )

    stream = GLMakie.VideoStream(
        fig; format = "mp4", framerate = fps, visible = false,
    )
    frames = Ref(0)

    function record_frame(particles::ParticleArrays, step::Integer)
        copyto!(xhost, Array(particles.x))
        copyto!(yhost, Array(particles.y))
        copyto!(zhost, Array(particles.z))

        points[] = GLMakie.Point3f.(xhost, yhost, zhost)
        colors[] = copy(zhost)

        GLMakie.recordframe!(stream)
        frames[] += 1
        return nothing
    end

    particles = try
        simulate_n_particles(
            n, radii_input, domain;
            snapshot_every = frame_every,
            snapshot_callback = record_frame,
            kwargs...,
        )
    catch
        try
            close(stream.process)
        catch
        end
        rethrow()
    end

    path = String(name)
    GLMakie.save(path, stream)

    return (;
        particles,
        name = path,
        frames = frames[],
        fps = Float64(fps),
        frame_every,
    )
end