struct CellGrid{I<:AbstractVector{Int32}}
    counts::I
    ids::I
    overflow::I
    nx::Int
    ny::Int
    nz::Int
    max_per_cell::Int
    ncells::Int
end

@inline wrap_position(x, L) = x - L * floor(x / L)
@inline minimum_image(dx, L) = dx - L * round(dx / L)

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

@inline wrap_cell(i::Int, n::Int) = mod1(i, n)

function make_cell_grid(
    backend,
    domain::SVector{3,T},
    rmax::T,
    n::Int;
    max_per_cell::Union{Nothing,Int} = nothing,
) where {T<:AbstractFloat}
    cutoff = T(2) * rmax
    nx = max(1, floor(Int, domain[1] / cutoff))
    ny = max(1, floor(Int, domain[2] / cutoff))
    nz = max(1, floor(Int, domain[3] / cutoff))
    ncells = nx * ny * nz

    mpc = max_per_cell === nothing ?
        max(32, ceil(Int, 8 * n / ncells)) : max_per_cell
    mpc >= 1 || throw(ArgumentError("max_per_cell must be positive"))

    return CellGrid(
        zeros_on(backend, Int32, ncells),
        zeros_on(backend, Int32, mpc * ncells),
        zeros_on(backend, Int32, 1),
        nx, ny, nz, mpc, ncells,
    )
end