@kernel function kick_kernel!(
    vx, vy, vz, wx, wy, wz,
    @Const(fx), @Const(fy), @Const(fz),
    @Const(tx), @Const(ty), @Const(tz),
    @Const(mass), @Const(inertia),
    dt_half::T,
) where {T}
    i = @index(Global, Linear)
    if i <= length(vx)
        @inbounds begin
            vx[i] += dt_half * fx[i] / mass[i]
            vy[i] += dt_half * fy[i] / mass[i]
            vz[i] += dt_half * fz[i] / mass[i]
            wx[i] += dt_half * tx[i] / inertia[i]
            wy[i] += dt_half * ty[i] / inertia[i]
            wz[i] += dt_half * tz[i] / inertia[i]
        end
    end
end

@kernel function kick_drift_kernel!(
    x, y, z,
    vx, vy, vz,
    wx, wy, wz,
    @Const(fx), @Const(fy), @Const(fz),
    @Const(tx), @Const(ty), @Const(tz),
    @Const(mass), @Const(inertia), @Const(radius),
    dt_half::T, dt::T, Lx::T, Ly::T, Lz::T,
) where {T}
    i = @index(Global, Linear)
    if i <= length(x)
        @inbounds begin
            vx[i] += dt_half * fx[i] / mass[i]
            vy[i] += dt_half * fy[i] / mass[i]
            vz[i] += dt_half * fz[i] / mass[i]
            wx[i] += dt_half * tx[i] / inertia[i]
            wy[i] += dt_half * ty[i] / inertia[i]
            wz[i] += dt_half * tz[i] / inertia[i]

            x[i] = mod(x[i] + dt * vx[i], Lx)
            y[i] = mod(y[i] + dt * vy[i], Ly)
            z[i] += dt * vz[i]

            if z[i] < radius[i]
                z[i] = radius[i]
                vz[i] = max(vz[i], zero(T))
            elseif z[i] > Lz - radius[i]
                z[i] = Lz - radius[i]
                vz[i] = min(vz[i], zero(T))
            end
        end
    end
end

@kernel function box_kernel!(
    x, y, z, vz,
    @Const(radius),
    Lx::T, Ly::T, Lz::T,
) where {T}
    i = @index(Global, Linear)
    if i <= length(x)
        @inbounds begin
            x[i] = mod(x[i], Lx)
            y[i] = mod(y[i], Ly)
            if z[i] < radius[i]
                z[i] = radius[i]
                vz[i] = max(vz[i], zero(T))
            elseif z[i] > Lz - radius[i]
                z[i] = Lz - radius[i]
                vz[i] = min(vz[i], zero(T))
            end
        end
    end
end

function half_kick!(particles::ParticleArrays{T}, dt::T) where {T}
    backend = get_backend(particles.vx)
    kick_kernel!(backend)(
        particles.vx, particles.vy, particles.vz,
        particles.wx, particles.wy, particles.wz,
        particles.fx, particles.fy, particles.fz,
        particles.tx, particles.ty, particles.tz,
        particles.mass, particles.inertia,
        dt * T(0.5);
        ndrange = length(particles),
    )
    return particles
end

function kick_drift!(
    particles::ParticleArrays{T},
    dt::T,
    domain::SVector{3,T},
) where {T}
    backend = get_backend(particles.vx)
    kick_drift_kernel!(backend)(
        particles.x, particles.y, particles.z,
        particles.vx, particles.vy, particles.vz,
        particles.wx, particles.wy, particles.wz,
        particles.fx, particles.fy, particles.fz,
        particles.tx, particles.ty, particles.tz,
        particles.mass, particles.inertia, particles.radius,
        dt * T(0.5), dt, domain[1], domain[2], domain[3];
        ndrange = length(particles),
    )
    return particles
end

function keep_in_box!(
    particles::ParticleArrays{T},
    domain::SVector{3,T},
) where {T}
    backend = get_backend(particles.x)
    box_kernel!(backend)(
        particles.x, particles.y, particles.z, particles.vz,
        particles.radius, domain[1], domain[2], domain[3];
        ndrange = length(particles),
    )
    return particles
end