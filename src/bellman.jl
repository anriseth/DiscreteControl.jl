function bellmanobjective(system::DynamicSystem1D,
                          t,x,a,ω, Iv::AbstractInterpolation)
    numsamples = length(ω)
    retval = 0.
    for n = 1:numsamples
        retval += system.U(t+1,x,a,ω[n]) + Iv[system.f(t+1,x,a,ω[n])]
    end
    return -retval / numsamples
end

function solvebellman!(v, α, system::DynamicSystem1D, xtup, ω)
    K = length(xtup[1])
    T = system.T
    for i = 1:K
        v[i,T+1] = system.Ubar(xtup[1][i])
    end

    for ti = T:-1:1
        Iv = interpolate(xtup, v[:,ti+1], Gridded(Linear()))
        t = ti-1
        for i = K:-1:2
            objective(a) = bellmanobjective(system,t,xtup[1][i],a, ω[:,ti], Iv)
            res = optimize(objective, system.amin, system.amax)
            v[i,ti] = -Optim.minimum(res)
            α[i,ti] = Optim.minimizer(res)
        end
    end
    α[1,:] = NaN # No policy at x=0

    return α
end
