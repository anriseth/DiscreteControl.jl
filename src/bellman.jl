using LineSearches
function bellmanobjective(system::DynamicSystem1D,
                          t,x,a,ω, vt::Function)
    numsamples = length(ω)
    retval = 0.
    @simd for n = 1:numsamples
        retval += system.U(t+1,x,a,ω[n]) + vt(system.f(t+1,x,a,ω[n]))
    end
    return -retval / numsamples # Optim minimizes
end

function bellmanobjective(system::DynamicSystem1D,
                          t,x,a,ω, Iv::AbstractInterpolation)
    vt(x) = Iv[x]
    bellmanobjective(system,t,x,a,ω,vt)
end


function solvebellman!(v, α, system::DynamicSystem1D, xtup, ω)
    K = length(xtup[1])
    T = system.T
    @simd for i = 1:K
        v[i,T+1] = system.Ubar(xtup[1][i])
    end
    ti = T
    t = ti-1
    v[1,ti] = system.Ubar(xtup[1][1]) # Boundary condition
    @simd for i = K:-1:2
        objective(a) = bellmanobjective(system,t,xtup[1][i],a, ω[:,ti], system.Ubar)
        res = optimize(objective, system.amin, system.amax)
        v[i,ti] = -Optim.minimum(res)
        α[i,ti] = Optim.minimizer(res)
    end

    for ti = T-1:-1:1
        Iv = interpolate(xtup, v[:,ti+1], Gridded(Linear()))
        t = ti-1
        v[1,ti] = system.Ubar(xtup[1][1]) # Boundary condition
        @simd for i = K:-1:2
            objective(a) = bellmanobjective(system,t,xtup[1][i],a, ω[:,ti], Iv)
            res = optimize(objective, system.amin, system.amax)
            v[i,ti] = -Optim.minimum(res)
            α[i,ti] = Optim.minimizer(res)
        end
    end
    #α[1,:] = NaN # No policy at x=0
    α[1,:] = system.amax
end


function bellmanobjective(system::DynamicSystem,
                          t,x,a,ω::Tuple, vt::Function)
    numsamples = length(ω[1])
    retval = 0.
    @simd for n = 1:numsamples
        wn = [ωi[n] for ωi in ω]
        retval += system.U(t+1,x,a,wn) + vt(system.f(t+1,x,a,wn))
    end
    return -retval / numsamples
end

function bellmanobjective(system::DynamicSystem,
                          t,x,a,ω::Tuple,Iv::AbstractInterpolation)
    vt(x) = Iv[x...]
    bellmanobjective(system,t,x,a,ω,vt)
end


function solvebellman!(v, α, system::DynamicSystem, xtup, ω::Tuple)
    # TODO: use CartesianIndex stuff to allow any dimension
    #       see http://julialang.org/blog/2016/02/iteration
    @assert system.statedim == 2     # TODO: generalise
    @assert length(system.amin(0,xtup[1])) == 1 # TODO: generalise

    K = [length(xtup[i]) for i = 1:system.statedim]
    T = system.T
    vti = view(v,:,:,T+1)
    for i = 1:K[1]
        @simd for j=1:K[2]
            xij = [xtup[1][i], xtup[2][j]]
            vti[j,i] = system.Ubar(xij)
        end
    end

    ti = T
    vti = view(v,:,:,ti)
    αti = view(α,:,:,ti)
    ωti = tuple([view(ωi,:,ti) for ωi in ω]...)
    t = ti-1
    @show t
    # Boundary condition
    @simd for j = 1:K[2]
        xij = [xtup[1][1], xtup[2][j]] # TODO: CartesianIndex on xtup somehow?
        vti[j,1] = system.Ubar(xij)
    end
    for i = 2:K[1] # TODO: CartesianIndex on vti
        @simd for j = 1:K[2]
            xij = [xtup[1][i], xtup[2][j]] # TODO: CartesianIndex on xtup somehow?
            objective(a) = bellmanobjective(system,t,xij,a, ωti, system.Ubar)
            res = optimize(objective, system.amin(t,xij), system.amax(t,xij))
            #res = optimize(objective, [0.5], LBFGS(linesearch! = LineSearches.interpbacktrack!),
            #               OptimizationOptions(autodiff=true))
            @inbounds vti[j,i] = -Optim.minimum(res)#[1]
            @inbounds αti[j,i] = Optim.minimizer(res)#[1]
        end
    end

    for ti = T-1:-1:1
        # TODO: How to deal with extrapolations?
        Iv = interpolate(xtup, vti', Gridded(Linear()))
        #Iv = extrapolate(Iv,Interpolations.Throw())
        vti = view(v,:,:,ti)
        αti = view(α,:,:,ti)
        ωti = tuple([view(ωi,:,ti) for ωi in ω]...)
        t = ti-1
        @show t
        # Boundary condition
        @simd for j = 1:K[2]
            xij = [xtup[1][1], xtup[2][j]] # TODO: CartesianIndex on xtup somehow?
            vti[j,1] = system.Ubar(xij)
        end
        for i = 2:K[1] # TODO: CartesianIndex on vti
            @simd for j = 1:K[2]
                xij = [xtup[1][i], xtup[2][j]] # TODO: CartesianIndex on xtup somehow?
                objective(a) = bellmanobjective(system,t,xij,a, ωti, Iv)
                res = optimize(objective, system.amin(t,xij), system.amax(t,xij))
                vti[j,i] = -Optim.minimum(res)
                αti[j,i] = Optim.minimizer(res)
                #@show xij
                #@show vti[j,i], αti[j,i]
            end
        end
    end
    @inbounds α[:,1,:] = NaN # No policy at x[1]=0
end
