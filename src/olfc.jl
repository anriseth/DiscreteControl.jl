function createscenarios{T1<:Real}(olfc::OLFCSystem1D{T1}, t::Int,
                                   ω::UnivariateDistribution,
                                   includemean::Bool = true)
    # Assumes each disturbance to the system is i.i.d. ω for each time step
    T = olfc.system.T
    scenarios = rand(ω, T-t, olfc.numscenarios[t+1])
    if includemean == true
        scenarios[:,1] = mean(ω)
    end
    return scenarios
end

function OLFCobjective{T1<:Real}(a, x::T1, t::Int, system::DynamicSystem1D,
                                 scenarios::Array{T1,2})
    numsteps, numscenarios = size(scenarios)
    @assert length(a) == numsteps

    retval = 0.
    @simd for i = 1:numscenarios
        @inbounds retval += predictedobjective(a, x, t, system, scenarios[:,i])
    end

    # retval = @parallel (+) for i = 1:numscenarios
    #     predictedobjective(a, x, t, system, scenarios[:,i])
    # end

    return retval/numscenarios
end

function onlinedecision{T1<:Real}(olfc::OLFCSystem1D{T1}, x::T1,
                                  t::Int,
                                  aguess::Vector{T1},
                                  ω::UnivariateDistribution,
                                  includemean::Bool = true;
                                  verbose::Bool = false,
                                  optimizer::Optim.Optimizer = LBFGS(),
                                  maxiter::Int = 1000,
                                  autodiff::Symbol = :forward)
    w = createscenarios(olfc,w,ω,includemean)
    return onlinedecision(olfc,x,t,aguess,w;
                          verbose=verbose,optimizer=optimizer,
                          maxiter=maxiter,autodiff=autodiff)

end

function onlinedecision{T1<:Real}(olfc::OLFCSystem1D{T1}, x::T1,
                                  t::Int,
                                  aguess::Vector{T1},
                                  w::Array{T1};
                                  verbose::Bool = false,
                                  optimizer = LBFGS,
                                  linesearch = LineSearches.bt3!,
                                  maxiter::Int = 1000,
                                  autodiff::Symbol = :forward)
    system = olfc.system
    objective(a) = OLFCobjective(a, x, t, system, w)

    @assert length(aguess) == system.T-t
    # aguess contains the initial guess for
    # decisions a_t,\dots,a_{T-1}
    numsteps, numscenarios = size(w)

    if optimizer <: Optim.SecondOrderSolver
        df = TwiceDifferentiable(objective, aguess; autodiff=autodiff)
    elseif optimizer <: Optim.FirstOrderSolver
        df = OnceDifferentiable(objective, aguess; autodiff=autodiff)
    else
        df = NonDifferentiable(objective,aguess)
    end

    # res = optimize(df, aguess,
    #                optimizer,
    #                Optim.Options(show_trace=verbose, extended_trace=verbose,
    #                              iterations = maxiter))

    res = optimize(df, aguess,
                   fill(system.amin, length(aguess)), fill(system.amax, length(aguess)),
                   Fminbox{optimizer}(),
                   linesearch = linesearch,
                   show_trace=verbose, extended_trace=verbose,
                   iterations = maxiter,
                   optimizer_o = Optim.Options(
                       show_trace=verbose, extended_trace=verbose,
                       iterations = maxiter))

    # TODO: return Optim.minimum(res) somehow as well? (for solveolfc!)
    return Optim.minimizer(res)
end


function solveolfc!(α, olfc::OLFCSystem1D, xtup, w;
                    verbose::Bool = false,
                    optimizer = LBFGS,
                    linesearch = LineSearches.bt3!,
                    maxiter::Int = 1000,
                    autodiff::Symbol = :forward)
    # This finds the policy that OLFC would have created
    # for each x\in xtup, t \in 0,..,system.T
    # and saves the corresponding decision a(t,x) in α
    #
    # Mainly useful to save time in "simulatetrajectories" et. al.
    #
    # TODO: also create value function v? (Update onlinedecision)
    #
    system = olfc.system
    K = length(xtup[1])
    T = system.T

    # α provides the initial guess
    aguessarr = repmat(α[:,1],1,T)
    for ti = 1:T
        t = ti-1
        @simd for i = K:-1:2
            if i < K
                aguess = aguessarr[i+1,ti:end]
            else
                aguess = aguessarr[i,ti:end]
            end
            aguessarr[i,ti:end] = onlinedecision(olfc, xtup[1][i], t, aguess, w[ti:end,:];
                                                 verbose = verbose, optimizer = optimizer,
                                                 autodiff = autodiff)
            α[i,ti] = aguessarr[i,ti]
        end
    end
    #α[1,:] = NaN # No policy at x=0
    α[1,:] = system.amax
end
