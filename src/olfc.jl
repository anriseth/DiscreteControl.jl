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
    system = olfc.system
    w = createscenarios(olfc, t, ω, includemean)
    objective(a) = OLFCobjective(a, x, t, system, w)

    @assert length(aguess) == system.T-t
    # aguess contains the initial guess for
    # decisions a_t,\dots,a_{T-1}
    numsteps, numscenarios = size(w)

    if typeof(optimizer) <: Optim.SecondOrderSolver
        # TODO: allow reversediff?
        df = TwiceDifferentiable(objective, aguess; autodiff=autodiff)
    elseif typeof(optimizer) <: Optim.FirstOrderSolver
        df = OnceDifferentiable(objective, aguess; autodiff=autodiff)
    else
        df = NonDifferentiable(objective,aguess)
    end

    res = optimize(df, aguess,
                   optimizer,
                   Optim.Options(show_trace=verbose, extended_trace=verbose,
                                 iterations = maxiter))

    # TODO: we need to do Fminbox on this one
    # res = optimize(df, aguess, fill(system.amin, length(aguess)), fill(system.amax, length(aguess)),
    #                Fminbox{optimizer}(),
    #                show_trace=verbose, extended_trace=verbose,
    #                iterations = maxiter,
    #                optimizer_o = Optim.Options(
    #                    show_trace=verbose, extended_trace=verbose,
    #                    iterations = maxiter))


    return Optim.minimizer(res)
end
