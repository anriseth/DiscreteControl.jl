function OLFCobjective{T1<:Real}(a, x::T1, t::Int, system::DynamicSystem1D,
                                 scenarios::Array{T1,2})
    numsteps, numscenarios = size(scenarios)
    @assert length(a) == numsteps
    retval = 0.

    for i = 1:numscenarios
        @inbounds retval += predictedobjective(a, x, t, system, scenarios[:,i])
    end

    return retval/numscenarios
end

function olfcdecision{T1<:Real}(mpc::MPCSystem1D{T1}, x::T1,
                                t::Int,
                                aguess::Vector{T1},
                                ω::UnivariateDistribution,
                                includemean::Bool = true;
                                verbose::Bool = false,
                                optimizer::Optim.Optimizer = LBFGS(),
                                maxiter::Int = 1000)
    system = mpc.system
    w = createscenarios(mpc, t, ω, includemean)
    objective(a) = OLFCobjective(a, x, t, system, w)

    @assert length(a) == system.T-t
    # aguess contains the initial guess for
    # decisions a_t,\dots,a_{T-1}
    numsteps, numscenarios = size(w)

    if typeof(optimizer) <: Optim.SecondOrderSolver
        # TODO: allow reversediff?
        df = TwiceDifferentiable(objective, ainit; autodiff = :forward)
    elseif typeof(optimizer) <: Optim.FirstorderSolver
        df = OnceDifferentiable(objective, ainit; autodiff = :forward)
    else
        df = NonDifferentiable(objective,ainit)
    end

    res = optimize(df,
                   optimizer,
                   Optim.Options(show_trace=verbose, extended_trace=verbose,
                                 iterations = maxiter))

    # TODO: we need to do Fminbox on this one
    # res = optimize(df, ainit, fill(system.amin, length(ainit)), fill(system.amax, length(ainit)),
    #                Fminbox{optimizer}(),
    #                show_trace=verbose, extended_trace=verbose,
    #                iterations = maxiter,
    #                optimizer_o = Optim.Options(
    #                    show_trace=verbose, extended_trace=verbose,
    #                    iterations = maxiter))


    return Optim.minimizer(res)
end
