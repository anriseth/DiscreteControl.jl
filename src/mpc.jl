function createscenarios{T1<:Real}(mpc::MPCSystem1D{T1}, t::Int,
                                   ω::UnivariateDistribution,
                                   includemean::Bool = true)
    # Assumes each disturbance to the system is i.i.d. ω for each time step
    T = mpc.system.T
    scenarios = rand(ω, T-t, mpc.numscenarios[t+1])
    if includemean == true
        scenarios[:,1] = mean(ω)
    end
    return scenarios
end

function predictedobjective{T1<:Real}(a, x::T1, t::Int, system::DynamicSystem1D,
                                      scenario::Vector{T1})
    numsteps = length(scenario)
    @assert t+numsteps == system.T
    @assert length(a) == numsteps
    retval = 0.
    Xs = copy(x)
    for step = 1:numsteps
        @inbounds retval += system.U(t+step, Xs, a[step], scenario[step])
        @inbounds Xs = system.f(t+step, Xs, a[step], scenario[step])
    end

    retval += system.Ubar(Xs)

    return -retval # Minimizing in Optim
end

function MCobjective{T1<:Real}(a, x::T1, t::Int, system::DynamicSystem1D,
                               scenarios::Array{T1,2})
    numsteps, numscenarios = size(scenarios)
    @assert length(a) == (numsteps-1)*numscenarios+1
    retval = 0.

    for i = 1:numscenarios
        j = (numsteps-1)*(i-1)
        idxs = [1; collect(j+2:j+numsteps)]
        @inbounds retval += predictedobjective(a[idxs], x, t, system, scenarios[:,i])
    end

    return retval/numscenarios
end

#==
# On smaller tests, this seems to be slower than doing ForwardDiff directly on MCobjective?
#
function gradpredictedobjective!{T1<:Real}(grad, x::T1, t::Int,
system::DynamicSystem1D,
scenario::Vector{T1})
objective(a) = predictedobjective(a, x, t, system, scenario)
ForwardDiff.gradient!(grad, objective, a)
end

function gradMCobjective!{T1<:Real}(grad, a, x::T1, t::Int, system::DynamicSystem1D,
scenarios::Array{T1,2})
numscenarios = size(scenarios,2)
tmpgrad = zeros(grad)
for i = 1:numscenarios
grad[:] += gradpredictedobjective!(tmpgrad, a, x, t, system, scenarios[:,i])
end
scale!(grad, 1/N)
end
==#

function onlinedecision{T1<:Real}(mpc::MPCSystem1D{T1}, x::T1,
                                  t::Int,
                                  aguess::Vector{T1},
                                  ω::UnivariateDistribution,
                                  includemean::Bool = true)
    system = mpc.system
    w = createscenarios(mpc, t, ω, includemean)
    objective(a) = MCobjective(a, x, t, system, w)
    #grad!(a, out) = gradMCobjective!(out, a, x, t, system, w)
    #grad!(a, out) = ForwardDiff.gradient!(out, objective, a)


    # aguess contains the initial guess for
    # decisions a_t,\dots,a_{T-1}
    # ainit should take into account different decisions for different scenarios
    # (i.e. a[2:numsteps] should also depend on scenario)
    # so we should have a = [at, a[t+1:T-1,1], a[t+1:T-1,2],..., a[t+1:T-1,numscenarios]]
    numsteps, numscenarios = size(w)
    ainit = zeros(T1, (numsteps-1)*numscenarios+1)
    ainit[1] = aguess[1]
    for scenario = 1:numscenarios
        j = (numsteps-1)*(scenario-1)
        ainit[j+2:j+numsteps] = aguess[2:end]
    end

    res = optimize(objective, ainit, BFGS(),
                   OptimizationOptions(autodiff=true,
                   show_trace=true))#, extended_trace=true))

    return Optim.minimizer(res)
end
