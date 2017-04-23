function simulatetrajectories(mpc::MPCSystem1D,
                              osc::OfflineSystemControl1D,
                              ωmodel::UnivariateDistribution,
                              x0,
                              numsimulations::Int = 1000,
                              includemean::Bool = true;
                              ωtrue::UnivariateDistribution = ωmodel,
                              verbose::Bool = false,
                              optimizer::Optim.Optimizer = LBFGS(),
                              aguessinit::Vector = fill(0.5*(mpc.system.amin+mpc.system.amax), mpc.system.T))
    @assert mpc.system === osc.system
    system = mpc.system

    osctrajectories = Vector{DynamicSystemTrajectory1D}(numsimulations)
    mpctrajectories = Vector{DynamicSystemTrajectory1D}(numsimulations)
    atmpc = copy(aguessinit)

    for sim = 1:numsimulations
        trajosc = DynamicSystemTrajectory1D(system)
        trajmpc = DynamicSystemTrajectory1D(system)
        initializestate!(trajosc, x0)
        initializestate!(trajmpc, x0)
        aguess = copy(aguessinit)

        for t = 0:system.T-1
            numscenarios = mpc.numscenarios[t+1]
            includemean = includemean || (numscenarios == 1) # Always predict the future using the mean when only one sample
            if t > 0
                aguess = zeros(system.T-t)
                numsteps = length(aguess)

                for step = 1:numsteps
                    idxs = [(step+1)+numsteps*(scenario-1) for scenario=1:numscenarios]
                    aguess[step] = mean(atmpc[idxs])
                end
            end

            atosc = osc.policy(t, trajosc.state[t+1])
            atmpc = onlinedecision(mpc, trajmpc.state[t+1], t, aguess, ωmodel, includemean;
                                   verbose = verbose, optimizer = optimizer)

            # Update guess for next time
            w = rand(ωtrue)
            step!(trajosc, atosc, t, w)
            step!(trajmpc, atmpc[1], t, w)
        end
        osctrajectories[sim] = trajosc
        mpctrajectories[sim] = trajmpc
    end

    return osctrajectories, mpctrajectories
end

function simulatetrajectories{T<:Real}(oscvec::Vector{OfflineSystemControl1D{T}},
                                 ωmodel::UnivariateDistribution,
                                 x0,
                                 numsimulations::Int = 1000;
                                 ωtrue::UnivariateDistribution = ωmodel)
    @assert length(oscvec) == 2 # TODO: generalise to > 1
    systems = [osc.system for osc in oscvec]
    @assert length(unique(systems)) == 1
    system = systems[1]
    osc1 = oscvec[1]; osc2 = oscvec[2]

    osc1trajectories = Vector{DynamicSystemTrajectory1D}(numsimulations)
    osc2trajectories = Vector{DynamicSystemTrajectory1D}(numsimulations)

    for sim = 1:numsimulations
        trajosc1 = DynamicSystemTrajectory1D(system)
        trajosc2 = DynamicSystemTrajectory1D(system)
        initializestate!(trajosc1, x0)
        initializestate!(trajosc2, x0)

        for t = 0:system.T-1
            atosc1 = osc1.policy(t, trajosc1.state[t+1])
            atosc2 = osc2.policy(t, trajosc2.state[t+1])
            # Update guess for next time
            w = rand(ωtrue)
            step!(trajosc1, atosc1, t, w)
            step!(trajosc2, atosc2, t, w)
        end
        osc1trajectories[sim] = trajosc1
        osc2trajectories[sim] = trajosc2
    end

    return osc1trajectories, osc2trajectories
end
