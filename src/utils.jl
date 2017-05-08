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
    numosc = length(oscvec)
    systems = [osc.system for osc in oscvec]
    @assert length(unique(systems)) == 1
    system = systems[1]

    osctrajectories = tuple([Vector{DynamicSystemTrajectory1D}(numsimulations)
                             for k = 1:numosc]...)

    for sim = 1:numsimulations
        trajosc = Vector{DynamicSystemTrajectory1D}(numosc)
        for k = 1:numosc
            trajosc[k] = DynamicSystemTrajectory1D(system)
            initializestate!(trajosc[k], x0)
        end

        for t = 0:system.T-1
            atosc = [oscvec[k].policy(t, trajosc[k].state[t+1]) for k = 1:numosc]
            # Update guess for next time
            w = rand(ωtrue)
            for k = 1:numosc
                step!(trajosc[k], atosc[k], t, w)
            end

        end
        for k = 1:numosc
            osctrajectories[k][sim] = trajosc[k]
        end
    end

    return osctrajectories
end

function simulatetrajectories(olfc::OLFCSystem1D,
                              osc::OfflineSystemControl1D,
                              ωmodel::UnivariateDistribution,
                              x0,
                              numsimulations::Int = 1000,
                              includemean::Bool = true;
                              ωtrue::UnivariateDistribution = ωmodel,
                              verbose::Bool = false,
                              optimizer::Optim.Optimizer = LBFGS(),
                              aguessinit::Vector = fill(0.5*(olfc.system.amin+olfc.system.amax), olfc.system.T))
    @assert olfc.system === osc.system
    system = olfc.system

    osctrajectories = Vector{DynamicSystemTrajectory1D}(numsimulations)
    olfctrajectories = Vector{DynamicSystemTrajectory1D}(numsimulations)
    atolfc = copy(aguessinit)

    for sim = 1:numsimulations
        trajosc = DynamicSystemTrajectory1D(system)
        trajolfc = DynamicSystemTrajectory1D(system)
        initializestate!(trajosc, x0)
        initializestate!(trajolfc, x0)
        aguess = copy(aguessinit)

        for t = 0:system.T-1
            numscenarios = olfc.numscenarios[t+1]
            includemean = includemean || (numscenarios == 1) # Always predict the future using the mean when only one sample
            if t > 0
                aguess = atolfc[2:end]
            end

            atosc = osc.policy(t, trajosc.state[t+1])
            atolfc = onlinedecision(olfc, trajolfc.state[t+1], t, aguess, ωmodel, includemean;
                                    verbose = verbose, optimizer = optimizer)

            # Update guess for next time
            w = rand(ωtrue)
            step!(trajosc, atosc, t, w)
            step!(trajolfc, atolfc[1], t, w)
        end
        osctrajectories[sim] = trajosc
        olfctrajectories[sim] = trajolfc
    end

    return osctrajectories, olfctrajectories
end
