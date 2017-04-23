function initializestate!{T<:Real}(trajectory::DynamicSystemTrajectory1D{T}, state)
    @assert isempty(trajectory.state)
    push!(trajectory.state, state)
    push!(trajectory.value, zero(T))
end

function step!{T<:Real}(trajectory::DynamicSystemTrajectory1D{T}, a::T, t::Int, ω::UnivariateDistribution)
    step!(trajectory, a, t, rand(ω))
end


function step!{T<:Real}(trajectory::DynamicSystemTrajectory1D{T}, a::T, t::Int, w::T)
    # Moves the system from time t to time t+1, and collects data for time t+1
    system = trajectory.system
    @assert 0 <= t < system.T
    x0 = trajectory.state[t+1]
    gainedvalue = system.U(t+1, x0, a, w)
    x1 = system.f(t+1,x0,a,w)
    if t == system.T-1
        gainedvalue += system.Ubar(x1)
    end

    push!(trajectory.control, a)
    push!(trajectory.state, x1)
    push!(trajectory.value, trajectory.value[t+1] + gainedvalue)
end


function initializestate!{T<:Real}(trajectory::DynamicSystemTrajectory{T}, state::Vector{T})
    trajectory.state[:,1] =  state
    trajectory.value[1] = zero(T)
end

function step!{T<:Real}(trajectory::DynamicSystemTrajectory{T}, a::Vector{T},
                        t::Int, W::Tuple)
    step!(trajectory, a, t, rand(ω))
end


function step!{T<:Real}(trajectory::DynamicSystemTrajectory{T}, a,
                        t::Int, w)
    # Moves the system from time t to time t+1, and collects data for time t+1
    system = trajectory.system
    @assert 0 <= t < system.T
    x0 = trajectory.state[:,t+1]
    gainedvalue = system.U(t+1, x0, a, w)
    x1 = system.f(t+1,x0,a,w)
    if t == system.T-1
        gainedvalue += system.Ubar(x1)
    end

    trajectory.control[:,t+1] = a
    trajectory.state[:,t+2] = x1
    trajectory.value[t+2] = trajectory.value[t+1] + gainedvalue
end
