immutable DynamicSystem
    f::Function # Transition function (function of time, state, control, disturbance)
    U::Function # "Gain" function (function of time, state, control, disturbance)
    Ubar::Function # Terminal value (function of state)
    T::Int # Terminal time
    statedim::Int # Dimension of state
    amin::Function  # Control minimum value (function of time and state)
    amax::Function  # Control maximum value (function of time and state)
    W::Tuple     # Disturbance (draws independent random variables)
end

immutable DynamicSystem1D{T1<:Real}
    f::Function    # Transition function
    U::Function    # "Gain" function
    Ubar::Function # Terminal value
    T::Int         # Terminal time
    amin::T1       # Control minimum value
    amax::T1       # Control maximum value
end

type DynamicSystemTrajectory1D{T1<:Real}
    system::DynamicSystem1D
    state::Vector{T1} # Observed states (t=0,\dots,T)
    control::Vector{T1} # Applied controls (t=0,\dots,T-1)
    value::Vector{T1} # Achieved value (t=0,\dots,T), value[0] = 0
end

function DynamicSystemTrajectory1D{T<:Real}(system::DynamicSystem1D{T})
    DynamicSystemTrajectory1D(system, Vector{T}(), Vector{T}(), Vector{T}())
end

type DynamicSystemTrajectory{T1<:Real}
    system::DynamicSystem
    state::Array{T1,2} # Observed states (t=0,\dots,T)
    control::Array{T1,2} # Applied controls (t=0,\dots,T-1)
    value::Vector{T1} # Achieved value (t=0,\dots,T), value[t=0] = 0
end

function DynamicSystemTrajectory(system::DynamicSystem)
    DynamicSystemTrajectory(system, Array{Float64,2}(system.statedim, system.T+1),
                            Array{Float64,2}(system.statedim, system.T),
                            Vector{Float64}(system.T+1))
end

abstract AbstractSystemController1D

immutable OfflineSystemControl1D{T1<:Real} <: AbstractSystemController1D
    system::DynamicSystem1D{T1}
    initialstate::T1
    policy::Function
end

function OfflineSystemControl1D{T1<:Real}(system::DynamicSystem1D{T1},
                                          initialstate::T1,
                                          xarr::Vector{T1},
                                          αvec::Array{T1,2})
    αinterp = interpolate((xarr,0:system.T-1), αvec, Gridded(Linear()))
    α(t, x) = αinterp[x, t]
    OfflineSystemControl1D(system, initialstate, α)
end

immutable MPCSystem1D{T1<:Real} <: AbstractSystemController1D
    system::DynamicSystem1D{T1}
    initialstate::T1
    numscenarios::Vector{Int} # Number of scenarios for given time step
end

function MPCSystem1D{T1<:Real}(system::DynamicSystem1D{T1}, initialstate::T1, numscenarios::Int)
    MPCSystem1D(system, initialstate, fill(numscenarios,system.T))
end


immutable OfflineSystemControl
    system::DynamicSystem
    policy::Function
end

function OfflineSystemControl{T1<:Real,N<:Int}(system::DynamicSystem,
                                               xtup,
                                               αvec::Array{T1,N})
    αinterp = interpolate(reverse((0:system.T-1, xtup...)),
                          αvec, Gridded(Linear()))
    α(t, x) = αinterp[reverse(x)..., t]
    OfflineSystemControl1D(system, α)
end
