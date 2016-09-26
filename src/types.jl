type DynamicSystem{T1<:Real, Na<:Int}
    f::Function # Transition function
    U::Function # "Gain" function
    Ubar::Function # Terminal value
    T::Int # Terminal time
    statedim::Int # Dimension of state
    amin::Array{T1,Na}  # Control minimum value
    amax::Array{T1,Na}  # Control maximum value
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
