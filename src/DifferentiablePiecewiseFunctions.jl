module DifferentiablePiecewiseFunctions


struct ConstFunc{T}
    c::T
end
(x::ConstFunc)(_) = x.c

tocallable(x::Base.Callable) = x
tocallable(x) = ConstFunc(x)

struct DifferentiablePiecewise{FL,FR,TX,TY}
    f_left::FL
    f_right::FR
    x_split::TX
    ampl::TY
    dy_left::TY
    dy_right::TY
    b_ampl::Float64
    b_diff::Float64
end
Base.broadcastable(p::DifferentiablePiecewise) = Ref(p)

"""
    DifferentiablePiecewise(f_left, f_right, x_split = 0.0; b1 = 1.0, b2 = b1)

Create a function `f(x)` where `f(x)= x<x_split ? f_left(x) : f_right(x)`, i.e. the function has different definitions
depending on if `x` is larger or smaller than `x_split`. `f_left` and `f_right` can either be callable julia 
functions or numerical values which means we assume a constant function for the respective side. 
In addition, the derivative computed by ForwardDiff is manipulated to guarantee a smooth derivate in the 
vicinity of the break point `x_split`. The width of this transition zone where derivatives are modified is 
determined by the parameters `b1` which determines the width of the smoothing of "jumps" in the function
at `x_split` while `b2` determines the width of smoothing of jumps in derivatives at `x_split`. 

### Examples

To construct a heaviside step function with smooth derivative you can define

````julia
dheaviside = DifferentiablePiecewise(0.0,1.0)
````

A relu function with smooth gradient would be defined as:

````julia
drelu = DifferentiablePiecewise(0.0,identity)
````
"""
function DifferentiablePiecewise(f_left, f_right, x_split = 0.0; b1 = 1.0, b2 = 1.0)
    f_left = tocallable(f_left)
    f_right = tocallable(f_right)
    dy_left = ForwardDiff.derivative(f_left, x_split)
    dy_right = ForwardDiff.derivative(f_right, x_split)
    ampl = f_right(x_split) - f_left(x_split)
    b_ampl = inv(b1)
    b_diff = inv(b2)
    DifferentiablePiecewise(
        f_left,
        f_right,
        x_split,
        ampl,
        dy_left,
        dy_right,
        b_ampl,
        b_diff,
    )
end
(p::DifferentiablePiecewise)(x) = ifelse(x > p.x_split, p.f_right(x), p.f_left(x))

function _deriv(p::DifferentiablePiecewise,vx)
    f_lr, d_otherside = vx > p.x_split ? (p.f_right, p.dy_left) : (p.f_left, p.dy_right)
    #Derivative mixing
    d_atx = ForwardDiff.derivative(f_lr, vx) # Actual derivative
    dweight = 0.5 * exp(-abs(vx - p.x_split) * p.b_diff) # Mixing weights
    d_mixed = d_otherside * dweight + d_atx * (1 - dweight) # Mixed derivative
    p.ampl * p.b_ampl * (1.0 - tanh(p.b_ampl * vx)^2) + d_mixed
end

import ChainRulesCore: ChainRulesCore, Tangent
function ChainRulesCore.frule((_, Δx), p::DifferentiablePiecewise, x)
    return p(x), _deriv(p,x) * Δx
end
function ChainRulesCore.rrule(p::DifferentiablePiecewise, x)
    d = transpose(_deriv(p,x))
    dp_pullback(Δy) = (Tangent{DifferentiablePiecewise}(), d * Δy)
    return p(x), dp_pullback
end

import ForwardDiff
function (p::DifferentiablePiecewise)(x::ForwardDiff.Dual{T}) where {T}
    vx = ForwardDiff.value(x)
    px = ForwardDiff.partials(x)
    d = _deriv(p,vx)
    ForwardDiff.Dual{T}(vx, d * px)
end


export DifferentiablePiecewise

end
