using DifferentiablePiecewiseFunctions
using Test

@testset "DifferentiablePiecewiseFunctions.jl" begin
    using Test
    dheaviside = DifferentiablePiecewise(0.0,1.0)
    @test dheaviside(0.1) == 1
    @test dheaviside(-0.1) == 0
    @test ForwardDiff.derivative(dheaviside,0.0) == 1.0
    @test ForwardDiff.derivative(dheaviside,1.0) == ForwardDiff.derivative(tanh,1.0)
    
    dheavisidewide = DifferentiablePiecewise(0.0,1.0;b1=10.0)
    @test dheavisidewide(0.1) == 1
    @test dheavisidewide(-0.1) == 0
    @test ForwardDiff.derivative(dheavisidewide,0.0) == 0.1
    @test ForwardDiff.derivative(dheavisidewide,1.0) == ForwardDiff.derivative(x->tanh(x*0.1),1.0)
    
    relu1 = DifferentiablePiecewise(zero,x->x-1.0,1.0)
    @test relu1(0.5) == 0.0
    @test relu1(1.5) == 0.5
    @test ForwardDiff.derivative(relu1,1.0) == 0.5
    @test ForwardDiff.derivative(relu1,0.5) == 0.5*exp(-0.5)
    @test ForwardDiff.derivative(relu1,2.0) == (1-0.5*exp(-1.0))
    
end
