module GAML

import Flux
import Flux: @treelike
import LinearAlgebra: Matrix, I
import Statistics: mean

export Projection, minmaxmean

init_projection(dims...) = Matrix(1.0*I,dims...)/100 .+ randn(dims...)/1000

struct Projection{F,S}
	P::S # Projection matrix
	nl::F # Non-linear transform
end

Projection(P) = Projection(P, identity)

function Projection(in::Integer, out::Integer, transform=identity; initP=init_projection)
	return Projection(Flux.param(init_projection(out,in)), transform)
end

@treelike Projection

function (a::Projection)(x::AbstractArray{T,2}) where T<:Real
	P, nl = a.P, a.nl
	return nl.(P * x)
end

function (a::Projection)(x::AbstractArray)
	P, nl = a.P, a.nl
	return [nl.(P * i) for i in x]
end

function Base.show(io::IO, l::Projection)
	print(io, "Projection(", size(l.P, 2), ", ", size(l.P, 1))
	l.nl == identity || print(io, ", ", l.nl)
	print(io, ")")
end

function minmaxmean(x::AbstractArray{T,2}; dims=2) where T<:Real
	return cat(minimum(x,dims=dims),maximum(x,dims=dims),mean(x,dims=dims),dims=1)
end

function minmaxmean(x::AbstractArray; dims=2)
	return hcat([cat(minimum(i,dims=dims),maximum(i,dims=dims),mean(i,dims=dims),dims=1) for i in x]...)
end

end # module
