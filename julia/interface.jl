
using StaticArrays, BenchmarkTools

const libsequentialknn = "../target/release/libsequentialknn.so"

function sequentialknn_condix(pts::Vector{SVector{D,Float64}}, k) where{D}
  knn = zeros(UInt64, (k, length(pts)))
  @ccall libsequentialknn.sequential_knn(knn::Ptr{UInt64}, pts::Ptr{Float64},
                                         length(pts)::Csize_t, D::Csize_t, 
                                         k::Csize_t)::Cvoid
  map(enumerate(eachcol(knn))) do (j, cj)
    c = Int64.(cj)[1:min(j-1, k)]
    sort!(c)
    c
  end
end

# 2s versus 53s with HNSW.jl.
const pts = rand(SVector{2,Float64}, 1_000_000)
@btime sequentialknn_condix($pts, $(30));

