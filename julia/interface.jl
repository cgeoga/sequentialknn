
using StaticArrays, BenchmarkTools

const libsequentialknn = "../target/release/libsequentialknn.so"

function required_bucket_size(pts::Vector{SVector{D,Float64}}) where{D}
  maximum(1:D) do j
    ptsj  = getindex.(pts, j)
    sort!(ptsj)
    dptsj = diff(ptsj)
    (ix, n_equal) = (1, 1)
    while true
      new_ix  = findnext(!=(0.0), dptsj, ix)
      isnothing(new_ix) && break
      n_equal = max(n_equal, new_ix - ix + 1)
      ix      = new_ix + 1
    end
    nextpow(2, n_equal)
  end
end

function sequentialknn_condix(pts::Vector{SVector{D,Float64}}, k) where{D}
  knn    = zeros(UInt64, (k, length(pts)))
  bucket = max(32, required_bucket_size(pts))
  if bucket > 4096
    throw(error("The required bucket size for this collection of points is prohibitively large. Unfortunately, this accelerated library wrapper does not currently support your use case."))
  end
  @info "Using bucket size $bucket."
  @ccall libsequentialknn.sequential_knn(knn::Ptr{UInt64}, pts::Ptr{Float64},
                                         length(pts)::Csize_t, D::Csize_t, 
                                         bucket::Csize_t, k::Csize_t)::Cvoid
  map(enumerate(eachcol(knn))) do (j, cj)
    c = Int64.(cj)[1:min(j-1, k)]
    sort!(c)
    c
  end
end

print("n=1M random 2D, k=30:  ")
const pts = rand(SVector{2,Float64}, 1_000_000)
@btime sequentialknn_condix($pts, $(30)) # 2s, vs 53s with HNSW.jl.

print("n=2^20 gridded 2D, k=30:  ")
pts1d = range(0.0, 1.0, length=2^10)
const pts = vec(SVector{2,Float64}.(Iterators.product(pts1d, pts1d)))
@btime sequentialknn_condix($pts, $(30)) # 10s. Not great. 

