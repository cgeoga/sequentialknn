
using StaticArrays

# 2s versus 53s with HNSW.jl for 1M pts and k=30.
function sequentialknn_condix(pts::Vector{SVector{D,Float64}}, k) where{D}
  knn = zeros(UInt64, (k, length(pts)))
  ccall((:sequential_knn, "../target/release/libsequentialknn.so"),
        Cvoid,
        (Ptr{UInt64}, Ptr{Float64}, Csize_t, Csize_t, Csize_t),
        knn, pts, Csize_t(length(pts)), Csize_t(D), Csize_t(k))
  map(enumerate(eachcol(knn))) do (j, cj)
    c = Int64.(cj)[1:min(j-1, k)]
    sort!(c)
    c
  end
end

