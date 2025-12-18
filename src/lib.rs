use kiddo::{KdTree, NearestNeighbour, SquaredEuclidean};
use std::slice;

#[unsafe(no_mangle)]
pub extern "C" fn sequential_knn(
    buf: *mut u64,   // (k, n) column-major
    pts: *const f64, // flat array: n * d
    n: usize,
    d: usize,
    k: usize,
) {
    match d {
        2 => sequential_knn_static::<2>(buf, pts, n, k),
        3 => sequential_knn_static::<3>(buf, pts, n, k),
        4 => sequential_knn_static::<4>(buf, pts, n, k),
        _ => panic!("This should not be reachable."),
    }
}

fn sequential_knn_static<const D: usize>(buf: *mut u64, pts: *const f64, n: usize, k: usize) {
    // Reinterpret Vector{SVector{D,Float64}} as a flat array.
    let points: &[[f64; D]] = unsafe { slice::from_raw_parts(pts as *const [f64; D], n) };
    // Reinterpret output buffer, a 2D column-major (k,n) array, as a flat
    // array (so buf[(i, j)] = buf[j*k+ i]).
    let out: &mut [u64] = unsafe { slice::from_raw_parts_mut(buf, n * k) };
    // build the tree.
    let mut tree: KdTree<f64, D> = KdTree::with_capacity(n);
    // sequentially update the tree. Note that for the first few indices that
    // are <= k, some entries of buf will be left un-filled. But that will be
    // handled on the Julia side.
    for (j, ptj) in points.iter().enumerate() {
        let nn = tree.nearest_n::<SquaredEuclidean>(ptj, k);
        for (i, NearestNeighbour { item, .. }) in nn.iter().enumerate() {
            out[j * k + i] = *item;
        }
        tree.add(ptj, (j + 1) as u64);
    }
}
