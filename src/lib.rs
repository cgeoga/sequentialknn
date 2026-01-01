use kiddo::{NearestNeighbour, SquaredEuclidean};
use kiddo::float::kdtree::KdTree;
use std::slice;

#[unsafe(no_mangle)]
pub extern "C" fn sequential_knn(
    buf: *mut u64,   // (k, n) column-major
    pts: *const f64, // flat array: n * d
    n: usize,
    d: usize,
    b: usize,
    k: usize,
) {
    match (d, b) {
      // 2D:
      (2, 32) => sequential_knn_static::<2,32>(buf, pts, n, k),
      (2, 64) => sequential_knn_static::<2,64>(buf, pts, n, k),
      (2, 128) => sequential_knn_static::<2,128>(buf, pts, n, k),
      (2, 256) => sequential_knn_static::<2,256>(buf, pts, n, k),
      (2, 512) => sequential_knn_static::<2,512>(buf, pts, n, k),
      (2, 1024) => sequential_knn_static::<2,1024>(buf, pts, n, k),
      (2, 2048) => sequential_knn_static::<2,2048>(buf, pts, n, k),
      (2, 4096) => sequential_knn_static::<2,4096>(buf, pts, n, k),
      // 3D:
      (3, 32) => sequential_knn_static::<3,32>(buf, pts, n, k),
      (3, 64) => sequential_knn_static::<3,64>(buf, pts, n, k),
      (3, 128) => sequential_knn_static::<3,128>(buf, pts, n, k),
      (3, 256) => sequential_knn_static::<3,256>(buf, pts, n, k),
      (3, 512) => sequential_knn_static::<3,512>(buf, pts, n, k),
      (3, 1024) => sequential_knn_static::<3,1024>(buf, pts, n, k),
      (3, 2048) => sequential_knn_static::<3,2048>(buf, pts, n, k),
      (3, 4096) => sequential_knn_static::<3,4096>(buf, pts, n, k),
      // 4D:
      (4, 32) => sequential_knn_static::<4,32>(buf, pts, n, k),
      (4, 64) => sequential_knn_static::<4,64>(buf, pts, n, k),
      (4, 128) => sequential_knn_static::<4,128>(buf, pts, n, k),
      (4, 256) => sequential_knn_static::<4,256>(buf, pts, n, k),
      (4, 512) => sequential_knn_static::<4,512>(buf, pts, n, k),
      (4, 1024) => sequential_knn_static::<4,1024>(buf, pts, n, k),
      (4, 2048) => sequential_knn_static::<4,2048>(buf, pts, n, k),
      (4, 4096) => sequential_knn_static::<4,4096>(buf, pts, n, k),
      // else:
      _ => panic!("Your given combination of dimension and required bucket size is not supported. But this panic shouldn't be reachable from the Julia wrapper, so please open an issue."),
    }
}

fn sequential_knn_static<const D: usize, const B: usize>(buf: *mut u64, pts: *const f64, n: usize, k: usize) {
    // Reinterpret Vector{SVector{D,Float64}} as a flat array.
    let points: &[[f64; D]] = unsafe { slice::from_raw_parts(pts as *const [f64; D], n) };
    // Reinterpret output buffer, a 2D column-major (k,n) array, as a flat
    // array (so buf[(i, j)] = buf[j*k+ i]).
    let out: &mut [u64] = unsafe { slice::from_raw_parts_mut(buf, n * k) };
    // build the tree.
    let mut tree: KdTree<f64, u64, D, B, u32> = KdTree::with_capacity(n);
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
