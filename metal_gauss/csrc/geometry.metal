#include <metal_stdlib>
using namespace metal;

// normals_from_depth: forward, and its GATHER-form adjoint.
//
// Derivation, masking rules and measured mutant kill-power live in
// research/normals-from-depth-adjoint.md (verified 31/31 against torch.autograd, f64 rel
// 1e-16..2e-15, f32 1e-7..6e-7).
//
// THE ADJOINT IS A GATHER, NOT A SCATTER. Each INPUT pixel is read by up to three output
// pixels, and in every one of those roles the SAME ray r(v,u) -- the input pixel's own --
// multiplies the result. So it is one thread per input pixel with NO ATOMICS, which also
// makes this lane deterministic, unlike the rasteriser.

// Ray of pixel (v,u). INTEGER indices, matching the forward and the cross-language
// fixture. `plane_depth_from_features` uses pixel CENTRES instead; both match Brush and
// both are deliberate. Measured, the two conventions differ by 7.7e-2 at the grazing
// fixture's intrinsics and only 1.3e-4 at fx=1000.
static inline float3 nfd_ray(uint v, uint u, float4 K) {
    return float3((float(u) - K.z) / K.x, (float(v) - K.w) / K.y, 1.0f);
}

// du, dv, cross, length and validity for OUTPUT pixel (v,u). Returns false when the pixel
// emits no normal, in which case all three of its cotangent roles vanish.
static inline bool nfd_geom(device const float* depth, uint h, uint w, float4 K,
                            uint v, uint u,
                            thread float3& du, thread float3& dv,
                            thread float3& c, thread float& L) {
    const float D0 = depth[v * w + u];
    const float D1 = depth[v * w + u + 1];
    const float D2 = depth[(v + 1) * w + u];
    const float3 P0 = D0 * nfd_ray(v, u, K);
    const float3 P1 = D1 * nfd_ray(v, u + 1, K);
    const float3 P2 = D2 * nfd_ray(v + 1, u, K);
    du = P1 - P0;
    dv = P2 - P0;
    c  = cross(dv, du);                       // dv x du, the order the forward uses
    // Same expression as torch: sqrt(max(s, 1e-24)) then L > 1e-12. A clamped pixel has
    // L == 1e-12 exactly (bit-for-bit in f32 too), so it fails the test and the clamped
    // Jacobian branch is unreachable -- (I - n n^T)/L is valid on every live pixel.
    L = sqrt(max(dot(c, c), 1e-24f));
    return (D0 > 0.0f) && (D1 > 0.0f) && (D2 > 0.0f) && (L > 1e-12f);
}

kernel void nfd_forward(device const float* depth [[buffer(0)]],
                        device float*       out   [[buffer(1)]],
                        constant uint2&     dim   [[buffer(2)]],   // H, W
                        constant float4&    K     [[buffer(3)]],   // fx, fy, cx, cy
                        uint2 gid [[thread_position_in_grid]])
{
    const uint h = dim.x, w = dim.y;
    const uint v = gid.y, u = gid.x;
    if (v + 1 >= h || u + 1 >= w) return;      // last row/column stay zero
    float3 du, dv, c; float L;
    if (!nfd_geom(depth, h, w, K, v, u, du, dv, c, L)) return;   // out is pre-zeroed
    const float3 n = c / L;
    const uint o = (v * w + u) * 3u;
    out[o + 0] = n.x; out[o + 1] = n.y; out[o + 2] = n.z;
}

// dL/dc for OUTPUT pixel (v,u): (I - n n^T) g / L, or exactly 0 when invalid.
// SELECT, never multiply by a mask: multiplying by a 0/1 mask with an unclamped L
// produces NaN at an exact-zero cross product. Measured both ways.
static inline bool nfd_gc(device const float* depth, device const float* g,
                          uint h, uint w, float4 K, uint v, uint u,
                          thread float3& du, thread float3& dv, thread float3& gc) {
    float3 c; float L;
    if (!nfd_geom(depth, h, w, K, v, u, du, dv, c, L)) return false;
    const float3 n = c / L;
    const uint o = (v * w + u) * 3u;
    const float3 gout = float3(g[o + 0], g[o + 1], g[o + 2]);
    gc = (gout - n * dot(n, gout)) / L;
    return true;
}

kernel void nfd_backward(device const float* depth [[buffer(0)]],
                         device const float* g     [[buffer(1)]],   // (H,W,3) cotangent
                         device float*       grad  [[buffer(2)]],   // (H,W)
                         constant uint2&     dim   [[buffer(3)]],
                         constant float4&    K     [[buffer(4)]],
                         uint2 gid [[thread_position_in_grid]])
{
    const uint h = dim.x, w = dim.y;
    const uint v = gid.y, u = gid.x;
    if (v >= h || u >= w) return;

    float3 acc = float3(0.0f);
    float3 du, dv, gc;

    // Role A (base): output (v,u) reads this pixel as D0.
    // gP0 = (dv - du) x gc. THE SIGN HERE IS THE SINGLE LARGEST TRAP in the derivation;
    // getting it wrong reads ~2.0 relative error.
    if (v + 1 < h && u + 1 < w && nfd_gc(depth, g, h, w, K, v, u, du, dv, gc))
        acc += cross(dv - du, gc);

    // Role B (+1 in u): output (v, u-1) reads this pixel as D1.  gP1 = gc x dv
    if (u >= 1 && v + 1 < h && nfd_gc(depth, g, h, w, K, v, u - 1, du, dv, gc))
        acc += cross(gc, dv);

    // Role C (+1 in v): output (v-1, u) reads this pixel as D2.  gP2 = du x gc
    if (v >= 1 && u + 1 < w && nfd_gc(depth, g, h, w, K, v - 1, u, du, dv, gc))
        acc += cross(du, gc);

    // One dot with the INPUT pixel's own ray -- the same r multiplies all three roles.
    grad[v * w + u] = dot(nfd_ray(v, u, K), acc);
}
