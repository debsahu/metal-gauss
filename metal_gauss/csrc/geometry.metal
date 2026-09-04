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
    // isfinite(L) makes `valid` coincide with the downstream |n_d| > 0.5 gate BY
    // CONSTRUCTION. Measured, the two agree on 775,946,240 real pixels with zero
    // mismatches; the exact relation is `gate == valid AND isfinite(L)`, and they separate
    // only when |c|^2 overflows f32 (depth ~1e15 m here). Today that is harmless -- c/inf
    // gives a zero normal, which the gate rejects anyway -- but a FUSED loss kernel that
    // masks on `valid` instead of on |n| would score a spurious loss of 1.0 there. One
    // comparison closes the question instead of relying on that argument.
    return (D0 > 0.0f) && (D1 > 0.0f) && (D2 > 0.0f) && (L > 1e-12f) && isfinite(L);
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

// ---------------------------------------------------------------- fused geometry losses
//
// One pass over the image for the alpha divide, the normal normalise, and all three loss
// numerators + counts. Replaces ~30 torch elementwise dispatches, each of which read and
// wrote 2.7M x 3 floats at 1920x1440.
//
// Derivation and every rule below: research/depth-normal-loss-adjoint.md (44/44 against
// torch.autograd; composed chain f32 3.4e-8..1.13e-6, cosine 1.0000000000).
//
// N IS COUNTED IN uint. An f32 sum of ones is exact only to 2^24, which is EXACTLY one
// 4096^2 face -- and our cube faces are that size. The value sum is a two-level
// fixed-order reduction (per-threadgroup partials, then a fixed-order final pass): a
// sequential f32 accumulate over 2.7M pixels errs 6.0e-4 on the loss VALUE while every
// gradient check still passes.
//
// SUBSTITUTE, NEVER MULTIPLY BY A MASK. `err = 1 - dot(n_d, n_r)` computed first and
// multiplied by m afterwards turns a non-finite n_d into a NaN sum. Branch instead.

#define GL_TG 256

struct GLPartial { float d_num; float n_num; float dn_num; uint d_cnt; uint n_cnt; uint dn_cnt; };

kernel void geom_loss_forward(
    device const float* z_img    [[buffer(0)]],   // (H,W,3), aux depth; channel 0 used
    device const float* n_sum    [[buffer(1)]],   // (H,W,3), aux normals, alpha-weighted
    device const float* alpha    [[buffer(2)]],   // (H,W), DETACHED upstream
    device const float* n_d      [[buffer(3)]],   // (H,W,3) from nfd_forward
    device const float* gt_depth [[buffer(4)]],   // (H,W), 0 = invalid
    device const float* gt_norm  [[buffer(5)]],   // (H,W,3), (0,0,0) = invalid
    device float*       out_num  [[buffer(6)]],   // 3 per threadgroup
    device uint*        out_cnt  [[buffer(7)]],   // 3 per threadgroup
    device float*       depth_o  [[buffer(8)]],   // (H,W)   depth_img, saved for backward
    device float*       nr_o     [[buffer(9)]],   // (H,W,3) n_r,       saved for backward
    constant uint2&     dim      [[buffer(10)]],
    constant uint&      space    [[buffer(11)]],  // 0 = disparity, 1 = metric
    // Mask support. The torch path multiplies the PRIORS and the dn alpha by `keep`, so a
    // dropped pixel's gt_depth becomes exactly 0 (invalid), its gt_normal becomes the zero
    // vector (invalid), and its dn alpha falls below the 0.5 gate. Multiplying here
    // reproduces that exactly rather than approximating it with a separate branch.
    device const float*  keep    [[buffer(12)]],   // (H,W), 1 = keep, or empty
    constant uint&      has_keep [[buffer(13)]],
    // PGSR plane-aux (Task 19). 0 = CENTRE depth: buffer 0 is (H,W,3) and its channel 0 is
    // an alpha-WEIGHTED z that must be divided by alpha to recover the depth. 1 = GIVEN:
    // buffer 0 is a finished (H,W) depth map -- PGSR's ray-plane intersection, where alpha
    // cancels exactly between the ray-plane numerator and denominator and dividing again
    // would be wrong by 1/alpha (see plane_depth_from_features).
    //
    // ONE uniform decides BOTH the stride and the division because they are the same fact:
    // mode 0's buffer is 3-channel and needs the divide, mode 1's is 1-channel and does
    // not. Splitting them into two uniforms would admit a combination that means nothing.
    //
    // The BACKWARD kernel needs no equivalent. It emits dL/d(depth_img), which is the same
    // quantity in both modes; the 1/alpha that turns it into dL/d(z_img) is applied by
    // _FusedGeometryLosses.backward in Python, because section 11.4's deferred
    // optimisation left the gather's final multiply outside the kernel.
    constant uint&   depth_mode  [[buffer(14)]],
    uint  gidx [[thread_position_in_grid]],
    uint  lidx [[thread_index_in_threadgroup]],
    uint  tgid [[threadgroup_position_in_grid]])
{
    threadgroup float s_d[GL_TG], s_n[GL_TG], s_dn[GL_TG];
    threadgroup uint  c_d[GL_TG], c_n[GL_TG], c_dn[GL_TG];

    const uint H = dim.x, W = dim.y, NPIX = H * W;
    float d_num = 0.0f, n_num = 0.0f, dn_num = 0.0f;
    uint  d_cnt = 0u,   n_cnt = 0u,   dn_cnt = 0u;

    if (gidx < NPIX) {
        const uint o3 = gidx * 3u;
        const float k  = has_keep ? keep[gidx] : 1.0f;
        const float a  = max(alpha[gidx], 1e-10f);
        const uint  dstride = (depth_mode == 0u) ? 3u : 1u;
        const float dz = z_img[gidx * dstride];
        const float di = (depth_mode == 0u) ? (dz / a) : dz;
        depth_o[gidx] = di;

        float3 nr = float3(n_sum[o3], n_sum[o3+1], n_sum[o3+2]) / a;
        nr = nr / max(length(nr), 1e-6f);
        nr_o[o3] = nr.x; nr_o[o3+1] = nr.y; nr_o[o3+2] = nr.z;

        // ---- depth loss: valid = gt > 0; uncovered pred scores the FULL disparity ----
        const float gd = gt_depth[gidx] * k;
        if (gd > 0.0f) {
            float r;
            if (space == 0u) {                       // disparity
                const float dp = (di > 0.0f) ? (1.0f / di) : 0.0f;
                r = dp - 1.0f / gd;
            } else {
                r = di - gd;
            }
            d_num = fabs(r); d_cnt = 1u;
        }

        // ---- normal loss: component L1 over pixels whose PRIOR is valid ----
        const float3 gn = float3(gt_norm[o3], gt_norm[o3+1], gt_norm[o3+2]) * k;
        if (length(gn) > 0.5f) {
            const float3 e = abs(nr - gn);
            n_num = e.x + e.y + e.z; n_cnt = 1u;
        }

        // ---- depth-normal consistency: gate on alpha AND both magnitudes ----
        const float3 nd = float3(n_d[o3], n_d[o3+1], n_d[o3+2]);
        if (alpha[gidx] * k > 0.5f && length(nd) > 0.5f && length(nr) > 0.5f) {
            dn_num = 1.0f - dot(nd, nr); dn_cnt = 1u;   // branch, never `* m`
        }
    }

    s_d[lidx] = d_num; s_n[lidx] = n_num; s_dn[lidx] = dn_num;
    c_d[lidx] = d_cnt; c_n[lidx] = n_cnt; c_dn[lidx] = dn_cnt;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Fixed-order tree reduction: deterministic, and pairwise rather than sequential.
    for (uint s = GL_TG / 2u; s > 0u; s >>= 1u) {
        if (lidx < s) {
            s_d[lidx] += s_d[lidx+s]; s_n[lidx] += s_n[lidx+s]; s_dn[lidx] += s_dn[lidx+s];
            c_d[lidx] += c_d[lidx+s]; c_n[lidx] += c_n[lidx+s]; c_dn[lidx] += c_dn[lidx+s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lidx == 0u) {
        out_num[tgid*3u+0u] = s_d[0]; out_num[tgid*3u+1u] = s_n[0]; out_num[tgid*3u+2u] = s_dn[0];
        out_cnt[tgid*3u+0u] = c_d[0]; out_cnt[tgid*3u+1u] = c_n[0]; out_cnt[tgid*3u+2u] = c_dn[0];
    }
}

// Pointwise backward. N arrives as a saved scalar per term -- it is an integer count under
// the same predicate as the sum and is never differentiated.
//
// NO ALPHA COTANGENT IS PRODUCED. Un-detaching alpha gives max|dL/dalpha| = 0.37, all of it
// from the depth branch; the trainer detaches it and this kernel must not reintroduce it.
kernel void geom_loss_backward(
    device const float* depth_i  [[buffer(0)]],   // (H,W) depth_img from the forward
    device const float* nr_i     [[buffer(1)]],   // (H,W,3) n_r from the forward
    device const float* n_sum    [[buffer(2)]],
    device const float* alpha    [[buffer(3)]],
    device const float* n_d      [[buffer(4)]],
    device const float* gt_depth [[buffer(5)]],
    device const float* gt_norm  [[buffer(6)]],
    device float*       g_depth  [[buffer(7)]],   // (H,W)   dL/d depth_img
    device float*       g_nd     [[buffer(8)]],   // (H,W,3) dL/d n_d  -> feeds nfd_backward
    device float*       g_nsum   [[buffer(9)]],   // (H,W,3) dL/d n_sum
    constant uint2&     dim      [[buffer(10)]],
    constant uint&      space    [[buffer(11)]],
    constant float4&    wts      [[buffer(12)]],  // w_depth, w_normal, w_dn, unused
    constant float4&    invN     [[buffer(13)]],  // 1/Nd, 1/Nn, 1/Ndn, unused
    device const float* keep     [[buffer(14)]],
    constant uint&      has_keep [[buffer(15)]],
    uint gidx [[thread_position_in_grid]])
{
    const uint H = dim.x, W = dim.y;
    if (gidx >= H * W) return;
    const uint o3 = gidx * 3u;
    const float k  = has_keep ? keep[gidx] : 1.0f;
    const float a  = max(alpha[gidx], 1e-10f);
    const float di = depth_i[gidx];
    const float3 nr = float3(nr_i[o3], nr_i[o3+1], nr_i[o3+2]);

    float gd_acc = 0.0f;
    float3 gnr = float3(0.0f), gnd = float3(0.0f);

    const float gd = gt_depth[gidx] * k;
    if (gd > 0.0f) {
        if (space == 0u) {
            if (di > 0.0f) {
                const float r = 1.0f / di - 1.0f / gd;
                gd_acc += wts.x * invN.x * sign(r) * (-1.0f / (di * di));
            }
            // di <= 0: the uncovered lane's value gradient is exactly 0 (Brush Count mode
            // produces NaN here; guarding before the reciprocal is a deliberate divergence)
        } else {
            gd_acc += wts.x * invN.x * sign(di - gd);
        }
    }
    const float3 gn = float3(gt_norm[o3], gt_norm[o3+1], gt_norm[o3+2]) * k;
    if (length(gn) > 0.5f) gnr += wts.y * invN.y * sign(nr - gn);

    const float3 nd = float3(n_d[o3], n_d[o3+1], n_d[o3+2]);
    if (alpha[gidx] * k > 0.5f && length(nd) > 0.5f && length(nr) > 0.5f) {
        gnd += -wts.z * invN.z * nr;          // ghat_n_d = -m n_r / N
        gnr += -wts.z * invN.z * nd;          // ghat_n_r = -m n_d / N
    }

    // n_r chain: x = n_sum/a, n_r = x/|x|. Alpha cancels exactly, so
    // dL/dn_sum = (I - n_r n_r^T) ghat_n_r / |x|, unclamped.
    float3 x = float3(n_sum[o3], n_sum[o3+1], n_sum[o3+2]) / a;
    const float xl = length(x);
    float3 gx;
    if (xl < 1e-6f) gx = gnr / 1e-6f;                       // clamped branch: NO projection
    else            gx = (gnr - nr * dot(nr, gnr)) / xl;
    g_nsum[o3] = gx.x / a; g_nsum[o3+1] = gx.y / a; g_nsum[o3+2] = gx.z / a;

    g_depth[gidx] = gd_acc;
    g_nd[o3] = gnd.x; g_nd[o3+1] = gnd.y; g_nd[o3+2] = gnd.z;
}
