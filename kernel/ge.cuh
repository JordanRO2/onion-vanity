// Group arithmetic on Edwards curve25519, built on fe.cuh. Decompression (ref10 "approach B",
// no inversion), the y-only dual addition law for P±jT (2 muls per pair, shared cross products),
// and the mixed-addition giant step (hwcd-3). All formulas verified against dalek; the
// decompress and dual-law paths are GPU-checked by decompress_points / t_ge_pair.
#ifndef OV_GE_CUH
#define OV_GE_CUH
#include "fe.cuh"

// Curve constants as little-endian limbs (reversed from the canonical big-endian hex).
// d  = 0x52036cee...135978a3 ; sqrt(-1) = 0x2b832480...4a0ea0b0.
__device__ const uint32_t FE_D[8] = {0x135978a3u, 0x75eb4dcau, 0x4141d8abu, 0x00700a4du,
                                     0x7779e898u, 0x8cc74079u, 0x2b6ffe73u, 0x52036ceeu};
__device__ const uint32_t FE_SQRTM1[8] = {0x4a0ea0b0u, 0xc4ee1b27u, 0xad2fe478u, 0x2f431806u,
                                          0x3dfbd7a7u, 0x2b4d0099u, 0x4fc1df0bu, 0x2b832480u};

struct ge_tab {
    uint32_t x2[8], y2[8], x2y2[8];
};
struct ge_niels {
    uint32_t ypx[8], ymx[8], xy2d[8];
};

// z^((p-5)/8) = z^(2^252-3), for the decompression square root. Cold.
__device__ __noinline__ void fe_pow_p58(uint32_t o[8], const uint32_t z[8]) {
    uint32_t t0[8], t1[8], t2[8];
    int i;
    fe_sqr(t0, z);
    fe_sqr(t1, t0); fe_sqr(t1, t1); fe_mul(t1, z, t1);
    fe_mul(t0, t0, t1); fe_sqr(t0, t0); fe_mul(t0, t1, t0);              // 2^5-1
    fe_sqr(t1, t0); for (i = 1; i < 5; i++) fe_sqr(t1, t1); fe_mul(t0, t1, t0);    // 2^10-1
    fe_sqr(t1, t0); for (i = 1; i < 10; i++) fe_sqr(t1, t1); fe_mul(t1, t1, t0);   // 2^20-1
    fe_sqr(t2, t1); for (i = 1; i < 20; i++) fe_sqr(t2, t2); fe_mul(t1, t2, t1);   // 2^40-1
    fe_sqr(t1, t1); for (i = 1; i < 10; i++) fe_sqr(t1, t1); fe_mul(t0, t1, t0);   // 2^50-1
    fe_sqr(t1, t0); for (i = 1; i < 50; i++) fe_sqr(t1, t1); fe_mul(t1, t1, t0);   // 2^100-1
    fe_sqr(t2, t1); for (i = 1; i < 100; i++) fe_sqr(t2, t2); fe_mul(t1, t2, t1);  // 2^200-1
    fe_sqr(t1, t1); for (i = 1; i < 50; i++) fe_sqr(t1, t1); fe_mul(t0, t1, t0);   // 2^250-1
    fe_sqr(t0, t0); fe_sqr(t0, t0); fe_mul(o, t0, z);                   // 2^252-3
}

// Decompress a 32-byte compressed Edwards point to affine (x, y). Returns 1 if not on the curve.
__device__ int ge_decompress(uint32_t x[8], uint32_t y[8], const unsigned char s[32]) {
    uint32_t u[8], v[8], v3[8], vxx[8], chk[8];
    fe_frombytes(y, s);
    unsigned sign = s[31] >> 7;
    fe_sqr(u, y);                                   // y^2
    fe_mul(v, u, FE_D);                             // d y^2
    fe_sub(u, u, FE_ONE);                           // y^2 - 1
    fe_add(v, v, FE_ONE);                           // d y^2 + 1
    fe_sqr(v3, v); fe_mul(v3, v3, v);               // v^3
    fe_sqr(x, v3); fe_mul(x, x, v); fe_mul(x, x, u); // u v^7
    fe_pow_p58(x, x);
    fe_mul(x, x, v3); fe_mul(x, x, u);              // u v^3 (u v^7)^((p-5)/8)
    fe_sqr(vxx, x); fe_mul(vxx, vxx, v);            // v x^2
    fe_sub(chk, vxx, u);
    int bad = 0;
    if (!fe_iszero(chk)) {
        fe_add(chk, vxx, u);
        if (!fe_iszero(chk)) {
            bad = 1;
        } else {
            fe_mul(x, x, FE_SQRTM1);
        }
    }
    unsigned char xb[32];
    fe_tobytes(xb, x);
    if ((xb[0] & 1u) != sign) fe_neg(x, x);
    if (fe_iszero(x) && sign) bad = 1; // (0,1) has even x; sign=1 is no point
    return bad;
}

// y-only dual addition law. Shares the two cross products, so one pair costs 2 fe_mul here; the
// two y's are recovered later as num_± · inv(den_±). The center candidate (m=0) uses y1 directly.
__device__ void dual_pair(const uint32_t x1[8], const uint32_t y1[8], const uint32_t x1y1[8],
                          const uint32_t x2[8], const uint32_t y2[8], const uint32_t x2y2[8],
                          uint32_t den_p[8], uint32_t den_m[8],
                          uint32_t num_p[8], uint32_t num_m[8]) {
    uint32_t cx[8], cy[8];
    fe_mul(cx, x1, y2);         // x1 y2
    fe_mul(cy, y1, x2);         // y1 x2
    fe_sub(den_p, cx, cy);      // x1y2 - y1x2
    fe_add(den_m, cx, cy);      // x1y2 + y1x2
    fe_sub(num_p, x1y1, x2y2);  // x1y1 - x2y2
    fe_add(num_m, x1y1, x2y2);  // x1y1 + x2y2
}

// 2d·x·y for a niels table entry (2d computed from d to avoid hardcoding a second constant).
__device__ __forceinline__ void ge_niels_build(ge_niels *n, const uint32_t x[8], const uint32_t y[8]) {
    uint32_t d2[8], xy[8];
    fe_add(d2, FE_D, FE_D);
    fe_add(n->ypx, y, x);
    fe_sub(n->ymx, y, x);
    fe_mul(xy, x, y);
    fe_mul(n->xy2d, xy, d2);
}

// Advance affine base P=(x1,y1), Z1=1, by the niels giant step G. Output extended (X3:Y3:Z3) with
// affine = (X3/Z3, Y3/Z3); Z3 is folded into the batch inversion (EFD add-2008-hwcd-3, a=-1).
// __noinline__: called once per batch, so the call cost is negligible, but keeping its 7 live field
// temps out of the search kernel's register allocation improves occupancy.
__device__ __noinline__ void giant_step(const uint32_t x1[8], const uint32_t y1[8], const uint32_t x1y1[8],
                           const ge_niels *G, uint32_t X3[8], uint32_t Y3[8], uint32_t Z3[8]) {
    uint32_t A[8], B[8], C[8], E[8], F[8], GG[8], H[8];
    fe_sub(A, y1, x1); fe_mul(A, A, G->ymx);  // (Y1-X1)(y-x)
    fe_add(B, y1, x1); fe_mul(B, B, G->ypx);  // (Y1+X1)(y+x)
    fe_mul(C, x1y1, G->xy2d);                 // T1·2dxy   (T1=x1y1, Z1=1)
    fe_sub(E, B, A);                          // B - A
    fe_sub(F, FE_TWO, C);                     // D - C     (D = 2Z1 = 2)
    fe_add(GG, FE_TWO, C);                    // D + C
    fe_add(H, B, A);                          // B + A
    fe_mul(X3, E, F);
    fe_mul(Y3, GG, H);
    fe_mul(Z3, F, GG);
}

#endif // OV_GE_CUH
