// onion-vanity device kernels: add_one smoke test, field (t_fe_ops) and group
// (decompress_points, t_ge_pair) differential-test entries, throughput benches, search kernel.
#include "fe.cuh"
#include "ge.cuh"

extern "C" __global__ void add_one(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += 1.0f;
}

// Field differential test: one op per thread over flat 8-u32 (little-endian) elements.
// op 0: r = a*b   1: r = a^2   2: r = a^-1 (Fermat)   3: r = freeze(a)   4: r = a+b   5: r = a-b.
extern "C" __global__ void t_fe_ops(int op, const uint32_t* a, const uint32_t* b,
                                    uint32_t* out, int n) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    uint32_t x[8], y[8], r[8];
#pragma unroll
    for (int i = 0; i < 8; i++) {
        x[i] = a[t * 8 + i];
        y[i] = b[t * 8 + i];
    }
    switch (op) {
        case 0: fe_mul(r, x, y); break;
        case 1: fe_sqr(r, x); break;
        case 2: fe_invert(r, x); break;
        case 3: fe_freeze(r, x); break;
        case 4: fe_add(r, x, y); break;
        case 5: fe_sub(r, x, y); break;
    }
#pragma unroll
    for (int i = 0; i < 8; i++) out[t * 8 + i] = r[i];
}

// clock64 throughput benches: each thread runs K dependent ops; out[tid] = SM-cycle span.
// cycles-per-op = span / (warps_per_SM * K) at full occupancy (see ov-gpu::bench_cycles).
extern "C" __global__ void bench_fe_mul(long long* out, int K) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t a[8], b[8], r[8];
#pragma unroll
    for (int i = 0; i < 8; i++) {
        a[i] = 0x10000003u + (uint32_t)tid + i;
        b[i] = 0x2000000bu + (uint32_t)tid * 7u + i;
    }
    long long t0 = clock64();
#pragma unroll 1
    for (int k = 0; k < K; k++) {
        fe_mul(r, a, b);
#pragma unroll
        for (int i = 0; i < 8; i++) {
            a[i] = r[i];
            b[i] = r[(i + 1) & 7] + (uint32_t)k;
        }
    }
    long long t1 = clock64();
    uint32_t s = 0;
#pragma unroll
    for (int i = 0; i < 8; i++) s += r[i];
    out[tid] = (t1 - t0) + (s == 0xdeadbeefu ? 1 : 0); // sink defeats dead-code elimination
}

extern "C" __global__ void bench_fe_sqr(long long* out, int K) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t a[8], r[8];
#pragma unroll
    for (int i = 0; i < 8; i++) a[i] = 0x10000003u + (uint32_t)tid + i;
    long long t0 = clock64();
#pragma unroll 1
    for (int k = 0; k < K; k++) {
        fe_sqr(r, a);
#pragma unroll
        for (int i = 0; i < 8; i++) a[i] = r[i] + (i == 0 ? (uint32_t)k : 0u);
    }
    long long t1 = clock64();
    uint32_t s = 0;
#pragma unroll
    for (int i = 0; i < 8; i++) s += r[i];
    out[tid] = (t1 - t0) + (s == 0xdeadbeefu ? 1 : 0);
}

// Decompress n compressed points to affine (x,y); status[t]=1 if the point is invalid.
extern "C" __global__ void decompress_points(const unsigned char* in, uint32_t* out_xy,
                                             unsigned char* status, int n) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    uint32_t x[8], y[8];
    int bad = ge_decompress(x, y, in + t * 32);
#pragma unroll
    for (int i = 0; i < 8; i++) {
        out_xy[t * 16 + i] = x[i];
        out_xy[t * 16 + 8 + i] = y[i];
    }
    status[t] = (unsigned char)bad;
}

// Dual-law differential: for each pair (base P, table jT) emit freeze bytes of y(P+jT), y(P-jT).
extern "C" __global__ void t_ge_pair(const unsigned char* base, const unsigned char* tabj,
                                     unsigned char* out_yp, unsigned char* out_ym,
                                     unsigned char* status, int m) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= m) return;
    uint32_t x1[8], y1[8], x2[8], y2[8], x1y1[8], x2y2[8];
    int bad = ge_decompress(x1, y1, base + t * 32);
    bad |= ge_decompress(x2, y2, tabj + t * 32);
    fe_mul(x1y1, x1, y1);
    fe_mul(x2y2, x2, y2);
    uint32_t den_p[8], den_m[8], num_p[8], num_m[8], ip[8], im[8], yp[8], ym[8];
    dual_pair(x1, y1, x1y1, x2, y2, x2y2, den_p, den_m, num_p, num_m);
    fe_invert(ip, den_p);
    fe_invert(im, den_m);
    fe_mul(yp, num_p, ip);
    fe_mul(ym, num_m, im);
    fe_tobytes(out_yp + t * 32, yp);
    fe_tobytes(out_ym + t * 32, ym);
    status[t] = (unsigned char)bad;
}

// ---- search kernel ----

#define OV_MAXPAT 15
#define OV_W (2 * OV_H + 1)
#define OV_ST_INVALID 1u
#define OV_ST_ZERODEN 2u

// Pattern table + canary, uniform reads broadcast from constant memory (filled per epoch by host).
struct ov_const_t {
    unsigned long long pat_mask[OV_MAXPAT];   // little-endian low64(y) domain
    unsigned long long pat_target[OV_MAXPAT];
    unsigned long long canary_mask;
    unsigned long long canary_target;
    int n_active;
    int _pad;
};
__constant__ ov_const_t ov_c;

__device__ __forceinline__ void ov_copy(uint32_t d[8], const uint32_t s[8]) {
#pragma unroll
    for (int i = 0; i < 8; i++) d[i] = s[i];
}

// bit-255 fold then scan the active pattern slots; first match writes a hit via atomicAdd. The
// canary is a separate record so a full hit buffer can never drop it.
__device__ void ov_prefilter(const uint32_t y[8], int thread, int offset, int launch, unsigned *buf) {
    unsigned long long lo = (unsigned long long)y[0] | ((unsigned long long)y[1] << 32);
    unsigned long long v = lo + 19ull * (unsigned long long)(y[7] >> 31);
#pragma unroll 1
    for (int k = 0; k < ov_c.n_active; k++) {
        if ((v & ov_c.pat_mask[k]) == ov_c.pat_target[k]) {
            unsigned slot = atomicAdd(&buf[1], 1u);
            if (slot < 1024u) {
                unsigned *h = buf + 8 + 3u * slot;
                h[0] = (unsigned)thread;
                h[1] = (unsigned)launch;
                h[2] = (unsigned)offset;
            } else {
                buf[2] = 1u;
            }
            break;
        }
    }
    if ((v & ov_c.canary_mask) == ov_c.canary_target) {
        buf[3] = 1u;
        buf[4] = (unsigned)thread;
        buf[5] = (unsigned)offset;
        buf[6] = (unsigned)launch;
    }
}

// Build the epoch-independent tables from host-decompressed compressed points (T = 8B is fixed):
// comp_jt[t] = compressed (8·(t+1))·B for j=1..H; comp_g = compressed (2H+1)·8·B.
extern "C" __global__ void build_tables(const unsigned char *comp_jt, ge_tab *table,
                                        const unsigned char *comp_g, ge_niels *giant, int h) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < h) {
        uint32_t x2[8], y2[8];
        ge_decompress(x2, y2, comp_jt + t * 32); // dalek inputs are always valid
        ov_copy(table[t].x2, x2);
        ov_copy(table[t].y2, y2);
        fe_mul(table[t].x2y2, x2, y2);
    }
    if (t == 0) {
        uint32_t gx[8], gy[8];
        ge_decompress(gx, gy, comp_g);
        ge_niels_build(giant, gx, gy);
    }
}

// One launch: nb batches of 2H+1 candidates per thread, one inversion per batch (Montgomery), the
// persistent affine base advanced by G each batch. offset = b·(2H+1) + j, j in [-H,H].
extern "C" __global__ void __launch_bounds__(128, 2) search(
    uint32_t *state, const ge_tab *table, const ge_niels *giant, const unsigned char *valid,
    unsigned *buf, int nb, int launch, int n_threads) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_threads) return;
    if (!valid[tid]) {
        atomicOr(&buf[0], OV_ST_INVALID);
        return;
    }

    uint32_t x1[8], y1[8], x1y1[8];
#pragma unroll
    for (int i = 0; i < 8; i++) {
        x1[i] = state[tid * 16 + i];
        y1[i] = state[tid * 16 + 8 + i];
    }
    fe_mul(x1y1, x1, y1);

    uint32_t prod_p[OV_H][8], prod_m[OV_H][8]; // per-thread local running-product snapshots

#pragma unroll 1
    for (int b = 0; b < nb; b++) {
        int base_off = b * OV_W;
        ov_prefilter(y1, tid, base_off, launch, buf); // center (m=0): y = y1

        uint32_t accp[8], accm[8];
        ov_copy(accp, FE_ONE);
        ov_copy(accm, FE_ONE);
#pragma unroll 1
        for (int jj = 1; jj <= OV_H; jj++) {
            const ge_tab *te = &table[jj - 1];
            uint32_t den_p[8], den_m[8], num_p[8], num_m[8];
            dual_pair(x1, y1, x1y1, te->x2, te->y2, te->x2y2, den_p, den_m, num_p, num_m);
            ov_copy(prod_p[jj - 1], accp);
            ov_copy(prod_m[jj - 1], accm);
            fe_mul(accp, accp, den_p);
            fe_mul(accm, accm, den_m);
        }

        uint32_t X3[8], Y3[8], Z3[8];
        giant_step(x1, y1, x1y1, giant, X3, Y3, Z3);
        uint32_t PP[8], grand[8], ig[8];
        fe_mul(PP, accp, accm);
        fe_mul(grand, PP, Z3);
        if (fe_iszero(grand)) {
            atomicOr(&buf[0], OV_ST_ZERODEN);
            return;
        }
        fe_invert(ig, grand);
        uint32_t invZ3[8], invPP[8], rinv_p[8], rinv_m[8];
        fe_mul(invZ3, ig, PP);      // 1/Z3
        fe_mul(invPP, ig, Z3);      // 1/(accp·accm)
        fe_mul(rinv_p, invPP, accm);// 1/accp
        fe_mul(rinv_m, invPP, accp);// 1/accm

#pragma unroll 1
        for (int jj = OV_H; jj >= 1; jj--) {
            const ge_tab *te = &table[jj - 1];
            uint32_t den_p[8], den_m[8], num_p[8], num_m[8];
            dual_pair(x1, y1, x1y1, te->x2, te->y2, te->x2y2, den_p, den_m, num_p, num_m);
            uint32_t inv_dp[8], inv_dm[8], yp[8], ym[8];
            fe_mul(inv_dp, rinv_p, prod_p[jj - 1]);
            fe_mul(rinv_p, rinv_p, den_p);
            fe_mul(inv_dm, rinv_m, prod_m[jj - 1]);
            fe_mul(rinv_m, rinv_m, den_m);
            fe_mul(yp, num_p, inv_dp);
            fe_mul(ym, num_m, inv_dm);
            ov_prefilter(yp, tid, base_off + jj, launch, buf);
            ov_prefilter(ym, tid, base_off - jj, launch, buf);
        }

        fe_mul(x1, X3, invZ3); // advance base to P + G
        fe_mul(y1, Y3, invZ3);
        fe_mul(x1y1, x1, y1);
    }

#pragma unroll
    for (int i = 0; i < 8; i++) {
        state[tid * 16 + i] = x1[i];
        state[tid * 16 + 8 + i] = y1[i];
    }
}
