// Field arithmetic over GF(2^255-19). 8 saturated 32-bit little-endian limbs, loosely reduced:
// every result fits in [0, 2^256) (congruent mod p via 2^256 == 38), canonical only in fe_freeze.
// fe_mul is a product-scanned (Comba) multiply whose per-(i,j) carry chain lives in one asm
// statement so ptxas fuses each into a single IMAD.WIDE.U32 on sm_120 (72 muls, 0 IMAD.HI).
// Verified against num-bigint mod p on the RTX 5080.
#ifndef OV_FE_CUH
#define OV_FE_CUH
#include <stdint.h>

__device__ const uint32_t FE_ZERO[8] = {0, 0, 0, 0, 0, 0, 0, 0};
__device__ const uint32_t FE_ONE[8] = {1, 0, 0, 0, 0, 0, 0, 0};
__device__ const uint32_t FE_TWO[8] = {2, 0, 0, 0, 0, 0, 0, 0};

// One (i,j) partial product into a 3-word column accumulator (c0,c1,c2). mad.lo.cc sets CF,
// madc.hi.cc uses+sets CF, addc consumes it; the chain is contained in the triple.
#define OV_MAC(c0, c1, c2, a, b)                 \
    asm("mad.lo.cc.u32  %0, %3, %4, %0;\n\t"     \
        "madc.hi.cc.u32 %1, %3, %4, %1;\n\t"     \
        "addc.u32       %2, %2, 0;\n\t"          \
        : "+r"(c0), "+r"(c1), "+r"(c2)           \
        : "r"(a), "r"(b))

// Fold a 16-limb product p[] into r[] in [0,2^256): r = L + 38*H, double-folded.
__device__ __forceinline__ void fe_fold(uint32_t r[8], const uint32_t p[16]) {
    uint64_t acc = 0;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        acc += (uint64_t)p[i] + (uint64_t)38u * p[i + 8];
        r[i] = (uint32_t)acc;
        acc >>= 32;
    }
    uint64_t a2 = (uint64_t)r[0] + 38u * acc; // acc <= 38
    r[0] = (uint32_t)a2;
    a2 >>= 32;
#pragma unroll
    for (int i = 1; i < 8; i++) {
        a2 += r[i];
        r[i] = (uint32_t)a2;
        a2 >>= 32;
    }
    r[0] += 38u * (uint32_t)a2; // final carry bit
}

__device__ __forceinline__ void fe_mul(uint32_t r[8], const uint32_t a[8], const uint32_t b[8]) {
    uint32_t p[16];
    uint32_t c0 = 0, c1 = 0, c2 = 0;
#define COL(k) p[k] = c0; c0 = c1; c1 = c2; c2 = 0;
    OV_MAC(c0, c1, c2, a[0], b[0]); COL(0);
    OV_MAC(c0, c1, c2, a[0], b[1]); OV_MAC(c0, c1, c2, a[1], b[0]); COL(1);
    OV_MAC(c0, c1, c2, a[0], b[2]); OV_MAC(c0, c1, c2, a[1], b[1]); OV_MAC(c0, c1, c2, a[2], b[0]); COL(2);
    OV_MAC(c0, c1, c2, a[0], b[3]); OV_MAC(c0, c1, c2, a[1], b[2]); OV_MAC(c0, c1, c2, a[2], b[1]); OV_MAC(c0, c1, c2, a[3], b[0]); COL(3);
    OV_MAC(c0, c1, c2, a[0], b[4]); OV_MAC(c0, c1, c2, a[1], b[3]); OV_MAC(c0, c1, c2, a[2], b[2]); OV_MAC(c0, c1, c2, a[3], b[1]); OV_MAC(c0, c1, c2, a[4], b[0]); COL(4);
    OV_MAC(c0, c1, c2, a[0], b[5]); OV_MAC(c0, c1, c2, a[1], b[4]); OV_MAC(c0, c1, c2, a[2], b[3]); OV_MAC(c0, c1, c2, a[3], b[2]); OV_MAC(c0, c1, c2, a[4], b[1]); OV_MAC(c0, c1, c2, a[5], b[0]); COL(5);
    OV_MAC(c0, c1, c2, a[0], b[6]); OV_MAC(c0, c1, c2, a[1], b[5]); OV_MAC(c0, c1, c2, a[2], b[4]); OV_MAC(c0, c1, c2, a[3], b[3]); OV_MAC(c0, c1, c2, a[4], b[2]); OV_MAC(c0, c1, c2, a[5], b[1]); OV_MAC(c0, c1, c2, a[6], b[0]); COL(6);
    OV_MAC(c0, c1, c2, a[0], b[7]); OV_MAC(c0, c1, c2, a[1], b[6]); OV_MAC(c0, c1, c2, a[2], b[5]); OV_MAC(c0, c1, c2, a[3], b[4]); OV_MAC(c0, c1, c2, a[4], b[3]); OV_MAC(c0, c1, c2, a[5], b[2]); OV_MAC(c0, c1, c2, a[6], b[1]); OV_MAC(c0, c1, c2, a[7], b[0]); COL(7);
    OV_MAC(c0, c1, c2, a[1], b[7]); OV_MAC(c0, c1, c2, a[2], b[6]); OV_MAC(c0, c1, c2, a[3], b[5]); OV_MAC(c0, c1, c2, a[4], b[4]); OV_MAC(c0, c1, c2, a[5], b[3]); OV_MAC(c0, c1, c2, a[6], b[2]); OV_MAC(c0, c1, c2, a[7], b[1]); COL(8);
    OV_MAC(c0, c1, c2, a[2], b[7]); OV_MAC(c0, c1, c2, a[3], b[6]); OV_MAC(c0, c1, c2, a[4], b[5]); OV_MAC(c0, c1, c2, a[5], b[4]); OV_MAC(c0, c1, c2, a[6], b[3]); OV_MAC(c0, c1, c2, a[7], b[2]); COL(9);
    OV_MAC(c0, c1, c2, a[3], b[7]); OV_MAC(c0, c1, c2, a[4], b[6]); OV_MAC(c0, c1, c2, a[5], b[5]); OV_MAC(c0, c1, c2, a[6], b[4]); OV_MAC(c0, c1, c2, a[7], b[3]); COL(10);
    OV_MAC(c0, c1, c2, a[4], b[7]); OV_MAC(c0, c1, c2, a[5], b[6]); OV_MAC(c0, c1, c2, a[6], b[5]); OV_MAC(c0, c1, c2, a[7], b[4]); COL(11);
    OV_MAC(c0, c1, c2, a[5], b[7]); OV_MAC(c0, c1, c2, a[6], b[6]); OV_MAC(c0, c1, c2, a[7], b[5]); COL(12);
    OV_MAC(c0, c1, c2, a[6], b[7]); OV_MAC(c0, c1, c2, a[7], b[6]); COL(13);
    OV_MAC(c0, c1, c2, a[7], b[7]); p[14] = c0; p[15] = c1;
#undef COL
    fe_fold(r, p);
}

__device__ __forceinline__ void fe_sqr(uint32_t r[8], const uint32_t a[8]) {
    uint32_t p[16];
    uint32_t c0 = 0, c1 = 0, c2 = 0;
#define COL(k) p[k] = c0; c0 = c1; c1 = c2; c2 = 0;
    COL(0); // col 0 is diagonal only, added below
    OV_MAC(c0, c1, c2, a[0], a[1]); COL(1);
    OV_MAC(c0, c1, c2, a[0], a[2]); COL(2);
    OV_MAC(c0, c1, c2, a[0], a[3]); OV_MAC(c0, c1, c2, a[1], a[2]); COL(3);
    OV_MAC(c0, c1, c2, a[0], a[4]); OV_MAC(c0, c1, c2, a[1], a[3]); COL(4);
    OV_MAC(c0, c1, c2, a[0], a[5]); OV_MAC(c0, c1, c2, a[1], a[4]); OV_MAC(c0, c1, c2, a[2], a[3]); COL(5);
    OV_MAC(c0, c1, c2, a[0], a[6]); OV_MAC(c0, c1, c2, a[1], a[5]); OV_MAC(c0, c1, c2, a[2], a[4]); COL(6);
    OV_MAC(c0, c1, c2, a[0], a[7]); OV_MAC(c0, c1, c2, a[1], a[6]); OV_MAC(c0, c1, c2, a[2], a[5]); OV_MAC(c0, c1, c2, a[3], a[4]); COL(7);
    OV_MAC(c0, c1, c2, a[1], a[7]); OV_MAC(c0, c1, c2, a[2], a[6]); OV_MAC(c0, c1, c2, a[3], a[5]); COL(8);
    OV_MAC(c0, c1, c2, a[2], a[7]); OV_MAC(c0, c1, c2, a[3], a[6]); OV_MAC(c0, c1, c2, a[4], a[5]); COL(9);
    OV_MAC(c0, c1, c2, a[3], a[7]); OV_MAC(c0, c1, c2, a[4], a[6]); COL(10);
    OV_MAC(c0, c1, c2, a[4], a[7]); OV_MAC(c0, c1, c2, a[5], a[6]); COL(11);
    OV_MAC(c0, c1, c2, a[5], a[7]); COL(12);
    OV_MAC(c0, c1, c2, a[6], a[7]); COL(13);
    p[14] = c0; p[15] = c1;
#undef COL
    // double the cross sum
    uint32_t top = 0;
#pragma unroll
    for (int i = 0; i < 16; i++) {
        uint32_t nx = p[i] >> 31;
        p[i] = (p[i] << 1) | top;
        top = nx;
    }
    // add the diagonal a[i]^2 at columns 2i, one continuous carry chain
    uint64_t carry = 0;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        uint64_t sq = (uint64_t)a[i] * a[i];
        uint64_t lo = (uint64_t)p[2 * i] + (uint32_t)sq + carry;
        p[2 * i] = (uint32_t)lo;
        carry = lo >> 32;
        uint64_t hi = (uint64_t)p[2 * i + 1] + (uint32_t)(sq >> 32) + carry;
        p[2 * i + 1] = (uint32_t)hi;
        carry = hi >> 32;
    }
    fe_fold(r, p);
}

// z^(p-2) = z^(2^255-21) by the standard addition chain (254 sqr + 11 mul). Cold, kept out of line.
__device__ __noinline__ void fe_invert(uint32_t r[8], const uint32_t z[8]) {
    uint32_t t0[8], t1[8], z9[8], z11[8], z5[8], z10[8], z20[8], z50[8], z100[8], t[8];
    fe_sqr(t0, z);
    fe_sqr(t1, t0); fe_sqr(t1, t1);
    fe_mul(z9, t1, z);
    fe_mul(z11, z9, t0);
    fe_sqr(t, z11);
    fe_mul(z5, t, z9);
    fe_sqr(t, z5); for (int i = 1; i < 5; i++) fe_sqr(t, t); fe_mul(z10, t, z5);
    fe_sqr(t, z10); for (int i = 1; i < 10; i++) fe_sqr(t, t); fe_mul(z20, t, z10);
    fe_sqr(t, z20); for (int i = 1; i < 20; i++) fe_sqr(t, t); fe_mul(t, t, z20);
    fe_sqr(t, t); for (int i = 1; i < 10; i++) fe_sqr(t, t); fe_mul(z50, t, z10);
    fe_sqr(t, z50); for (int i = 1; i < 50; i++) fe_sqr(t, t); fe_mul(z100, t, z50);
    fe_sqr(t, z100); for (int i = 1; i < 100; i++) fe_sqr(t, t); fe_mul(t, t, z100);
    fe_sqr(t, t); for (int i = 1; i < 50; i++) fe_sqr(t, t); fe_mul(t, t, z50);
    fe_sqr(t, t); for (int i = 1; i < 5; i++) fe_sqr(t, t); fe_mul(r, t, z11);
}

// Canonical reduction to [0,p). step1 folds bit255 via 2^255==19; step2 conditionally subtracts p.
__device__ __forceinline__ void fe_freeze(uint32_t r[8], const uint32_t v[8]) {
    uint32_t top = v[7] >> 31;
    uint64_t acc = (uint64_t)v[0] + 19u * top;
    r[0] = (uint32_t)acc;
    acc >>= 32;
#pragma unroll
    for (int i = 1; i < 7; i++) {
        acc += v[i];
        r[i] = (uint32_t)acc;
        acc >>= 32;
    }
    acc += (v[7] & 0x7fffffff);
    r[7] = (uint32_t)acc; // m < 2^255+19
    uint32_t t[8];
    uint64_t cc = (uint64_t)r[0] + 19u;
    t[0] = (uint32_t)cc;
    cc >>= 32;
#pragma unroll
    for (int i = 1; i < 8; i++) {
        cc += r[i];
        t[i] = (uint32_t)cc;
        cc >>= 32;
    }
    uint32_t mask = -(int32_t)(t[7] >> 31); // all-ones iff m >= p
#pragma unroll
    for (int i = 0; i < 8; i++) r[i] = (r[i] & ~mask) | (t[i] & mask);
    r[7] &= 0x7fffffff;
}

// Loose add, result kept in [0, 2^256): fold the top carry back via 2^256 == 38.
__device__ __forceinline__ void fe_add(uint32_t r[8], const uint32_t a[8], const uint32_t b[8]) {
    uint64_t acc = 0;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        acc += (uint64_t)a[i] + b[i];
        r[i] = (uint32_t)acc;
        acc >>= 32;
    }
    uint64_t c = (uint64_t)r[0] + 38u * acc; // acc in {0,1}
    r[0] = (uint32_t)c;
    c >>= 32;
#pragma unroll
    for (int i = 1; i < 8; i++) {
        c += r[i];
        r[i] = (uint32_t)c;
        c >>= 32;
    }
    r[0] += 38u * (uint32_t)c;
}

// Loose subtract: add 4p = 2^257-76 first so the difference is never negative, then fold.
__device__ __forceinline__ void fe_sub(uint32_t r[8], const uint32_t a[8], const uint32_t b[8]) {
    const uint32_t FOURP[9] = {0xFFFFFFB4u, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu,
                               0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0x1u};
    uint32_t s[9];
    uint64_t acc = 0;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        acc += (uint64_t)a[i] + FOURP[i];
        s[i] = (uint32_t)acc;
        acc >>= 32;
    }
    s[8] = (uint32_t)(acc + FOURP[8]);
    // s - b (>= 2^256 - 76 > 0), borrow chain
    int64_t br = 0;
    uint32_t d[9];
#pragma unroll
    for (int i = 0; i < 8; i++) {
        int64_t t = (int64_t)s[i] - (int64_t)b[i] - br;
        if (t < 0) {
            t += (int64_t)1 << 32;
            br = 1;
        } else {
            br = 0;
        }
        d[i] = (uint32_t)t;
    }
    d[8] = (uint32_t)((int64_t)s[8] - br);
    // reduce the 9-limb d (d[8] small) mod 2^256 via 2^256 == 38
    uint64_t c = (uint64_t)d[0] + 38u * (uint64_t)d[8];
    r[0] = (uint32_t)c;
    c >>= 32;
#pragma unroll
    for (int i = 1; i < 8; i++) {
        c += d[i];
        r[i] = (uint32_t)c;
        c >>= 32;
    }
    r[0] += 38u * (uint32_t)c;
}

__device__ __forceinline__ void fe_neg(uint32_t r[8], const uint32_t a[8]) {
    fe_sub(r, FE_ZERO, a);
}

__device__ __forceinline__ int fe_iszero(const uint32_t a[8]) {
    uint32_t f[8];
    fe_freeze(f, a);
    uint32_t z = 0;
#pragma unroll
    for (int i = 0; i < 8; i++) z |= f[i];
    return z == 0;
}

// Load 32 LE bytes; the top bit (y-coordinate sign) is masked off.
__device__ __forceinline__ void fe_frombytes(uint32_t r[8], const unsigned char s[32]) {
#pragma unroll
    for (int i = 0; i < 8; i++) {
        r[i] = (uint32_t)s[4 * i] | ((uint32_t)s[4 * i + 1] << 8) |
               ((uint32_t)s[4 * i + 2] << 16) | ((uint32_t)s[4 * i + 3] << 24);
    }
    r[7] &= 0x7fffffff;
}

__device__ __forceinline__ void fe_tobytes(unsigned char s[32], const uint32_t a[8]) {
    uint32_t f[8];
    fe_freeze(f, a);
#pragma unroll
    for (int i = 0; i < 8; i++) {
        s[4 * i] = (unsigned char)f[i];
        s[4 * i + 1] = (unsigned char)(f[i] >> 8);
        s[4 * i + 2] = (unsigned char)(f[i] >> 16);
        s[4 * i + 3] = (unsigned char)(f[i] >> 24);
    }
}

#endif // OV_FE_CUH
