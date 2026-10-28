//! GPU search kernel differential test, against the cpu.rs reference walker. Needs the RTX
//! 5080; run with `cargo test -p onion-vanity -- --ignored`. Lives in the crate (not an integration
//! test) so it can use both cpu.rs and the ov-gpu dependency.

use std::collections::HashSet;

use curve25519_dalek::{edwards::EdwardsPoint, scalar::Scalar};
use ov_gpu::{decompress_points, CudaContext, Searcher};

use crate::cpu;
use crate::scalar::{self, ClampedScalar};

const H: u32 = ov_gpu::OV_H;
const N: usize = 256;
const NB: i32 = 2;
const W: i64 = 2 * H as i64 + 1; // 513 candidates per batch

fn xs(st: &mut u64) -> [u8; 32] {
    let mut b = [0u8; 32];
    for c in b.chunks_mut(8) {
        *st ^= *st << 13;
        *st ^= *st >> 7;
        *st ^= *st << 17;
        c.copy_from_slice(&st.to_le_bytes());
    }
    b
}

// Compressed jT = (8j)·B for j=1..=H, and the giant step G = (2H+1)·8·B.
fn comp_tables() -> (Vec<u8>, Vec<u8>) {
    let mut table = Vec::with_capacity(H as usize * 32);
    for j in 1..=H as u64 {
        table.extend_from_slice(&EdwardsPoint::mul_base(&Scalar::from(8 * j)).compress().to_bytes());
    }
    let giant = EdwardsPoint::mul_base(&Scalar::from(8 * (2 * H as u64 + 1)))
        .compress()
        .to_bytes()
        .to_vec();
    (table, giant)
}

fn epoch_scalars(seed: u64) -> Vec<ClampedScalar> {
    let mut st = seed;
    (0..N)
        .map(|_| ClampedScalar::from_bytes(scalar::clamp(xs(&mut st))).unwrap())
        .collect()
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn search_differential() {
    let ctx = ov_gpu::open().unwrap();
    let (ct, cg) = comp_tables();
    let mut searcher = Searcher::new(&ctx, H, N, &ct, &cg).unwrap();

    // 15 slots (kernel max), ~385 hits/launch (< 1024 buffer). Slot 1 (10-bit) is a superset of slot
    // 0 (12-bit): a candidate matching both must still be reported once (first match wins).
    let mut pats: Vec<(u64, u64)> = vec![(0xFFF, 0), (0x3FF, 0)];
    pats.extend([(!0u64, 0x0123_4567_89AB_CDEFu64); 12]); // never-matching fillers
    pats.push((0xFFFu64 << 12, 0x5u64 << 12));
    let (masks, targets): (Vec<u64>, Vec<u64>) = pats.iter().copied().unzip();
    let lo = -(H as i64);
    let hi = 3 * NB as i64 * W - H as i64 - 1; // [-256, 2821]

    for seed in [0x1111_1111_1111_1111u64, 0x2222_3333_4444_5555, 0xdead_beef_cafe_babe] {
        let a0 = epoch_scalars(seed);
        for a in &a0 {
            // The GPU does not clamp-check; walk skips non-clamped m. For this tiny range a random
            // a0 never hits a clamp boundary, so the sets are exact — assert the extremes hold.
            assert!(a.add_8m(lo).is_ok() && a.add_8m(hi).is_ok());
        }

        let mut comp_a0 = Vec::with_capacity(N * 32);
        for a in &a0 {
            comp_a0.extend_from_slice(&a.public_compressed());
        }
        searcher.upload_epoch(&comp_a0).unwrap();
        searcher.set_patterns(&masks, &targets, 0, 1).unwrap(); // canary disabled (mask 0, target 1)

        let mut gpu: HashSet<(i32, i64)> = HashSet::new();
        for l in 0..3i32 {
            let r = searcher.run_launch(NB, l).unwrap();
            assert_eq!(r.status, 0, "seed {seed:x} launch {l} status");
            assert!(!r.overflow, "seed {seed:x} launch {l} overflow");
            assert_eq!(r.count as usize, r.hits.len(), "seed {seed:x} launch {l} count vs hits");
            for h in &r.hits {
                assert_eq!(h.launch, l);
                let key = (h.thread, l as i64 * NB as i64 * W + h.offset as i64);
                assert!(gpu.insert(key), "seed {seed:x}: candidate {key:?} reported more than once");
            }
        }

        let mut reference: HashSet<(i32, i64)> = HashSet::new();
        for (t, a) in a0.iter().enumerate() {
            for (m, _pk) in cpu::walk(a, lo, hi, &pats) {
                reference.insert((t as i32, m));
            }
        }

        assert_eq!(
            gpu, reference,
            "seed {seed:x}: gpu {} hits vs cpu {} hits",
            gpu.len(),
            reference.len()
        );
        assert!(!reference.is_empty(), "test pattern produced no hits");
    }
}

fn setup(ctx: &std::sync::Arc<CudaContext>, seed: u64) -> (Searcher, Vec<ClampedScalar>, Vec<u8>) {
    let (ct, cg) = comp_tables();
    let mut s = Searcher::new(ctx, H, N, &ct, &cg).unwrap();
    let a0 = epoch_scalars(seed);
    let mut comp = Vec::with_capacity(N * 32);
    for a in &a0 {
        comp.extend_from_slice(&a.public_compressed());
    }
    s.upload_epoch(&comp).unwrap();
    (s, a0, comp)
}

// A candidate's canary target: the canonical low64 of (a0+8m)·B (bit 255 is the x sign, not in [0..8)).
fn canary_target(a: &ClampedScalar, m: i64) -> u64 {
    let pk = a.add_8m(m).unwrap().public_compressed();
    u64::from_le_bytes(pk[..8].try_into().unwrap())
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn canary_reported_once() {
    let ctx = ov_gpu::open().unwrap();
    let (mut s, a0, _) = setup(&ctx, 0xa1a1_a1a1_a1a1_a1a1);
    let (tc, mc) = (100usize, 1000i64); // launch 1
    s.set_patterns(&[], &[], !0u64, canary_target(&a0[tc], mc)).unwrap(); // no user patterns
    let lc = (mc + H as i64) / (NB as i64 * W);
    let off = mc - lc * NB as i64 * W;
    let mut found = 0;
    for l in 0..3i32 {
        let r = s.run_launch(NB, l).unwrap();
        assert!(r.hits.is_empty(), "n_active=0 must yield no user hits");
        if let Some(c) = r.canary {
            found += 1;
            assert_eq!((c.thread, c.launch, c.offset as i64), (tc as i32, lc as i32, off));
        }
    }
    assert_eq!(found, 1, "canary must be reported exactly once");
}

// Each special m, as a full 64-bit user pattern, must be reported exactly once at the (launch, offset)
// the host formula m = launch*NB*W + offset predicts.
#[test]
#[ignore = "needs a CUDA GPU"]
fn special_offsets_exact() {
    let ctx = ov_gpu::open().unwrap();
    let (mut s, a0, comp) = setup(&ctx, 0xd4d4_d4d4_d4d4_d4d4);
    let (h, w, lw) = (H as i64, W, NB as i64 * W);
    let cases = [
        ("batch0 center", 0),
        ("batch0 j=-H", -h),
        ("batch0 j=+H", h),
        ("batch1 j=-H", w - h),
        ("batch1 center", w),
        ("batch1 j=+H", w + h),
        ("launch1 first", lw - h),
        ("launch1 mid", lw + 37),
    ];
    for (i, (name, m)) in cases.into_iter().enumerate() {
        let t = (i * 29 + 3) % N;
        s.upload_epoch(&comp).unwrap(); // fresh state: launches advance the persistent base
        s.set_patterns(&[!0u64], &[canary_target(&a0[t], m)], 0, 1).unwrap();
        let launch = (m + h) / lw;
        let offset = m - launch * lw;
        let mut found = Vec::new();
        for l in 0..3i32 {
            let r = s.run_launch(NB, l).unwrap();
            assert!(!r.overflow, "{name}: overflow");
            assert_eq!(r.count as usize, r.hits.len(), "{name}: count vs hits");
            found.extend(r.hits.iter().map(|x| (x.thread, x.launch, x.offset as i64)));
        }
        assert_eq!(found, [(t as i32, launch as i32, offset)], "{name} (m={m})");
    }
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn forced_overflow_keeps_canary() {
    let ctx = ov_gpu::open().unwrap();
    let (mut s, a0, _) = setup(&ctx, 0xb2b2_b2b2_b2b2_b2b2);
    let (tc, mc) = (50usize, 200i64); // launch 0
    s.set_patterns(&[0xF], &[0], !0u64, canary_target(&a0[tc], mc)).unwrap(); // 4-bit floods >1024
    let (mut overflow_seen, mut canaries) = (false, 0);
    for l in 0..3i32 {
        let r = s.run_launch(NB, l).unwrap();
        if r.overflow {
            overflow_seen = true;
            assert!(r.count > 1024, "overflow bit set but count {}", r.count);
        }
        if r.canary.is_some() {
            canaries += 1;
        }
    }
    assert!(overflow_seen, "4-bit pattern must overflow the 1024-slot buffer");
    assert_eq!(canaries, 1, "canary must survive hit-buffer overflow");
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn invalid_point_sets_status() {
    let ctx = ov_gpu::open().unwrap();
    // Find a 32-byte value that is not a valid compressed point.
    let bad = (0u8..64)
        .map(|k| [k.wrapping_mul(37).wrapping_add(5); 32])
        .find(|c| decompress_points(&ctx, c).unwrap().1[0] == 1)
        .expect("a non-curve point");

    let (ct, cg) = comp_tables();
    let mut s = Searcher::new(&ctx, H, N, &ct, &cg).unwrap();
    let a0 = epoch_scalars(0xc3c3_c3c3_c3c3_c3c3);
    let mut comp = Vec::with_capacity(N * 32);
    for a in &a0 {
        comp.extend_from_slice(&a.public_compressed());
    }
    comp[0..32].copy_from_slice(&bad); // thread 0's base is off-curve
    s.upload_epoch(&comp).unwrap();
    s.set_patterns(&[], &[], 0, 1).unwrap();
    let r = s.run_launch(NB, 0).unwrap();
    assert_ne!(r.status & 1, 0, "an invalid base point must set the status bit");
}
