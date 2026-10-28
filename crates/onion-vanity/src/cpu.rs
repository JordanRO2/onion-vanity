//! Slow dalek reference walker: the `--cpu` search mode and the correctness oracle for the GPU
//! search kernel. Every candidate's public key is recomputed from scratch (`(a0+8m)·B`), which
//! is deliberately independent of the kernel's incremental point arithmetic.

use std::sync::atomic::{AtomicBool, Ordering};

use crate::pattern;
use crate::scalar::{scrub_stack, ClampedScalar};

pub struct CpuHit {
    pub scalar: ClampedScalar,
    pub pubkey: [u8; 32],
    pub m: i64,
}

/// Reference set: `(m, pubkey)` for every `m` in `[lo, hi]` whose key matches a `(mask, target)`
/// pattern. Candidates whose `a0 + 8m` is not a valid clamped scalar are skipped.
pub fn walk(a0: &ClampedScalar, lo: i64, hi: i64, patterns: &[(u64, u64)]) -> Vec<(i64, [u8; 32])> {
    let mut hits = Vec::new();
    for m in lo..=hi {
        if let Ok(a) = a0.add_8m(m) {
            let pk = a.public_compressed();
            if patterns.iter().any(|&(mask, t)| pattern::matches(mask, t, &pk)) {
                hits.push((m, pk));
            }
        }
    }
    hits
}

/// Draw random start scalars and walk `±h` until a pattern matches or `running` is cleared.
pub fn search(patterns: &[(u64, u64)], h: i64, running: &AtomicBool) -> Option<CpuHit> {
    while running.load(Ordering::Relaxed) {
        let a0 = ClampedScalar::random();
        for m in -h..=h {
            if let Ok(a) = a0.add_8m(m) {
                let pk = a.public_compressed();
                if patterns.iter().any(|&(mask, t)| pattern::matches(mask, t, &pk)) {
                    return Some(CpuHit { scalar: a, pubkey: pk, m });
                }
            }
        }
        scrub_stack();
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{address, verify};

    #[test]
    fn walk_hits_are_correct() {
        let (mask, t) = pattern::compile("a").unwrap(); // 5 bits: ~1 in 32
        let a0 = ClampedScalar::random();
        for (m, pk) in walk(&a0, -200, 200, &[(mask, t)]) {
            assert!(pattern::matches(mask, t, &pk), "reported hit must match");
            assert_eq!(a0.add_8m(m).unwrap().public_compressed(), pk, "pubkey must be a0+8m");
        }
    }

    #[test]
    fn search_finds_and_verifies() {
        let (mask, t) = pattern::compile("be").unwrap(); // 10 bits, ~1024 tries
        let running = AtomicBool::new(true);
        let hit = search(&[(mask, t)], 64, &running).expect("must find a 2-char match");
        assert!(pattern::matches(mask, t, &hit.pubkey));
        assert_eq!(hit.scalar.public_compressed(), hit.pubkey);
        assert!(address::encode(&hit.pubkey).starts_with("be"));

        // The found key passes full rule-5 verification.
        let nonce = verify::draw_nonce();
        let mut secret = [0u8; 64];
        secret[..32].copy_from_slice(hit.scalar.as_bytes());
        secret[32..].copy_from_slice(&nonce[..]);
        let v = verify::verify_key(&secret, &hit.pubkey, &["be"]).unwrap();
        assert_eq!(v.credited, Some(0));
    }
}
