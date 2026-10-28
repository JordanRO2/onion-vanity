//! Prefix patterns and the host side of the device prefilter.
//!
//! A pattern is a base32 (`a-z2-7`) prefix of the onion address. The address is the big-endian
//! bitstream of the 32-byte compressed key, so an L-character prefix pins the top 5L bits of the
//! first 8 key bytes. `compile` returns `(mask, target)` already byte-swapped into the device's
//! little-endian `low64(y)` domain, so the kernel compares `u64::from_le_bytes(key[..8]) & mask`
//! against `target` with no byte permutes.

/// 12 base32 chars = 60 bits ≤ 64; 13 would need 65.
pub const MAX_LEN: usize = 12;
/// The aggregate selectivity guard rejects a pattern set expected to exceed this many hits/launch.
pub const MAX_EXPECTED_HITS_PER_LAUNCH: f64 = 64.0;

#[derive(Debug, PartialEq)] // not Eq: TooUnselective holds an f64
pub enum PatternError {
    Empty,
    TooLong(usize),
    BadChar(u8),
    TooUnselective { expected_hits: f64 },
}

/// Map one base32 char to its 5-bit value, or `None` if outside `a-z2-7`.
fn char_val(c: u8) -> Option<u8> {
    match c {
        b'a'..=b'z' => Some(c - b'a'),
        b'2'..=b'7' => Some(c - b'2' + 26),
        _ => None,
    }
}

/// Compile a prefix to `(mask, target)` in the little-endian `low64(y)` domain.
pub fn compile(pattern: &str) -> Result<(u64, u64), PatternError> {
    let p = pattern.as_bytes();
    if p.is_empty() {
        return Err(PatternError::Empty);
    }
    if p.len() > MAX_LEN {
        return Err(PatternError::TooLong(p.len()));
    }
    let mut target_be = 0u64;
    for (i, &c) in p.iter().enumerate() {
        let v = char_val(c).ok_or(PatternError::BadChar(c))?;
        target_be |= (v as u64) << (59 - 5 * i); // char i occupies BE bits [59-5i, 63-5i]
    }
    let mask_be = (!0u64) << (64 - 5 * p.len()); // safe: shift in 4..=59 for len 1..=12
    Ok((mask_be.swap_bytes(), target_be.swap_bytes()))
}

/// Prefilter selectivity in bits: 5 per base32 char. Call only on an accepted pattern.
pub fn bits(pattern: &str) -> u32 {
    5 * pattern.len() as u32
}

/// Host reference matcher: identical test to the device's, on a full compressed key.
pub fn matches(mask: u64, target: u64, pubkey: &[u8; 32]) -> bool {
    u64::from_le_bytes(pubkey[..8].try_into().unwrap()) & mask == target
}

/// Candidates examined per launch, `C = threads · nb · (2H+1)`.
pub fn candidates_per_launch(threads: u64, nb: u64, h: u64) -> u64 {
    threads
        .saturating_mul(nb)
        .saturating_mul(2u64.saturating_mul(h).saturating_add(1))
}

/// Expected hits per launch for a pattern set: `Σ C · 2^-bits_i`.
pub fn expected_hits(candidates: u64, bits: &[u32]) -> f64 {
    bits.iter()
        .map(|&b| candidates as f64 * (-(b as f64)).exp2())
        .sum()
}

/// Reject a pattern set whose expected hits exceed the buffer-headroom cap (GPU path only).
pub fn check_selectivity(candidates: u64, bits: &[u32]) -> Result<f64, PatternError> {
    let expected = expected_hits(candidates, bits);
    if expected <= MAX_EXPECTED_HITS_PER_LAUNCH {
        Ok(expected)
    } else {
        Err(PatternError::TooUnselective {
            expected_hits: expected,
        })
    }
}

/// Host model of the device's bit-255 fold: `low64(y) + 19·bit255`, `y` as LE limbs in `[0, 2^256)`.
pub fn fold_low64(y: &[u64; 4]) -> u64 {
    y[0].wrapping_add(19 * (y[3] >> 63))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::address;

    #[test]
    fn worked_vectors() {
        // Cross-checked against an independent implementation of the encoding.
        assert_eq!(compile("duck").unwrap(), (0x0000_0000_00F0_FFFF, 0x0000_0000_00A0_041D));
        assert_eq!(compile("a").unwrap(), (0x0000_0000_0000_00F8, 0x0000_0000_0000_0000));
        assert_eq!(compile("7").unwrap(), (0x0000_0000_0000_00F8, 0x0000_0000_0000_00F8));
        assert_eq!(compile("aaaaaaaaaaaa").unwrap(), (0xF0FF_FFFF_FFFF_FFFF, 0));
        assert_eq!(
            compile("777777777777").unwrap(),
            (0xF0FF_FFFF_FFFF_FFFF, 0xF0FF_FFFF_FFFF_FFFF)
        );
        assert_eq!(
            compile("examplename").unwrap(),
            (0x00FE_FFFF_FFFF_FFFF, 0x0008_038D_ACC7_C025)
        );
    }

    #[test]
    fn matches_duck_vector() {
        let pk = address::parse("duckduckgogg42xjoc72x3sjasowoarfbgcmvfimaftt6twagswzczad").unwrap();
        let (m, t) = compile("duck").unwrap();
        assert!(matches(m, t, &pk));
        // A prefix the address does not start with must not match.
        let (m2, t2) = compile("quack").unwrap();
        assert!(!matches(m2, t2, &pk));
    }

    #[test]
    fn validation() {
        assert_eq!(compile(""), Err(PatternError::Empty));
        assert_eq!(compile(&"a".repeat(13)), Err(PatternError::TooLong(13)));
        assert_eq!(compile("Abc"), Err(PatternError::BadChar(b'A')));
        for d in [b'0', b'1', b'8', b'9', b'.'] {
            let s = format!("ab{}", d as char);
            assert_eq!(compile(&s), Err(PatternError::BadChar(d)));
        }
        assert!(compile(&"a".repeat(12)).is_ok());
        assert!(compile("234567").is_ok());
        assert!(compile("ab").is_ok()); // trailing-bit case address::parse would reject is fine here
        assert_eq!(bits("duck"), 20);
        assert_eq!(bits("examplename"), 55);
    }

    #[test]
    fn mask_shape() {
        for l in 1..=12 {
            let (m, t) = compile(&"a".repeat(l)).unwrap();
            assert_eq!(m.count_ones(), 5 * l as u32, "len {l}");
            assert_eq!(t & !m, 0, "len {l}");
        }
    }

    // Deterministic xorshift64 so failures reproduce without a dependency.
    fn xorshift(state: &mut u64) -> u64 {
        let mut x = *state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *state = x;
        x
    }

    #[test]
    fn property_prefix_matches_and_one_char_breaks() {
        let mut st = 0x9e37_79b9_7f4a_7c15u64;
        for _ in 0..20_000 {
            let mut k = [0u8; 32];
            for chunk in k.chunks_mut(8) {
                chunk.copy_from_slice(&xorshift(&mut st).to_le_bytes());
            }
            let addr = address::encode(&k);
            let ab = addr.as_bytes();
            for l in 1..=12usize {
                let (m, t) = compile(&addr[..l]).unwrap();
                assert!(matches(m, t, &k), "prefix len {l} must match its own key");
                // mutate one char inside the prefix -> must break (its 5 bits are all in the mask)
                let i = (xorshift(&mut st) as usize) % l;
                let orig = ab[i];
                let repl = if orig == b'a' { b'b' } else { b'a' };
                let mut mutated = addr.clone().into_bytes();
                mutated[i] = repl;
                let mutated = String::from_utf8(mutated).unwrap();
                let (m2, t2) = compile(&mutated[..l]).unwrap();
                assert!(!matches(m2, t2, &k), "mutating char {i} of len {l} must break");
            }
        }
    }

    #[test]
    fn selectivity_guard() {
        let c = 1u64 << 30;
        assert_eq!(expected_hits(c, &[30]), 1.0);
        assert_eq!(check_selectivity(c, &[24]), Ok(64.0)); // boundary accepted
        assert_eq!(
            check_selectivity(c, &[23]),
            Err(PatternError::TooUnselective { expected_hits: 128.0 })
        );
        assert_eq!(check_selectivity(c, &[25, 25]), Ok(64.0));
        assert!(check_selectivity(c, &[24, 24]).is_err());
        // realistic launch
        let c2 = candidates_per_launch(43_008, 36, 256);
        assert_eq!(c2, 794_271_744); // 43008 · 36 · 513
        assert!(check_selectivity(c2, &[25, 25]).is_ok());
        assert!(check_selectivity(c2, &[24, 24]).is_err());
        assert!(check_selectivity(c2, &[26, 26, 26, 26]).is_ok());
        assert!(check_selectivity(c2, &[25, 25, 25, 25]).is_err());
    }

    // Reference `y mod p` on LE limbs, for the fold test only.
    const P: [u64; 4] = [
        0xFFFF_FFFF_FFFF_FFED,
        0xFFFF_FFFF_FFFF_FFFF,
        0xFFFF_FFFF_FFFF_FFFF,
        0x7FFF_FFFF_FFFF_FFFF,
    ];

    fn ge_p(y: &[u64; 4]) -> bool {
        for i in (0..4).rev() {
            if y[i] != P[i] {
                return y[i] > P[i];
            }
        }
        true
    }

    fn reduce(mut y: [u64; 4]) -> [u64; 4] {
        for _ in 0..2 {
            if ge_p(&y) {
                let mut borrow = 0u128;
                for i in 0..4 {
                    let v = (y[i] as u128)
                        .wrapping_sub(P[i] as u128)
                        .wrapping_sub(borrow);
                    y[i] = v as u64;
                    borrow = (v >> 64) & 1;
                }
            }
        }
        y
    }

    // Bad set (fold != y mod p): [p, 2^255) ∪ [2p, 2^256). 2p = 2^256-38.
    fn predicted_bad(y: &[u64; 4]) -> bool {
        let in_lo = (y[3] >> 63 == 0) && ge_p(y);
        let ge_2p =
            y[3] == !0 && y[2] == !0 && y[1] == !0 && y[0] >= 0xFFFF_FFFF_FFFF_FFDA;
        in_lo || ge_2p
    }

    #[test]
    fn fold_formula() {
        assert_eq!(fold_low64(&[5, 0, 0, 0]), 5);
        assert_eq!(fold_low64(&[5, 0, 0, 1 << 63]), 24); // +19 for bit 255
    }

    #[test]
    fn fold_mismatch_only_in_predicted_set() {
        // The fold equals (y mod p) low64 except inside the predicted sets.
        let check = |y: [u64; 4]| {
            let eq = fold_low64(&y) == reduce(y)[0];
            assert_eq!(!eq, predicted_bad(&y), "y = {y:016x?}");
        };
        // Random sweep (bad set is ~57/2^256, so eq holds and predicted_bad is false).
        let mut st = 0x1234_5678_9abc_def0u64;
        for _ in 0..20_000 {
            let mut y = [0u64; 4];
            for limb in y.iter_mut() {
                *limb = xorshift(&mut st);
            }
            check(y);
        }
        // Crafted edges around p, 2^255, 2p, 0.
        let p = P;
        let two255 = [0, 0, 0, 1u64 << 63];
        let two256m1 = [!0, !0, !0, !0];
        let two_p = [0xFFFF_FFFF_FFFF_FFDA, !0, !0, !0]; // 2^256-38
        for y in [p, two255, two256m1, two_p, [0; 4], [18, 0, 0, 0], [19, 0, 0, 0]] {
            check(y);
        }
        // neighbours of p
        check([P[0] - 1, P[1], P[2], P[3]]); // p-1: good
        check([P[0] + 1, P[1], P[2], P[3]]); // p+1: bad (in [p, 2^255))
        // 2p-1 is good, 2p is bad
        check([0xFFFF_FFFF_FFFF_FFD9, !0, !0, !0]); // 2p-1
    }
}
