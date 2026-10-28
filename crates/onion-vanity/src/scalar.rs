//! Clamped Ed25519 scalars, the `a0 + 8m` walk rebuild, and stack scrubbing.
//!
//! The secret is the raw clamped 32-byte scalar `a` (little-endian, **not** reduced mod l), exactly
//! as Tor stores it. `a0 + 8m` is computed as checked 256-bit integer arithmetic (not mod l); a
//! result that is no longer a valid clamped scalar is rejected, never re-clamped.
//! Secret-holding types are zeroized on drop, have a redacted `Debug`, and are never
//! `Clone`/`PartialEq`.

use curve25519_dalek::{edwards::EdwardsPoint, scalar::Scalar};
use zeroize::{Zeroize, ZeroizeOnDrop};

#[derive(Zeroize, ZeroizeOnDrop)]
pub struct ClampedScalar([u8; 32]);

#[derive(Zeroize, ZeroizeOnDrop)]
pub struct Nonce([u8; 32]);

impl core::fmt::Debug for ClampedScalar {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("ClampedScalar(..)")
    }
}

impl core::fmt::Debug for Nonce {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("Nonce(..)")
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum ScalarError {
    Overflow,
    Underflow,
    NotClamped,
}

/// The rule-1 clamp: `s[0] &= 248; s[31] &= 127; s[31] |= 64`.
pub fn clamp(mut s: [u8; 32]) -> [u8; 32] {
    s[0] &= 248;
    s[31] &= 127;
    s[31] |= 64;
    s
}

/// 32 CSPRNG bytes, or abort the process — there is no fallback RNG.
pub(crate) fn fill_random32() -> [u8; 32] {
    let mut b = [0u8; 32];
    if getrandom::fill(&mut b).is_err() {
        std::process::abort();
    }
    b
}

fn is_clamped(o: &[u8; 32]) -> bool {
    o[0] & 0b0000_0111 == 0 && o[31] & 0x40 != 0 && o[31] & 0x80 == 0
}

/// `a0 + 8·m` as checked 256-bit integer arithmetic over LE bytes, then a clamp check.
/// Never reduces mod l and never re-clamps: a non-clamped or out-of-range result is an error.
pub fn add_8m_bytes(a0: &[u8; 32], m: i64) -> Result<[u8; 32], ScalarError> {
    let mut l = [0u64; 4];
    for i in 0..4 {
        l[i] = u64::from_le_bytes(a0[8 * i..8 * i + 8].try_into().unwrap());
    }
    let delta = (m.unsigned_abs() as u128) << 3; // 8·|m|, max 2^66, fits u128
    let dd = [delta as u64, (delta >> 64) as u64, 0, 0];
    let mut carry = 0u64;
    let result = if m >= 0 {
        for i in 0..4 {
            let (x, c1) = l[i].overflowing_add(dd[i]);
            let (y, c2) = x.overflowing_add(carry);
            l[i] = y;
            carry = (c1 as u64) | (c2 as u64);
        }
        if carry != 0 {
            Err(ScalarError::Overflow)
        } else {
            Ok(())
        }
    } else {
        for i in 0..4 {
            let (x, b1) = l[i].overflowing_sub(dd[i]);
            let (y, b2) = x.overflowing_sub(carry);
            l[i] = y;
            carry = (b1 as u64) | (b2 as u64);
        }
        if carry != 0 {
            Err(ScalarError::Underflow)
        } else {
            Ok(())
        }
    };
    let mut o = [0u8; 32];
    for i in 0..4 {
        o[8 * i..8 * i + 8].copy_from_slice(&l[i].to_le_bytes());
    }
    l.zeroize();
    let out = result.and_then(|()| {
        if is_clamped(&o) {
            Ok(o)
        } else {
            Err(ScalarError::NotClamped)
        }
    });
    if out.is_err() {
        o.zeroize();
    }
    out
}

impl ClampedScalar {
    /// A fresh clamped scalar from the CSPRNG.
    pub fn random() -> ClampedScalar {
        ClampedScalar(clamp(fill_random32()))
    }

    /// Accept 32 bytes only if they are already a valid clamped scalar (used by verify/tests).
    pub fn from_bytes(b: [u8; 32]) -> Result<ClampedScalar, ScalarError> {
        if is_clamped(&b) {
            Ok(ClampedScalar(b))
        } else {
            Err(ScalarError::NotClamped)
        }
    }

    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// `A = a·B`, compressed to 32 bytes. Uses the unreduced-then-reduced `from_bytes_mod_order`
    /// (same point as Tor), zeroizing the temporary dalek scalar.
    pub fn public_compressed(&self) -> [u8; 32] {
        let mut s = Scalar::from_bytes_mod_order(self.0);
        let p = EdwardsPoint::mul_base(&s).compress().to_bytes();
        s.zeroize();
        p
    }

    /// `a0 + 8m` as a new clamped scalar, or an error if it is out of range / not clamped.
    pub fn add_8m(&self, m: i64) -> Result<ClampedScalar, ScalarError> {
        add_8m_bytes(&self.0, m).map(ClampedScalar)
    }
}

impl Nonce {
    pub fn random() -> Nonce {
        Nonce(fill_random32())
    }

    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Zeroize a 32 KB stack frame, then `black_box` it so the writes are not elided. Call after each
/// chunk of dalek scalar work, which leaves scalar-derived digits on the stack.
#[inline(never)]
pub fn scrub_stack() {
    let mut buf = [0u8; 32 * 1024];
    buf.zeroize();
    core::hint::black_box(&buf);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unhex32(s: &str) -> [u8; 32] {
        let mut o = [0u8; 32];
        for i in 0..32 {
            o[i] = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap();
        }
        o
    }

    // KAT: seed s[i] = (i*7+1) mod 256, a0 = clamp(seed), verified against dalek 4.1.3.
    fn seed() -> [u8; 32] {
        let mut s = [0u8; 32];
        for (i, b) in s.iter_mut().enumerate() {
            *b = (i as u8).wrapping_mul(7).wrapping_add(1);
        }
        s
    }
    const A0_HEX: &str = "00080f161d242b323940474e555c636a71787f868d949ba2a9b0b7bec5ccd35a";
    const A0_PUB_HEX: &str = "22a51202e4e28893e58c12a34f1c42655b53395bd38fb98ddc462abdb78e80a8";

    #[test]
    fn clamp_and_public_kat() {
        let a0 = clamp(seed());
        assert_eq!(a0, unhex32(A0_HEX));
        let cs = ClampedScalar::from_bytes(a0).unwrap();
        assert_eq!(cs.public_compressed(), unhex32(A0_PUB_HEX));
    }

    #[test]
    fn add_8m_kat() {
        let a0 = unhex32(A0_HEX);
        let cases: [(i64, &str, &str); 4] = [
            (1, "08080f161d242b323940474e555c636a71787f868d949ba2a9b0b7bec5ccd35a",
                "d269fb415a15cf1cf363534d165a00d56c1f1790b8622336d7992614a69dbe83"),
            (-1, "f8070f161d242b323940474e555c636a71787f868d949ba2a9b0b7bec5ccd35a",
                "0928fab3dd436df5ee1b90371331b20e6c7c731eace7c0d40b2757f18b0dd25c"),
            (1000, "40270f161d242b323940474e555c636a71787f868d949ba2a9b0b7bec5ccd35a",
                "498bbc241bf2490afd058ac425382b0f2ee3f067219b48403cc2c95dbb297b0e"),
            (-1000, "c0e80e161d242b323940474e555c636a71787f868d949ba2a9b0b7bec5ccd35a",
                "e4179f99faf9da5cc1d58b9116d88768239d5edb3643d2b5b82caeed3340bc32"),
        ];
        for (m, a_hex, pub_hex) in cases {
            let a = add_8m_bytes(&a0, m).unwrap();
            assert_eq!(a, unhex32(a_hex), "m={m} scalar");
            let cs = ClampedScalar::from_bytes(a).unwrap();
            assert_eq!(cs.public_compressed(), unhex32(pub_hex), "m={m} public");
        }
        // ClampedScalar::add_8m wraps add_8m_bytes.
        let cs = ClampedScalar::from_bytes(a0).unwrap();
        assert_eq!(cs.add_8m(1).unwrap().as_bytes(), &unhex32(cases[0].1));
    }

    #[test]
    fn add_8m_boundaries() {
        let mut hi = [0xffu8; 32];
        hi[0] = 0xf8;
        hi[31] = 0x7f; // 2^255 - 8, clamped
        assert!(add_8m_bytes(&hi, 0).is_ok());
        assert_eq!(add_8m_bytes(&hi, 1), Err(ScalarError::NotClamped)); // sets bit 255
        assert!(add_8m_bytes(&hi, i64::MIN).is_ok()); // subtraction stays in range

        let mut top = [0xffu8; 32];
        top[0] = 0xf8; // 2^256 - 8, bit 255 set
        assert_eq!(add_8m_bytes(&top, 0), Err(ScalarError::NotClamped));
        assert_eq!(add_8m_bytes(&top, 1), Err(ScalarError::Overflow));

        let mut lo = [0u8; 32];
        lo[31] = 0x40; // 2^254, clamped
        assert!(add_8m_bytes(&lo, 0).is_ok());
        assert_eq!(add_8m_bytes(&lo, -1), Err(ScalarError::NotClamped)); // clears bit 254

        let zero = [0u8; 32];
        assert_eq!(add_8m_bytes(&zero, -1), Err(ScalarError::Underflow));
        assert_eq!(add_8m_bytes(&zero, i64::MIN), Err(ScalarError::Underflow));
    }

    // Expected values derived independently with Python big ints: (a + 8*m) as 256-bit LE.
    #[test]
    fn add_8m_limb_carry_vectors() {
        // Limbs 0..2 all ones except limb 0 = ...f8, limb 3 = 0x40000000000000aa.
        let a = "f8ffffffffffffffffffffffffffffffffffffffffffffffaa00000000000040";
        // Limbs (0, 0, 5, 0x4000000000000000 + 0x1234): borrow must ripple through limbs 0 and 1.
        let b = "0000000000000000000000000000000005000000000000003412000000000040";
        let h = "f8ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f"; // 2^255 - 8
        let cases: [(&str, i64, &str); 7] = [
            // +8 carries out of limbs 0, 1, 2 into limb 3.
            (a, 1, "000000000000000000000000000000000000000000000000ab00000000000040"),
            // -8 needs no borrow.
            (a, -1, "f0ffffffffffffffffffffffffffffffffffffffffffffffaa00000000000040"),
            // -8 borrows across limbs 0 and 1.
            (b, -1, "f8ffffffffffffffffffffffffffffff04000000000000003412000000000040"),
            // 8m = 2^64: lands in limb 1, carries through limbs 1 and 2 into limb 3.
            (a, 1 << 61, "f8ffffffffffffff00000000000000000000000000000000ab00000000000040"),
            // 8m = -2^64: borrows from limb 2 through limb 1.
            (b, -(1 << 61), "0000000000000000ffffffffffffffff04000000000000003412000000000040"),
            // 8m = 2^66 - 8: high limb of 8m is 3.
            (a, i64::MAX, "f0ffffffffffffff03000000000000000000000000000000ab00000000000040"),
            // 8m = -2^66: high limb of 8m is 4.
            (h, i64::MIN, "f8fffffffffffffffbffffffffffffffffffffffffffffffffffffffffffff7f"),
        ];
        for (base, m, want) in cases {
            let base = unhex32(base);
            assert_eq!(add_8m_bytes(&base, m).unwrap(), unhex32(want), "m={m}");
            let cs = ClampedScalar::from_bytes(base).unwrap();
            assert_eq!(cs.add_8m(m).unwrap().as_bytes(), &unhex32(want), "m={m} wrapper");
        }
    }

    #[test]
    fn from_bytes_rejects_each_clamp_bit() {
        let base = unhex32(A0_HEX);
        assert!(ClampedScalar::from_bytes(base).is_ok());
        let mutations: [(&str, usize, u8, bool); 5] = [
            ("bit 0 set", 0, 0x01, true),
            ("bit 1 set", 0, 0x02, true),
            ("bit 2 set", 0, 0x04, true),
            ("bit 254 cleared", 31, 0x40, false),
            ("bit 255 set", 31, 0x80, true),
        ];
        for (name, idx, mask, set) in mutations {
            let mut b = base;
            if set {
                b[idx] |= mask;
            } else {
                b[idx] &= !mask;
            }
            assert_eq!(ClampedScalar::from_bytes(b).err(), Some(ScalarError::NotClamped), "{name}");
        }
    }

    #[test]
    fn debug_is_redacted() {
        assert_eq!(format!("{:?}", ClampedScalar::random()), "ClampedScalar(..)");
        assert_eq!(format!("{:?}", Nonce::random()), "Nonce(..)");
    }
}
