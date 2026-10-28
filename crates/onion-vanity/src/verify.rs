//! Independent verification of a hit before any key is written, and the
//! `verify <dir>` command. Every check uses an independent path from the kernel: the scalar is
//! re-multiplied **unclamped** (as Tor does), the clamp bits are checked explicitly on the raw
//! bytes, the address is re-derived (recomputing the SHA3 checksum), and the exact 64 bytes that
//! will be written are signed and verified.

use std::fs;
use std::path::Path;

use curve25519_dalek::{edwards::EdwardsPoint, scalar::Scalar};
use ed25519_dalek::hazmat::{raw_sign, ExpandedSecretKey};
use ed25519_dalek::{Sha512, VerifyingKey};
use zeroize::{Zeroize, Zeroizing};

use crate::address::{self, AddressError};
use crate::keyfile;
use crate::scalar::scrub_stack;

/// Domain-separated message for the self-signature check. Contains no secret material.
const SIGN_MSG: &[u8] = b"onion-vanity verify v1";

#[derive(Debug, PartialEq, Eq)]
pub enum VerifyError {
    PubkeyMismatch,
    /// The raw scalar failed the clamp check at this bit index (0, 1, 2, 254 or 255).
    BadClamp(u16),
    NoPatternMatch,
    Address(AddressError),
    AddressMismatch,
    SignFailed,
    BadKeyFile(&'static str),
    Io(String),
}

#[derive(Debug, PartialEq, Eq)]
pub struct Verified {
    pub pubkey: [u8; 32],
    /// 56-char address (no `.onion`).
    pub address: String,
    /// Index of the longest matching pattern; `None` only when `patterns` is empty.
    pub credited: Option<usize>,
}

/// Steps 1–4 on the exact 64 bytes (`scalar || nonce`) that will be written. `patterns` are
/// lowercase `a-z2-7` prefixes; an empty slice skips step 2 (used by `verify <dir>` without `-p`).
pub fn verify_key(
    secret: &[u8; 64],
    pubkey: &[u8; 32],
    patterns: &[&str],
) -> Result<Verified, VerifyError> {
    let mut raw = [0u8; 32];
    raw.copy_from_slice(&secret[..32]);
    let mut nonce = [0u8; 32];
    nonce.copy_from_slice(&secret[32..]);
    let mut s = Scalar::from_bytes_mod_order(raw);

    let result = verify_steps(&s, &raw, &nonce, pubkey, patterns);

    // Single cleanup on every path. `Scalar` is `Copy`, so the one moved into `esk` was a copy;
    // this `s` is the live secret-equivalent scalar and must be zeroized here.
    s.zeroize();
    raw.zeroize();
    nonce.zeroize();
    scrub_stack();
    result
}

fn verify_steps(
    s: &Scalar,
    raw: &[u8; 32],
    nonce: &[u8; 32],
    pubkey: &[u8; 32],
    patterns: &[&str],
) -> Result<Verified, VerifyError> {
    // Step 1: unclamped re-multiply, then explicit clamp-bit check on the raw bytes.
    let a = EdwardsPoint::mul_base(s).compress().to_bytes();
    if a != *pubkey {
        return Err(VerifyError::PubkeyMismatch);
    }
    check_clamp(raw)?;

    let addr = address::encode(pubkey);

    // Step 2: independent host matcher, credited to the longest pattern.
    let credited = longest_match(patterns, &addr);
    if !patterns.is_empty() && credited.is_none() {
        return Err(VerifyError::NoPatternMatch);
    }

    // Step 3: re-derive the address and parse it back (recomputes the checksum).
    match address::parse(&addr) {
        Ok(pk) if pk == *pubkey => {}
        Ok(_) => return Err(VerifyError::AddressMismatch),
        Err(e) => return Err(VerifyError::Address(e)),
    }

    // Step 4: sign the exact bytes with a key built from the stored pubkey, then verify.
    let vk = VerifyingKey::from_bytes(pubkey).map_err(|_| VerifyError::PubkeyMismatch)?;
    let esk = ExpandedSecretKey {
        scalar: *s,
        hash_prefix: *nonce,
    };
    let sig = raw_sign::<Sha512>(&esk, SIGN_MSG, &vk);
    let ok = vk.verify_strict(SIGN_MSG, &sig).is_ok();
    drop(esk); // zeroizes esk's scalar + hash_prefix copies
    if !ok {
        return Err(VerifyError::SignFailed);
    }

    Ok(Verified {
        pubkey: *pubkey,
        address: addr,
        credited,
    })
}

/// A fresh 32-byte nonce for the key file. Aborts on RNG failure — no fallback.
pub fn draw_nonce() -> Zeroizing<[u8; 32]> {
    let mut b = Zeroizing::new([0u8; 32]);
    if getrandom::fill(&mut b[..]).is_err() {
        std::process::abort();
    }
    b
}

/// Run rule-5 verification on a `<addr>.onion` directory's key files.
pub fn verify_dir(dir: &Path, patterns: &[&str]) -> Result<Verified, VerifyError> {
    let read = |name: &str| -> Result<Vec<u8>, VerifyError> {
        fs::read(dir.join(name)).map_err(|e| VerifyError::Io(e.to_string()))
    };

    let pubkey = keyfile::parse_public_key(&read(keyfile::PUBLIC_FILE)?)
        .map_err(|_| VerifyError::BadKeyFile("public key"))?;

    let host = String::from_utf8(read(keyfile::HOSTNAME_FILE)?)
        .map_err(|_| VerifyError::BadKeyFile("hostname"))?;
    let host_pk =
        keyfile::parse_hostname(&host).map_err(|_| VerifyError::BadKeyFile("hostname"))?;
    if host_pk != pubkey {
        return Err(VerifyError::AddressMismatch);
    }

    let secret_data = Zeroizing::new(read(keyfile::SECRET_FILE)?);
    let (scalar, nonce) =
        keyfile::parse_secret_key(&secret_data).map_err(|_| VerifyError::BadKeyFile("secret key"))?;
    let mut secret = Zeroizing::new([0u8; 64]);
    secret[..32].copy_from_slice(&scalar[..]);
    secret[32..].copy_from_slice(&nonce[..]);

    verify_key(&secret, &pubkey, patterns)
}

fn check_clamp(raw: &[u8; 32]) -> Result<(), VerifyError> {
    if raw[0] & 1 != 0 {
        return Err(VerifyError::BadClamp(0));
    }
    if raw[0] & 2 != 0 {
        return Err(VerifyError::BadClamp(1));
    }
    if raw[0] & 4 != 0 {
        return Err(VerifyError::BadClamp(2));
    }
    if raw[31] & 0x40 == 0 {
        return Err(VerifyError::BadClamp(254));
    }
    if raw[31] & 0x80 != 0 {
        return Err(VerifyError::BadClamp(255));
    }
    Ok(())
}

/// Index of the longest pattern that `addr` starts with; ties go to the lowest index.
fn longest_match(patterns: &[&str], addr: &str) -> Option<usize> {
    let mut best: Option<usize> = None;
    for (i, p) in patterns.iter().enumerate() {
        if addr.starts_with(p) && best.is_none_or(|b| p.len() > patterns[b].len()) {
            best = Some(i);
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gen_key() -> ([u8; 64], [u8; 32]) {
        let mut raw = [0u8; 32];
        getrandom::fill(&mut raw).unwrap();
        raw = crate::scalar::clamp(raw);
        let pubkey = EdwardsPoint::mul_base(&Scalar::from_bytes_mod_order(raw))
            .compress()
            .to_bytes();
        let mut secret = [0u8; 64];
        secret[..32].copy_from_slice(&raw);
        getrandom::fill(&mut secret[32..]).unwrap();
        (secret, pubkey)
    }

    #[test]
    fn valid_key_passes() {
        let (secret, pubkey) = gen_key();
        let v = verify_key(&secret, &pubkey, &[]).unwrap();
        assert_eq!(v.pubkey, pubkey);
        assert_eq!(v.credited, None);
    }

    #[test]
    fn clamp_bit_flips_rejected() {
        let (secret, _) = gen_key();
        let mut nonce = [0u8; 32];
        nonce.copy_from_slice(&secret[32..]);
        for (idx, byte, mask) in [(0u16, 0, 1u8), (1, 0, 2), (2, 0, 4), (254, 31, 0x40), (255, 31, 0x80)] {
            let mut raw = [0u8; 32];
            raw.copy_from_slice(&secret[..32]);
            raw[byte] ^= mask;
            // recompute the pubkey from the flipped raw so only the clamp check can reject
            let pubkey = EdwardsPoint::mul_base(&Scalar::from_bytes_mod_order(raw))
                .compress()
                .to_bytes();
            let mut s = [0u8; 64];
            s[..32].copy_from_slice(&raw);
            s[32..].copy_from_slice(&nonce);
            assert_eq!(verify_key(&s, &pubkey, &[]), Err(VerifyError::BadClamp(idx)));
        }
    }

    #[test]
    fn pubkey_mismatch_rejected() {
        let (secret, mut pubkey) = gen_key();
        pubkey[0] ^= 1;
        assert_eq!(verify_key(&secret, &pubkey, &[]), Err(VerifyError::PubkeyMismatch));
    }

    #[test]
    fn pattern_credit() {
        assert_eq!(longest_match(&["ab", "abcd", "abc"], "abcdef"), Some(1));
        assert_eq!(longest_match(&["ab", "ab"], "abcdef"), Some(0));
        assert_eq!(longest_match(&["zz"], "abcdef"), None);
        assert_eq!(longest_match(&[], "abcdef"), None);

        let (secret, pubkey) = gen_key();
        let addr = address::encode(&pubkey);
        let p2 = addr[..2].to_string();
        let p4 = addr[..4].to_string();
        let v = verify_key(&secret, &pubkey, &[&p2, &p4]).unwrap();
        assert_eq!(v.credited, Some(1)); // longer prefix wins
        assert_eq!(verify_key(&secret, &pubkey, &["zzzz"]), Err(VerifyError::NoPatternMatch));
    }

    #[test]
    fn verify_dir_roundtrip() {
        let mut r = [0u8; 8];
        getrandom::fill(&mut r).unwrap();
        let parent = std::env::temp_dir().join(format!("ov-verify-{:016x}", u64::from_le_bytes(r)));
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt;
            fs::DirBuilder::new().mode(0o700).create(&parent).unwrap();
        }
        #[cfg(windows)]
        fs::create_dir(&parent).unwrap();

        let (secret, pubkey) = gen_key();
        let scalar: [u8; 32] = secret[..32].try_into().unwrap();
        let nonce: [u8; 32] = secret[32..].try_into().unwrap();
        let dir = keyfile::write_key_dir(&parent, &scalar, &nonce, &pubkey).unwrap();

        let v = verify_dir(&dir, &[]).unwrap();
        assert_eq!(v.pubkey, pubkey);

        // Corrupt the public key file -> verification fails.
        fs::write(dir.join(keyfile::PUBLIC_FILE), keyfile::public_key_bytes(&[0u8; 32])).unwrap();
        assert!(verify_dir(&dir, &[]).is_err());

        fs::remove_dir_all(&parent).ok();
    }

    #[test]
    fn verify_dir_rejects_each_tampered_file() {
        let mut r = [0u8; 8];
        getrandom::fill(&mut r).unwrap();
        let parent = std::env::temp_dir().join(format!("ov-verify-{:016x}", u64::from_le_bytes(r)));
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt;
            fs::DirBuilder::new().mode(0o700).create(&parent).unwrap();
        }
        #[cfg(windows)]
        fs::create_dir(&parent).unwrap();

        let (secret, pubkey) = gen_key();
        let scalar: [u8; 32] = secret[..32].try_into().unwrap();
        let nonce: [u8; 32] = secret[32..].try_into().unwrap();
        let dir = keyfile::write_key_dir(&parent, &scalar, &nonce, &pubkey).unwrap();

        let sk = dir.join(keyfile::SECRET_FILE);
        let pk = dir.join(keyfile::PUBLIC_FILE);
        let host = dir.join(keyfile::HOSTNAME_FILE);
        let orig_sk = fs::read(&sk).unwrap();
        let orig_pk = fs::read(&pk).unwrap();
        let orig_host = fs::read(&host).unwrap();
        let restore = || {
            fs::write(&sk, &orig_sk).unwrap();
            fs::write(&pk, &orig_pk).unwrap();
            fs::write(&host, &orig_host).unwrap();
        };
        assert_eq!(verify_dir(&dir, &[]).unwrap().pubkey, pubkey);

        // 1. A different valid key's secret file: pubkey/hostname untouched -> PubkeyMismatch.
        let (secret2, _) = gen_key();
        let scalar2: [u8; 32] = secret2[..32].try_into().unwrap();
        let nonce2: [u8; 32] = secret2[32..].try_into().unwrap();
        fs::write(&sk, &keyfile::secret_key_bytes(&scalar2, &nonce2)[..]).unwrap();
        assert_eq!(verify_dir(&dir, &[]), Err(VerifyError::PubkeyMismatch));
        restore();

        // 2. Secret file truncated to 95 bytes, and with tag byte 0 flipped.
        fs::write(&sk, &orig_sk[..95]).unwrap();
        assert_eq!(verify_dir(&dir, &[]), Err(VerifyError::BadKeyFile("secret key")));
        restore();
        let mut bad_tag = orig_sk.clone();
        bad_tag[0] ^= 1;
        fs::write(&sk, &bad_tag).unwrap();
        assert_eq!(verify_dir(&dir, &[]), Err(VerifyError::BadKeyFile("secret key")));
        restore();

        // 3. Public file truncated.
        fs::write(&pk, &orig_pk[..63]).unwrap();
        assert_eq!(verify_dir(&dir, &[]), Err(VerifyError::BadKeyFile("public key")));
        restore();

        // 4. Zeroed public key: well-formed file, but the hostname no longer matches it.
        fs::write(&pk, keyfile::public_key_bytes(&[0u8; 32])).unwrap();
        assert_eq!(verify_dir(&dir, &[]), Err(VerifyError::AddressMismatch));
        restore();

        // 5. hostname removed -> I/O error (message is OS-specific, so match the variant only).
        fs::remove_file(&host).unwrap();
        assert!(matches!(verify_dir(&dir, &[]), Err(VerifyError::Io(_))));
        restore();

        // 6. Patterns that cannot match on the intact dir.
        assert_eq!(verify_dir(&dir, &["zzzz"]), Err(VerifyError::NoPatternMatch));

        fs::remove_dir_all(&parent).ok();
    }
}
