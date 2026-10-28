//! Tor v3 onion address: base32 of `pubkey || checksum || version`, where
//! `checksum = SHA3-256(".onion checksum" || pubkey || version)[..2]` and `version = 3`.
//! Hand-rolled base32 (RFC 4648 lowercase `a-z2-7`, no padding): no base32 crate.

use sha3::{Digest, Sha3_256};

pub const VERSION: u8 = 3;

const ALPHABET: &[u8; 32] = b"abcdefghijklmnopqrstuvwxyz234567";

#[derive(Debug, PartialEq, Eq)]
pub enum AddressError {
    BadLength(usize),
    BadChar(u8),
    TrailingBits,
    BadVersion(u8),
    BadChecksum,
}

/// The 56-character onion address (without the `.onion` suffix) for a public key.
pub fn encode(pubkey: &[u8; 32]) -> String {
    let mut data = [0u8; 35];
    data[..32].copy_from_slice(pubkey);
    let c = checksum(pubkey);
    data[32] = c[0];
    data[33] = c[1];
    data[34] = VERSION;
    b32_encode(&data)
}

/// Parse a 56-character address (an optional `.onion` suffix is accepted), verifying the
/// version and recomputing the SHA3 checksum. Returns the 32-byte public key.
pub fn parse(s: &str) -> Result<[u8; 32], AddressError> {
    let s = s.strip_suffix(".onion").unwrap_or(s);
    if s.len() != 56 {
        return Err(AddressError::BadLength(s.len()));
    }
    let raw = b32_decode(s)?; // exactly 35 bytes: 56 * 5 bits == 35 * 8 bits
    let mut pubkey = [0u8; 32];
    pubkey.copy_from_slice(&raw[..32]);
    let version = raw[34];
    if version != VERSION {
        return Err(AddressError::BadVersion(version));
    }
    if [raw[32], raw[33]] != checksum(&pubkey) {
        return Err(AddressError::BadChecksum);
    }
    Ok(pubkey)
}

fn checksum(pubkey: &[u8; 32]) -> [u8; 2] {
    let mut h = Sha3_256::new();
    h.update(b".onion checksum");
    h.update(pubkey);
    h.update([VERSION]);
    let d = h.finalize();
    [d[0], d[1]]
}

fn b32_encode(data: &[u8]) -> String {
    let mut out = String::with_capacity(data.len() * 8 / 5 + 1);
    let mut buf = 0u32;
    let mut bits = 0u32;
    for &b in data {
        buf = (buf << 8) | b as u32;
        bits += 8;
        while bits >= 5 {
            bits -= 5;
            out.push(ALPHABET[((buf >> bits) & 0x1f) as usize] as char);
        }
    }
    if bits > 0 {
        out.push(ALPHABET[((buf << (5 - bits)) & 0x1f) as usize] as char);
    }
    out
}

fn b32_decode(s: &str) -> Result<Vec<u8>, AddressError> {
    let mut out = Vec::with_capacity(s.len() * 5 / 8);
    let mut buf = 0u32;
    let mut bits = 0u32;
    for c in s.bytes() {
        let v = match c {
            b'a'..=b'z' => c - b'a',
            b'2'..=b'7' => c - b'2' + 26,
            _ => return Err(AddressError::BadChar(c)),
        } as u32;
        buf = (buf << 5) | v;
        bits += 5;
        if bits >= 8 {
            bits -= 8;
            out.push((buf >> bits) as u8);
        }
    }
    if bits > 0 && (buf & ((1 << bits) - 1)) != 0 {
        return Err(AddressError::TrailingBits);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const DUCK: &str = "duckduckgogg42xjoc72x3sjasowoarfbgcmvfimaftt6twagswzczad.onion";

    #[test]
    fn roundtrip_known_vector() {
        let pk = parse(DUCK).unwrap();
        assert_eq!(&pk[..4], &[0x1d, 0x04, 0xa1, 0xd0]); // test vector: pk 1d04a1d0…
        assert_eq!(checksum(&pk), [0x91, 0x64]); // …checksum 9164
        assert_eq!(encode(&pk), DUCK.strip_suffix(".onion").unwrap());
        assert_eq!(parse(&encode(&pk)).unwrap(), pk); // round trip without suffix
    }

    #[test]
    fn rejects_uppercase() {
        let bad = DUCK.strip_suffix(".onion").unwrap().to_uppercase();
        assert!(matches!(parse(&bad), Err(AddressError::BadChar(_))));
    }

    #[test]
    fn rejects_impossible_digits() {
        for d in ['0', '1', '8', '9'] {
            let mut s: Vec<char> = "a".repeat(56).chars().collect();
            s[0] = d;
            let s: String = s.into_iter().collect();
            assert!(matches!(parse(&s), Err(AddressError::BadChar(_))), "digit {d}");
        }
    }

    #[test]
    fn rejects_bad_length() {
        assert_eq!(parse(&"a".repeat(55)), Err(AddressError::BadLength(55)));
        assert_eq!(parse(&"a".repeat(57)), Err(AddressError::BadLength(57)));
    }

    #[test]
    fn rejects_bad_checksum() {
        let pk = parse(DUCK).unwrap();
        let mut data = [0u8; 35];
        data[..32].copy_from_slice(&pk);
        let c = checksum(&pk);
        data[32] = c[0] ^ 0xff; // corrupt only the checksum, keep version valid
        data[33] = c[1];
        data[34] = VERSION;
        assert_eq!(parse(&b32_encode(&data)), Err(AddressError::BadChecksum));
    }

    #[test]
    fn rejects_bad_version() {
        let pk = parse(DUCK).unwrap();
        let mut data = [0u8; 35];
        data[..32].copy_from_slice(&pk);
        let c = checksum(&pk);
        data[32] = c[0];
        data[33] = c[1];
        data[34] = 4; // wrong version; checked before the checksum
        assert_eq!(parse(&b32_encode(&data)), Err(AddressError::BadVersion(4)));
    }

    #[test]
    fn b32_decode_rejects_nonzero_trailing_bits() {
        // "ab" -> 0b00000 0b00001 = 10 bits; 1 byte out, 2 trailing bits == 0b01 != 0.
        assert_eq!(b32_decode("ab"), Err(AddressError::TrailingBits));
        // "aa" -> trailing bits are zero, so it decodes.
        assert!(b32_decode("aa").is_ok());
    }
}
