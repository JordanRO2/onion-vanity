//! Tor v3 hidden-service key files and their private on-disk write sequence.
//!
//! Writes `hs_ed25519_secret_key` (96 B), `hs_ed25519_public_key` (64 B) and `hostname` into a
//! private staging directory, then renames it to `<addr>.onion` with a **no-replace** primitive so
//! nothing is ever overwritten. On Windows the staging directory gets a
//! protected DACL with a single ACE for the current user; on Unix it is mode 0700 with 0600 files.
//! This module does no curve math or hashing: the caller supplies the already-derived bytes.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use zeroize::Zeroizing;

use crate::address::{self, AddressError};

pub const SECRET_FILE: &str = "hs_ed25519_secret_key";
pub const PUBLIC_FILE: &str = "hs_ed25519_public_key";
pub const HOSTNAME_FILE: &str = "hostname";

/// `b"== ed25519v1-secret: type0 =="` (29 bytes) padded to 32 with NULs.
pub const SECRET_TAG: [u8; 32] = tag(b"== ed25519v1-secret: type0 ==");
/// `b"== ed25519v1-public: type0 =="` (29 bytes) padded to 32 with NULs.
pub const PUBLIC_TAG: [u8; 32] = tag(b"== ed25519v1-public: type0 ==");

const fn tag(s: &[u8]) -> [u8; 32] {
    let mut out = [0u8; 32];
    let mut i = 0;
    while i < s.len() {
        out[i] = s[i];
        i += 1;
    }
    out
}

#[derive(Debug)]
pub enum KeyFileError {
    BadLength(usize),
    BadTag,
    BadHostname,
    Address(AddressError),
}

/// 96-byte secret key file: `SECRET_TAG || scalar || nonce`. The scalar is the raw clamped
/// little-endian 32 bytes (not reduced); the nonce is the hash-prefix half.
pub fn secret_key_bytes(scalar: &[u8; 32], nonce: &[u8; 32]) -> Zeroizing<[u8; 96]> {
    let mut out = Zeroizing::new([0u8; 96]);
    out[..32].copy_from_slice(&SECRET_TAG);
    out[32..64].copy_from_slice(scalar);
    out[64..96].copy_from_slice(nonce);
    out
}

/// 64-byte public key file: `PUBLIC_TAG || pubkey`.
pub fn public_key_bytes(pubkey: &[u8; 32]) -> [u8; 64] {
    let mut out = [0u8; 64];
    out[..32].copy_from_slice(&PUBLIC_TAG);
    out[32..].copy_from_slice(pubkey);
    out
}

/// `hostname` contents: the 56-char address + `.onion\n` (63 bytes).
pub fn hostname_string(pubkey: &[u8; 32]) -> String {
    format!("{}.onion\n", address::encode(pubkey))
}

/// Parse a 96-byte secret key file, returning `(scalar, nonce)` zeroized on drop.
pub fn parse_secret_key(
    data: &[u8],
) -> Result<(Zeroizing<[u8; 32]>, Zeroizing<[u8; 32]>), KeyFileError> {
    if data.len() != 96 {
        return Err(KeyFileError::BadLength(data.len()));
    }
    if data[..32] != SECRET_TAG {
        return Err(KeyFileError::BadTag);
    }
    let mut scalar = Zeroizing::new([0u8; 32]);
    let mut nonce = Zeroizing::new([0u8; 32]);
    scalar.copy_from_slice(&data[32..64]);
    nonce.copy_from_slice(&data[64..96]);
    Ok((scalar, nonce))
}

/// Parse a 64-byte public key file, returning the 32-byte public key.
pub fn parse_public_key(data: &[u8]) -> Result<[u8; 32], KeyFileError> {
    if data.len() != 64 {
        return Err(KeyFileError::BadLength(data.len()));
    }
    if data[..32] != PUBLIC_TAG {
        return Err(KeyFileError::BadTag);
    }
    let mut pk = [0u8; 32];
    pk.copy_from_slice(&data[32..]);
    Ok(pk)
}

/// Parse a `hostname` file: exactly `<56 chars>.onion\n`, returning the public key.
pub fn parse_hostname(s: &str) -> Result<[u8; 32], KeyFileError> {
    let body = s.strip_suffix(".onion\n").ok_or(KeyFileError::BadHostname)?;
    if body.len() != 56 {
        return Err(KeyFileError::BadHostname);
    }
    address::parse(body).map_err(KeyFileError::Address)
}

/// Refuse a parent that principals other than the user (and SYSTEM/Administrators on Windows) can
/// write to; propagates `NotFound` if it does not exist. No directory is created here.
pub fn check_parent(parent: &Path) -> io::Result<()> {
    let md = fs::metadata(parent)?;
    if !md.is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::NotADirectory,
            "parent is not a directory",
        ));
    }
    if platform::parent_writable_by_others(parent)? {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "parent directory is writable by other principals",
        ));
    }
    Ok(())
}

/// Create a missing `--out`, private to the user on Unix (0700 whatever the umask) so the tool never
/// refuses a directory it made itself; on Windows it inherits the parent's ACL, which `check_parent`
/// still vets.
pub fn create_out_dir(path: &Path) -> io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        fs::DirBuilder::new().recursive(true).mode(0o700).create(path)
    }
    #[cfg(windows)]
    {
        fs::create_dir_all(path)
    }
}

/// Startup check that keys can land in `parent`: it passes `check_parent`, and a private
/// staging dir can be created there and no-replace-renamed, the exact sequence `write_key_dir` uses.
/// Catches a hostile or read-only `--out`, a plain file, and filesystems that reject
/// `RENAME_NOREPLACE` (WSL `/mnt/c`, NFS) before any mining, not at the first found key.
pub fn preflight(parent: &Path) -> io::Result<()> {
    check_parent(parent)?;
    let mut rnd = [0u8; 8];
    getrandom::fill(&mut rnd).map_err(|_| io::Error::other("CSPRNG failure"))?;
    let id = u64::from_le_bytes(rnd);
    let staging = parent.join(format!(".probe.tmp-{id:016x}"));
    let target = parent.join(format!(".probe-{id:016x}"));
    platform::create_private_dir(&staging)?;
    match platform::rename_noreplace(&staging, &target) {
        Ok(()) => fs::remove_dir(&target),
        Err(e) => {
            let _ = fs::remove_dir(&staging);
            Err(e)
        }
    }
}

/// Directories named `.*.tmp-*` under `parent` (possible leftover staging from a hard kill). Never
/// deleted automatically: they may hold secrets.
pub fn leftover_staging(parent: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let Ok(rd) = fs::read_dir(parent) else {
        return out;
    };
    for entry in rd.flatten() {
        if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.starts_with('.') && name.contains(".tmp-") {
                out.push(entry.path());
            }
        }
    }
    out
}

struct Staging {
    path: PathBuf,
    armed: bool,
}

impl Drop for Staging {
    fn drop(&mut self) {
        if self.armed {
            let _ = fs::remove_dir_all(&self.path);
        }
    }
}

fn create_private_file(path: &Path) -> io::Result<fs::File> {
    let mut o = fs::OpenOptions::new();
    o.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        o.mode(0o600);
    }
    o.open(path)
}

fn write_private_file(path: &Path, data: &[u8]) -> io::Result<()> {
    use io::Write;
    let mut f = create_private_file(path)?;
    f.write_all(data)?;
    f.sync_all()
}

/// Write the three key files into a private staging dir, then no-replace-rename it to
/// `<addr>.onion`. Returns the final path. Fails with `AlreadyExists` if the target exists.
pub fn write_key_dir(
    parent: &Path,
    scalar: &[u8; 32],
    nonce: &[u8; 32],
    pubkey: &[u8; 32],
) -> io::Result<PathBuf> {
    check_parent(parent)?;
    let addr = address::encode(pubkey);

    let mut rnd = [0u8; 8];
    getrandom::fill(&mut rnd).map_err(|_| io::Error::other("CSPRNG failure"))?;
    let staging = parent.join(format!(".{addr}.tmp-{:016x}", u64::from_le_bytes(rnd)));

    platform::create_private_dir(&staging)?;
    let mut guard = Staging {
        path: staging.clone(),
        armed: true,
    };

    let sk = secret_key_bytes(scalar, nonce);
    write_private_file(&staging.join(SECRET_FILE), &sk[..])?;
    write_private_file(&staging.join(PUBLIC_FILE), &public_key_bytes(pubkey))?;
    write_private_file(&staging.join(HOSTNAME_FILE), hostname_string(pubkey).as_bytes())?;

    let final_path = parent.join(format!("{addr}.onion"));
    platform::rename_noreplace(&staging, &final_path)?;
    guard.armed = false;
    Ok(final_path)
}

#[cfg(unix)]
mod platform {
    use std::fs;
    use std::io;
    use std::path::Path;

    pub fn create_private_dir(path: &Path) -> io::Result<()> {
        use std::os::unix::fs::DirBuilderExt;
        fs::DirBuilder::new().mode(0o700).create(path)
    }

    pub fn parent_writable_by_others(parent: &Path) -> io::Result<bool> {
        use std::os::unix::fs::MetadataExt;
        Ok(fs::metadata(parent)?.mode() & 0o022 != 0)
    }

    #[cfg(target_os = "linux")]
    pub fn rename_noreplace(from: &Path, to: &Path) -> io::Result<()> {
        use std::ffi::CString;
        use std::os::unix::ffi::OsStrExt;
        let f = CString::new(from.as_os_str().as_bytes())?;
        let t = CString::new(to.as_os_str().as_bytes())?;
        let rc = unsafe {
            libc::renameat2(
                libc::AT_FDCWD,
                f.as_ptr(),
                libc::AT_FDCWD,
                t.as_ptr(),
                libc::RENAME_NOREPLACE,
            )
        };
        if rc != 0 {
            Err(io::Error::last_os_error())
        } else {
            Ok(())
        }
    }

    #[cfg(all(unix, not(target_os = "linux")))]
    pub fn rename_noreplace(from: &Path, to: &Path) -> io::Result<()> {
        // Not atomic (check-then-rename); non-Linux Unix is not a supported target.
        if to.exists() {
            return Err(io::Error::new(io::ErrorKind::AlreadyExists, "target exists"));
        }
        fs::rename(from, to)
    }
}

#[cfg(windows)]
mod platform {
    use std::ffi::c_void;
    use std::io;
    use std::os::windows::ffi::OsStrExt;
    use std::path::Path;
    use std::ptr;

    use windows_sys::Win32::Foundation::{CloseHandle, LocalFree, HANDLE};
    use windows_sys::Win32::Security::Authorization::{
        ConvertSidToStringSidW, ConvertStringSecurityDescriptorToSecurityDescriptorW,
        GetNamedSecurityInfoW, SE_FILE_OBJECT, SDDL_REVISION_1,
    };
    use windows_sys::Win32::Security::{
        EqualSid, GetTokenInformation, IsWellKnownSid, ACCESS_ALLOWED_ACE, ACL,
        DACL_SECURITY_INFORMATION, GetAce, PSID, SECURITY_ATTRIBUTES, TOKEN_QUERY, TOKEN_USER,
        TokenUser, WinBuiltinAdministratorsSid, WinLocalSystemSid,
    };
    use windows_sys::Win32::Storage::FileSystem::{CreateDirectoryW, MoveFileExW};
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, OpenProcessToken};

    const ACCESS_ALLOWED_ACE_TYPE: u8 = 0;
    const INHERIT_ONLY_ACE: u8 = 0x08;
    // Any write-ish right.
    const WRITE_MASK: u32 = 0x500D_0156;

    fn wide_path(p: &Path) -> Vec<u16> {
        p.as_os_str().encode_wide().chain(Some(0)).collect()
    }

    fn wide(s: &str) -> Vec<u16> {
        s.encode_utf16().chain(Some(0)).collect()
    }

    /// The process token's user SID bytes (8-aligned via a `u64` backing buffer).
    fn token_user_sid() -> io::Result<Vec<u64>> {
        unsafe {
            let mut token: HANDLE = ptr::null_mut();
            if OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &mut token) == 0 {
                return Err(io::Error::last_os_error());
            }
            let mut len = 0u32;
            // First call fails with ERROR_INSUFFICIENT_BUFFER and sets `len`.
            GetTokenInformation(token, TokenUser, ptr::null_mut(), 0, &mut len);
            let mut buf = vec![0u64; (len as usize).div_ceil(8).max(1)];
            let ok = GetTokenInformation(
                token,
                TokenUser,
                buf.as_mut_ptr() as *mut c_void,
                len,
                &mut len,
            );
            CloseHandle(token);
            if ok == 0 {
                return Err(io::Error::last_os_error());
            }
            Ok(buf)
        }
    }

    fn sid_ptr(buf: &[u64]) -> PSID {
        unsafe { (*(buf.as_ptr() as *const TOKEN_USER)).User.Sid }
    }

    fn sid_to_string(sid: PSID) -> io::Result<String> {
        unsafe {
            let mut out: *mut u16 = ptr::null_mut();
            if ConvertSidToStringSidW(sid, &mut out) == 0 {
                return Err(io::Error::last_os_error());
            }
            let mut n = 0usize;
            while *out.add(n) != 0 {
                n += 1;
            }
            let s = String::from_utf16_lossy(std::slice::from_raw_parts(out, n));
            LocalFree(out as *mut c_void);
            Ok(s)
        }
    }

    /// Create `path` with the SDDL `D:P(A;OICI;FA;;;<user SID>)`.
    pub fn create_private_dir(path: &Path) -> io::Result<()> {
        let buf = token_user_sid()?;
        let sddl = format!("D:P(A;OICI;FA;;;{})", sid_to_string(sid_ptr(&buf))?);
        create_dir_with_sddl(path, &sddl)
    }

    /// Create a directory with an explicit SDDL (also a test fixture for hostile parents).
    pub fn create_dir_with_sddl(path: &Path, sddl: &str) -> io::Result<()> {
        unsafe {
            let mut psd: *mut c_void = ptr::null_mut();
            if ConvertStringSecurityDescriptorToSecurityDescriptorW(
                wide(sddl).as_ptr(),
                SDDL_REVISION_1,
                &mut psd,
                ptr::null_mut(),
            ) == 0
            {
                return Err(io::Error::last_os_error());
            }
            let sa = SECURITY_ATTRIBUTES {
                nLength: size_of::<SECURITY_ATTRIBUTES>() as u32,
                lpSecurityDescriptor: psd,
                bInheritHandle: 0,
            };
            let ok = CreateDirectoryW(wide_path(path).as_ptr(), &sa);
            LocalFree(psd);
            if ok == 0 {
                return Err(io::Error::last_os_error());
            }
            Ok(())
        }
    }

    pub fn rename_noreplace(from: &Path, to: &Path) -> io::Result<()> {
        // MoveFileExW with flags 0: no MOVEFILE_REPLACE_EXISTING, so an existing target fails.
        let ok = unsafe { MoveFileExW(wide_path(from).as_ptr(), wide_path(to).as_ptr(), 0) };
        if ok == 0 {
            Err(io::Error::last_os_error())
        } else {
            Ok(())
        }
    }

    /// Scan `path`'s DACL; return true if any non-inherit-only allow ACE grants a write right to a
    /// principal other than the current user, SYSTEM or Administrators (NULL DACL counts as yes).
    pub fn parent_writable_by_others(path: &Path) -> io::Result<bool> {
        let buf = token_user_sid()?;
        let user = sid_ptr(&buf);
        with_dacl(path, |dacl| unsafe {
            if dacl.is_null() {
                return true; // NULL DACL = everyone full access
            }
            let count = (*dacl).AceCount;
            for i in 0..count as u32 {
                let mut ace: *mut c_void = ptr::null_mut();
                if GetAce(dacl, i, &mut ace) == 0 {
                    continue;
                }
                let hdr = ace as *const ACCESS_ALLOWED_ACE;
                let atype = (*hdr).Header.AceType;
                // Inherit-only ACEs do not apply to the directory itself.
                if (*hdr).Header.AceFlags & INHERIT_ONLY_ACE != 0 {
                    continue;
                }
                // Deny-family ACEs grant nothing; allow (0) and allow-callback (9) share the
                // {Header, Mask, SidStart} layout we parse. Anything else (object ACEs 5/13, or an
                // unknown type) we cannot safely read, so fail closed and refuse the parent.
                const DENY_TYPES: [u8; 4] = [1, 6, 10, 11];
                if DENY_TYPES.contains(&atype) {
                    continue;
                }
                if atype != ACCESS_ALLOWED_ACE_TYPE && atype != 9 {
                    return true;
                }
                if (*hdr).Mask & WRITE_MASK == 0 {
                    continue;
                }
                let sid = &(*hdr).SidStart as *const u32 as PSID;
                let is_self = EqualSid(sid, user) != 0;
                let is_system = IsWellKnownSid(sid, WinLocalSystemSid) != 0;
                let is_admins = IsWellKnownSid(sid, WinBuiltinAdministratorsSid) != 0;
                if !(is_self || is_system || is_admins) {
                    return true;
                }
            }
            false
        })
    }

    /// True iff `path`'s DACL has exactly one allow ACE and it is for the current user.
    #[cfg(test)]
    pub fn dacl_single_user_ace(path: &Path) -> io::Result<bool> {
        let buf = token_user_sid()?;
        let user = sid_ptr(&buf);
        with_dacl(path, |dacl| unsafe {
            if dacl.is_null() || (*dacl).AceCount != 1 {
                return false;
            }
            let mut ace: *mut c_void = ptr::null_mut();
            if GetAce(dacl, 0, &mut ace) == 0 {
                return false;
            }
            let hdr = ace as *const ACCESS_ALLOWED_ACE;
            (*hdr).Header.AceType == ACCESS_ALLOWED_ACE_TYPE
                && EqualSid(&(*hdr).SidStart as *const u32 as PSID, user) != 0
        })
    }

    fn with_dacl<T>(path: &Path, f: impl FnOnce(*const ACL) -> T) -> io::Result<T> {
        unsafe {
            let mut dacl: *mut ACL = ptr::null_mut();
            let mut psd: *mut c_void = ptr::null_mut();
            let rc = GetNamedSecurityInfoW(
                wide_path(path).as_ptr(),
                SE_FILE_OBJECT,
                DACL_SECURITY_INFORMATION,
                ptr::null_mut(),
                ptr::null_mut(),
                &mut dacl,
                ptr::null_mut(),
                &mut psd,
            );
            if rc != 0 {
                return Err(io::Error::from_raw_os_error(rc as i32));
            }
            let out = f(dacl);
            LocalFree(psd);
            Ok(out)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unique_dir(label: &str) -> PathBuf {
        let mut r = [0u8; 8];
        getrandom::fill(&mut r).unwrap();
        let d = std::env::temp_dir().join(format!("ov-test-{label}-{:016x}", u64::from_le_bytes(r)));
        // Private parent so check_parent passes.
        platform::create_private_dir(&d).unwrap();
        d
    }

    #[test]
    fn tag_bytes() {
        assert_eq!(&SECRET_TAG[..29], b"== ed25519v1-secret: type0 ==");
        assert_eq!(&SECRET_TAG[29..], &[0, 0, 0]);
        assert_eq!(&PUBLIC_TAG[..29], b"== ed25519v1-public: type0 ==");
        // Shared 13-byte prefix: == ed25519v1-  (byte 12 is 0x2d '-', not 0x20).
        assert_eq!(&SECRET_TAG[..13], &PUBLIC_TAG[..13]);
        assert_eq!(SECRET_TAG[12], 0x2d);
        assert_eq!(
            hex(&SECRET_TAG),
            "3d3d206564323535313976312d7365637265743a207479706530203d3d000000"
        );
    }

    fn hex(b: &[u8]) -> String {
        b.iter().map(|x| format!("{x:02x}")).collect()
    }

    #[test]
    fn file_bytes_and_parse() {
        let sk = secret_key_bytes(&[0x11; 32], &[0x22; 32]);
        assert_eq!(sk.len(), 96);
        assert_eq!(&sk[..29], b"== ed25519v1-secret: type0 ==");
        assert_eq!(&sk[32..64], &[0x11; 32]);
        assert_eq!(&sk[64..96], &[0x22; 32]);
        let (s, n) = parse_secret_key(&sk[..]).unwrap();
        assert_eq!(&s[..], &[0x11; 32]);
        assert_eq!(&n[..], &[0x22; 32]);
        assert!(matches!(parse_secret_key(&sk[..95]), Err(KeyFileError::BadLength(95))));
        let mut bad = *sk;
        bad[0] ^= 0xff;
        assert!(matches!(parse_secret_key(&bad[..]), Err(KeyFileError::BadTag)));

        let pk = public_key_bytes(&[0x33; 32]);
        assert_eq!(parse_public_key(&pk[..]).unwrap(), [0x33; 32]);
    }

    #[test]
    fn hostname_roundtrip() {
        let pk = address::parse("duckduckgogg42xjoc72x3sjasowoarfbgcmvfimaftt6twagswzczad").unwrap();
        let h = hostname_string(&pk);
        assert_eq!(h.len(), 63);
        assert!(h.ends_with(".onion\n"));
        assert_eq!(parse_hostname(&h).unwrap(), pk);
        assert!(parse_hostname(h.trim_end()).is_err()); // missing newline
        assert!(parse_hostname(&h.to_uppercase()).is_err());
    }

    #[test]
    fn write_and_rename_no_replace() {
        let parent = unique_dir("write");
        let pk = address::parse("duckduckgogg42xjoc72x3sjasowoarfbgcmvfimaftt6twagswzczad").unwrap();
        let path = write_key_dir(&parent, &[0x11; 32], &[0x22; 32], &pk).unwrap();
        assert!(path.ends_with("duckduckgogg42xjoc72x3sjasowoarfbgcmvfimaftt6twagswzczad.onion"));
        assert_eq!(fs::metadata(path.join(SECRET_FILE)).unwrap().len(), 96);
        assert_eq!(fs::metadata(path.join(PUBLIC_FILE)).unwrap().len(), 64);
        assert_eq!(fs::metadata(path.join(HOSTNAME_FILE)).unwrap().len(), 63);

        // Second write to the same address must fail (target exists) and leave no staging dir.
        let err = write_key_dir(&parent, &[0x11; 32], &[0x22; 32], &pk).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::AlreadyExists);
        assert!(leftover_staging(&parent).is_empty());

        fs::remove_dir_all(&parent).ok();
    }

    #[test]
    fn preflight_accepts_private_dir_and_leaves_nothing() {
        let parent = unique_dir("preflight");
        preflight(&parent).unwrap();
        assert_eq!(fs::read_dir(&parent).unwrap().count(), 0, "probe dirs must be removed");

        let file = parent.join("not-a-dir");
        fs::write(&file, b"x").unwrap();
        assert_eq!(preflight(&file).unwrap_err().kind(), io::ErrorKind::NotADirectory);
        fs::remove_dir_all(&parent).ok();
    }

    #[test]
    fn rename_noreplace_onto_empty_dir_fails() {
        let parent = unique_dir("noreplace");
        let (a, b) = (parent.join("a"), parent.join("b"));
        fs::create_dir(&a).unwrap();
        fs::create_dir(&b).unwrap();
        assert_eq!(platform::rename_noreplace(&a, &b).unwrap_err().kind(), io::ErrorKind::AlreadyExists);
        assert!(a.is_dir() && b.is_dir());
        fs::remove_dir_all(&parent).ok();
    }

    #[test]
    fn staging_guard_removes_only_when_armed() {
        let parent = unique_dir("guard");
        for armed in [true, false] {
            let d = parent.join(format!("s{armed}"));
            fs::create_dir(&d).unwrap();
            fs::write(d.join("f"), b"secret").unwrap();
            drop(Staging { path: d.clone(), armed });
            assert_eq!(d.exists(), !armed);
        }
        fs::remove_dir_all(&parent).ok();
    }

    #[test]
    fn leftover_reports_only_staging_dirs() {
        let parent = unique_dir("leftover");
        fs::create_dir(parent.join(".abc.tmp-1")).unwrap();
        fs::create_dir(parent.join("normal")).unwrap();
        fs::write(parent.join(".x.tmp-2"), b"not a dir").unwrap();
        let found = leftover_staging(&parent);
        assert_eq!(found.len(), 1);
        assert!(found[0].ends_with(".abc.tmp-1"));
        fs::remove_dir_all(&parent).ok();
    }

    #[cfg(unix)]
    #[test]
    fn unix_modes_and_parent_refusal() {
        use std::os::unix::fs::PermissionsExt;
        let parent = unique_dir("unix");
        let pk = address::parse("duckduckgogg42xjoc72x3sjasowoarfbgcmvfimaftt6twagswzczad").unwrap();
        let path = write_key_dir(&parent, &[0x11; 32], &[0x22; 32], &pk).unwrap();
        assert_eq!(fs::metadata(&path).unwrap().permissions().mode() & 0o777, 0o700);
        for f in [SECRET_FILE, PUBLIC_FILE, HOSTNAME_FILE] {
            assert_eq!(fs::metadata(path.join(f)).unwrap().permissions().mode() & 0o777, 0o600);
        }
        fs::set_permissions(&parent, fs::Permissions::from_mode(0o775)).unwrap();
        assert_eq!(check_parent(&parent).unwrap_err().kind(), io::ErrorKind::PermissionDenied);
        assert_eq!(preflight(&parent).unwrap_err().kind(), io::ErrorKind::PermissionDenied);
        fs::set_permissions(&parent, fs::Permissions::from_mode(0o700)).unwrap();
        fs::remove_dir_all(&parent).ok();
    }

    #[cfg(windows)]
    #[test]
    fn windows_dacl_and_parent_refusal() {
        let parent = unique_dir("win");
        let pk = address::parse("duckduckgogg42xjoc72x3sjasowoarfbgcmvfimaftt6twagswzczad").unwrap();
        let path = write_key_dir(&parent, &[0x11; 32], &[0x22; 32], &pk).unwrap();
        // The created key dir and each key file (by inheritance) have exactly one ACE, for the user.
        assert!(platform::dacl_single_user_ace(&path).unwrap());
        for f in [SECRET_FILE, PUBLIC_FILE, HOSTNAME_FILE] {
            assert!(platform::dacl_single_user_ace(&path.join(f)).unwrap(), "{f}");
        }

        // A parent anyone can write to is refused, by the check and by the startup preflight.
        let hostile = std::env::temp_dir().join(format!("ov-hostile-{}", std::process::id()));
        platform::create_dir_with_sddl(&hostile, "D:(A;OICI;FA;;;WD)").unwrap();
        assert_eq!(check_parent(&hostile).unwrap_err().kind(), io::ErrorKind::PermissionDenied);
        assert_eq!(preflight(&hostile).unwrap_err().kind(), io::ErrorKind::PermissionDenied);
        fs::remove_dir_all(&hostile).ok();
        fs::remove_dir_all(&parent).ok();
    }
}
