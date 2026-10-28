//! Compiles `kernel/onion.cu` to an `sm_120` + `compute_120` fatbin with nvcc. On Windows nvcc is
//! pointed at the MSVC `cl.exe` located via the `cc` crate's registry lookup.
//!
//! A committed fatbin (`kernel/onion.fatbin`) plus a srchash (`kernel/onion.fatbin.srchash`, a
//! SHA-256 over the kernel sources and the exact nvcc flags) let a machine with only the driver
//! build without nvcc: if nvcc is absent and the committed fatbin still matches the srchash, it is
//! used; if the sources or flags changed, the build fails with a clear message instead of linking a
//! stale kernel.

use std::env;
use std::path::PathBuf;
use std::process::Command;

use sha2::{Digest, Sha256};

// Semantic flags that affect the fatbin (not machine-specific -ccbin/-o/input paths).
const NVCC_FLAGS: &[&str] = &[
    "-gencode",
    "arch=compute_120,code=sm_120",
    "-gencode",
    "arch=compute_120,code=compute_120",
    "-fatbin",
    "-DOV_H=256",
];

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let kernel_dir = manifest.parent().unwrap().parent().unwrap().join("kernel");
    let src = kernel_dir.join("onion.cu");
    let committed = kernel_dir.join("onion.fatbin");
    let hashfile = kernel_dir.join("onion.fatbin.srchash");
    let out = PathBuf::from(env::var("OUT_DIR").unwrap()).join("onion.fatbin");

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=OV_FORCE_NO_NVCC");
    let srchash = srchash(&kernel_dir); // also emits rerun-if-changed for each source

    // The host takes H from the same -DOV_H the kernel is compiled with (ov_gpu::OV_H).
    let ov_h: u32 = NVCC_FLAGS
        .iter()
        .find_map(|f| f.strip_prefix("-DOV_H="))
        .and_then(|v| v.parse().ok())
        .expect("NVCC_FLAGS sets -DOV_H=<n>");
    let ov_h_rs = PathBuf::from(env::var("OUT_DIR").unwrap()).join("ov_h.rs");
    std::fs::write(&ov_h_rs, format!("pub const OV_H: u32 = {ov_h};\n")).expect("writing ov_h.rs");

    // Try nvcc; a missing nvcc OR one that cannot target sm_120 (too old) both fall back to the
    // committed fatbin as long as it still matches the sources/flags.
    let compiled = match find_nvcc() {
        Some(nvcc) => {
            let mut cmd = Command::new(nvcc);
            cmd.args(NVCC_FLAGS).arg(&src).arg("-o").arg(&out);
            cmd.args(["-Xptxas", "-v"]); // register/spill report only, not codegen: kept out of srchash
            #[cfg(windows)]
            if let Some(dir) = find_cl_dir() {
                cmd.arg("-ccbin").arg(dir);
            }
            match cmd.output() {
                Ok(o) if o.status.success() => {
                    let report = format!("{}{}", String::from_utf8_lossy(&o.stderr), String::from_utf8_lossy(&o.stdout));
                    warn_on_search_spills(&report);
                    true
                }
                other => {
                    if let Ok(o) = &other {
                        for l in String::from_utf8_lossy(&o.stderr).lines().take(5) {
                            println!("cargo:warning=nvcc: {l}");
                        }
                    }
                    println!("cargo:warning=nvcc could not build the kernel (missing or too old for sm_120); using the committed fatbin");
                    false
                }
            }
        }
        None => false,
    };

    if compiled {
        // Refresh the committed fatbin + srchash only when the sources/flags changed.
        let stale = std::fs::read_to_string(&hashfile)
            .map(|h| h.trim() != srchash)
            .unwrap_or(true);
        if stale || !committed.exists() {
            std::fs::copy(&out, &committed).expect("copying committed fatbin");
            std::fs::write(&hashfile, &srchash).expect("writing srchash");
        }
    } else {
        let stored = std::fs::read_to_string(&hashfile).ok();
        if stored.as_deref().map(str::trim) == Some(srchash.as_str()) && committed.exists() {
            std::fs::copy(&committed, &out).expect("copying committed fatbin to OUT_DIR");
        } else {
            panic!(
                "no usable nvcc and the committed kernel is stale (sources or flags changed since \
                 {}). Install CUDA (>= 12.8 for sm_120) to rebuild, or restore a matching fatbin. \
                 Expected srchash {srchash}.",
                committed.display()
            );
        }
    }
}

/// The search kernel must not spill: ptxas -v prints "<n> bytes stack frame, <s> bytes
/// spill stores, <l> bytes spill loads" under "Function properties for search". Warn, not fail, so
/// another ptxas version that spills still builds a working (slower) kernel.
fn warn_on_search_spills(ptxas: &str) {
    let mut in_search = false;
    for line in ptxas.lines() {
        if line.contains("Function properties for ") {
            in_search = line.trim_end().ends_with(" search");
        } else if in_search && line.contains("spill stores") {
            let n: Vec<u64> = line
                .split(|c: char| !c.is_ascii_digit())
                .filter_map(|s| s.parse().ok())
                .collect();
            if n.len() >= 3 && (n[1] > 0 || n[2] > 0) {
                println!("cargo:warning=search kernel spills registers ({}); expect lower throughput", line.trim());
            }
            in_search = false;
        }
    }
}

/// SHA-256 over every `*.cu`/`*.cuh` in `kernel/` (sorted) plus the nvcc flag list.
fn srchash(kernel_dir: &std::path::Path) -> String {
    let mut srcs: Vec<PathBuf> = std::fs::read_dir(kernel_dir)
        .expect("reading kernel dir")
        .flatten()
        .map(|e| e.path())
        .filter(|p| matches!(p.extension().and_then(|e| e.to_str()), Some("cu" | "cuh")))
        .collect();
    srcs.sort();

    let mut h = Sha256::new();
    for p in &srcs {
        println!("cargo:rerun-if-changed={}", p.display());
        h.update(p.file_name().unwrap().to_string_lossy().as_bytes());
        h.update([0]);
        h.update(std::fs::read(p).expect("reading kernel source"));
    }
    h.update(NVCC_FLAGS.join(" ").as_bytes());
    format!("{:x}", h.finalize())
}

/// nvcc from `CUDA_PATH`, else the default Windows toolkit location, else `nvcc` on PATH.
/// `OV_FORCE_NO_NVCC=1` forces the committed-fatbin fallback.
fn find_nvcc() -> Option<PathBuf> {
    if env::var_os("OV_FORCE_NO_NVCC").is_some() {
        return None;
    }
    let exe = if cfg!(windows) { "nvcc.exe" } else { "nvcc" };
    if let Ok(p) = env::var("CUDA_PATH") {
        let c = PathBuf::from(p).join("bin").join(exe);
        if c.exists() {
            return Some(c);
        }
    }
    #[cfg(windows)]
    {
        let root = PathBuf::from(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA");
        if let Ok(rd) = std::fs::read_dir(&root) {
            let mut versions: Vec<_> = rd.flatten().map(|e| e.path()).collect();
            versions.sort();
            if let Some(c) = versions.iter().rev().map(|v| v.join("bin").join(exe)).find(|c| c.exists()) {
                return Some(c);
            }
        }
    }
    #[cfg(unix)]
    {
        let c = PathBuf::from("/usr/local/cuda/bin/nvcc");
        if c.exists() {
            return Some(c);
        }
    }
    // Last resort: nvcc on PATH.
    Command::new(exe)
        .arg("--version")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|_| PathBuf::from(exe))
}

/// The directory holding MSVC `cl.exe`, for nvcc's `-ccbin`.
#[cfg(windows)]
fn find_cl_dir() -> Option<PathBuf> {
    let target = env::var("TARGET").unwrap();
    let tool = cc::windows_registry::find_tool(&target, "cl.exe")?;
    tool.path().parent().map(PathBuf::from)
}
