//! `onion-vanity` CLI: GPU `search`, its `run` supervisor, `verify`, `bench`, `gpu-info`.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use curve25519_dalek::{edwards::EdwardsPoint, scalar::Scalar};
use zeroize::{Zeroize, Zeroizing};

use onion_vanity::scalar::{self, ClampedScalar};
use onion_vanity::{address, cpu, keyfile, pattern, verify};

const EXAMPLES: &str = "\
Examples:
  # One address starting with \"hello\" (`run` supervises long searches)
  onion-vanity run -p hello --out $HOME/onion-keys
  # Three \"shadow\" keys and one \"night7\" key in the same pass
  onion-vanity search -p shadow=3 -p night7 --out $HOME/onion-keys
  # Re-check a generated key, then measure the search rate for 60 s
  onion-vanity verify $HOME/onion-keys/<address>.onion -p hello
  onion-vanity bench --secs 60

Patterns are address prefixes over a-z and 2-7 (no 0, 1, 8, 9), 1-12 characters, 5+ on the GPU.
Each extra character makes a match 32x rarer; the status line shows each pattern's expected time.
Each key is written to <out>/<address>.onion/ as hostname, hs_ed25519_public_key and
hs_ed25519_secret_key, the files Tor reads from a HiddenServiceDir.";

#[derive(Parser)]
#[command(
    name = "onion-vanity",
    about = "GPU vanity address generator for Tor v3 onion services",
    after_help = EXAMPLES
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Search for addresses matching one or more prefix patterns.
    ///
    /// Exits once every pattern has its keys (keys already in --out count); Ctrl-C stops it.
    Search {
        /// Use the slow CPU reference walker instead of the GPU (testing, or prefixes under 5
        /// characters; keys are not RAM-locked).
        #[arg(long)]
        cpu: bool,
        /// Address prefix (a-z, 2-7; 5+ characters on the GPU), optionally with a key count:
        /// `-p hello` or `-p hello=3`. Repeatable; up to 15 patterns on the GPU.
        #[arg(short = 'p', long = "pattern", required = true, value_name = "PREFIX[=N]")]
        patterns: Vec<String>,
        /// Output directory, writable only by you (created if missing); each key is written to
        /// `<out>/<address>.onion/`.
        #[arg(long, value_name = "DIR")]
        out: PathBuf,
        /// Half-width of the per-draw walk (CPU mode).
        #[arg(long, default_value_t = 256)]
        h: i64,
        /// Cap on batches per launch; lower means shorter GPU launches (`run` halves it after a
        /// GPU fault).
        #[arg(long)]
        nb_max: Option<i32>,
        /// Proceed even if the secret bases cannot be locked into RAM (may touch swap).
        #[arg(long)]
        allow_unlocked: bool,
    },
    /// Re-verify a generated `<address>.onion` directory's key files.
    Verify {
        /// A `<address>.onion` directory written by `search` or `run`.
        dir: PathBuf,
        /// Also require the address to start with this prefix (repeatable; any one may match).
        #[arg(short = 'p', long = "pattern", value_name = "PREFIX")]
        patterns: Vec<String>,
    },
    /// Print GPU info and run a quick kernel self-test.
    GpuInfo,
    /// Benchmark the search kernel: measured candidates/s with a hit-based cross-check.
    Bench {
        /// Measurement duration in seconds (300 gives a stable figure; the default is a quick look).
        #[arg(long, default_value_t = 20)]
        secs: u64,
    },
    /// Supervise a GPU `search`, restarting it after a recoverable GPU fault (TDR); use it for
    /// long searches.
    Run {
        /// Address prefix, as for `search`: `-p hello` or `-p hello=3`. Repeatable.
        #[arg(short = 'p', long = "pattern", required = true, value_name = "PREFIX[=N]")]
        patterns: Vec<String>,
        /// Output directory, as for `search`.
        #[arg(long, value_name = "DIR")]
        out: PathBuf,
    },
}

struct Target {
    text: String,
    mask: u64,
    target: u64,
    remaining: u32,
}

fn parse_spec(s: &str) -> Result<(String, u32)> {
    match s.split_once('=') {
        Some((p, n)) => {
            let cap: u32 = n.parse().context("pattern cap must be a number")?;
            if cap == 0 {
                bail!("pattern cap must be >= 1");
            }
            Ok((p.to_string(), cap))
        }
        None => Ok((s.to_string(), 1)),
    }
}

/// Parse `-p PATTERN[=N]` specs. A repeated pattern is refused: its second cap could never be met,
/// so the run would mine forever.
fn parse_targets(specs: &[String]) -> Result<Vec<Target>> {
    let mut targets: Vec<Target> = Vec::new();
    for s in specs {
        let (text, cap) = parse_spec(s)?;
        if targets.iter().any(|t| t.text == text) {
            bail!("pattern {text:?} is given twice; use -p {text}=N to ask for N keys");
        }
        let (mask, target) = pattern::compile(&text)
            .map_err(|e| anyhow::anyhow!("invalid pattern {text:?}: {e:?}"))?;
        targets.push(Target { text, mask, target, remaining: cap });
    }
    Ok(targets)
}

/// Create `--out` if needed and prove a key can be written there before mining starts, so
/// a bad directory fails now instead of losing the first key found hours later.
fn prepare_out(out: &Path) -> Result<()> {
    if !out.exists() {
        keyfile::create_out_dir(out).with_context(|| format!("creating {}", out.display()))?;
    }
    keyfile::preflight(out).with_context(|| {
        format!(
            "--out {} cannot hold keys: it must be a directory only you can write to (e.g. under your user profile), on a filesystem with no-replace rename (not WSL /mnt/c or NFS)",
            out.display()
        )
    })?;
    for d in keyfile::leftover_staging(out) {
        eprintln!("warning: leftover staging dir (may hold secrets): {}", d.display());
    }
    Ok(())
}

/// A verified key is never dropped on a write error: it stays in memory, mining pauses, and
/// the write is retried every `retry` until it lands or the user interrupts. `AlreadyExists` means
/// this address's directory is already on disk. A Ctrl-C that came before the first failure only
/// stops mining once the key is written; one pressed during the hold discards the key.
fn write_or_hold(
    out: &Path,
    addr: &str,
    running: &AtomicBool,
    retry: Duration,
    mut write: impl FnMut() -> std::io::Result<PathBuf>,
) -> Result<PathBuf> {
    let mut first_failure = true;
    let mut stop_after = false;
    let landed = loop {
        match write() {
            Ok(p) => break p,
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                eprintln!("warning: {addr}.onion already exists in {}; keeping that directory", out.display());
                break out.join(format!("{addr}.onion"));
            }
            Err(e) => {
                if std::mem::take(&mut first_failure) && !running.load(Ordering::Relaxed) {
                    stop_after = true;
                    running.store(true, Ordering::Relaxed);
                    eprintln!("(the earlier Ctrl-C will stop mining once this key is written)");
                }
                eprintln!(
                    "ERROR: found {addr}.onion but could not write it to {} ({e}). The key is held in memory and mining is paused; fix the directory and the write is retried every {}s. Ctrl-C now discards the key.",
                    out.display(),
                    retry.as_secs()
                );
                let until = Instant::now() + retry;
                while Instant::now() < until {
                    if !running.load(Ordering::Relaxed) {
                        bail!("interrupted: the key for {addr}.onion was never written and is discarded");
                    }
                    std::thread::sleep(Duration::from_millis(20));
                }
                keyfile::create_out_dir(out).ok(); // recreate a removed --out; the write re-checks it
            }
        }
    };
    if stop_after {
        running.store(false, Ordering::Relaxed);
    }
    Ok(landed)
}

/// Longest `targets` entry whose text is a prefix of `addr`; ties go to the lowest index.
fn longest_match(targets: &[Target], addr: &str) -> Option<usize> {
    let mut best: Option<usize> = None;
    for (i, t) in targets.iter().enumerate() {
        if addr.starts_with(&t.text) && best.is_none_or(|b| t.text.len() > targets[b].text.len()) {
            best = Some(i);
        }
    }
    best
}

/// Credit already-present `<addr>.onion` dirs against the caps so a restart does not overshoot.
fn credit_existing(out: &Path, targets: &mut [Target]) {
    let Ok(rd) = std::fs::read_dir(out) else {
        return;
    };
    for entry in rd.flatten() {
        let name = entry.file_name();
        if !name.to_string_lossy().ends_with(".onion") || !entry.path().is_dir() {
            continue;
        }
        let Ok(host) = std::fs::read_to_string(entry.path().join(keyfile::HOSTNAME_FILE)) else {
            continue;
        };
        let Ok(pk) = keyfile::parse_hostname(&host) else {
            continue;
        };
        if let Some(i) = longest_match(targets, &address::encode(&pk)) {
            targets[i].remaining = targets[i].remaining.saturating_sub(1);
        }
    }
}

fn run_search(use_cpu: bool, specs: Vec<String>, out: PathBuf, h: i64, nb_max: Option<i32>, allow_unlocked: bool) -> Result<()> {
    if !use_cpu {
        return run_search_gpu(specs, out, nb_max, allow_unlocked);
    }
    if h < 1 {
        bail!("--h must be >= 1");
    }
    eprintln!("warning: --cpu is the slow reference path; its key memory is not RAM-locked or excluded from crash reports");

    let mut targets = parse_targets(&specs)?;
    prepare_out(&out)?;
    credit_existing(&out, &mut targets);

    let running = Arc::new(AtomicBool::new(true));
    {
        let r = running.clone();
        ctrlc::set_handler(move || r.store(false, Ordering::Relaxed))
            .context("installing Ctrl-C handler")?;
    }

    let mut failures = 0u32;
    while running.load(Ordering::Relaxed) {
        let active: Vec<(u64, u64)> = targets
            .iter()
            .filter(|t| t.remaining > 0)
            .map(|t| (t.mask, t.target))
            .collect();
        if active.is_empty() {
            break;
        }

        let Some(hit) = cpu::search(&active, h, &running) else {
            break; // interrupted
        };

        let nonce = verify::draw_nonce();
        let mut secret = Zeroizing::new([0u8; 64]);
        secret[..32].copy_from_slice(hit.scalar.as_bytes());
        secret[32..].copy_from_slice(&nonce[..]);

        let texts: Vec<&str> = targets.iter().map(|t| t.text.as_str()).collect();
        let verified = match verify::verify_key(&secret, &hit.pubkey, &texts) {
            Ok(v) => v,
            Err(e) => {
                failures += 1;
                eprintln!("verification failed ({e:?}); {failures}/3");
                if failures >= 3 {
                    bail!("aborting after 3 verification failures");
                }
                continue;
            }
        };

        let idx = verified.credited.expect("an active pattern matched");
        if targets[idx].remaining == 0 {
            continue; // longest match is a full pattern: count only, no write
        }
        let path = write_or_hold(&out, &verified.address, &running, Duration::from_secs(30), || {
            keyfile::write_key_dir(&out, hit.scalar.as_bytes(), &nonce, &hit.pubkey)
        })?;
        targets[idx].remaining -= 1;
        println!("{}  ({})", verified.address, path.display());
    }

    Ok(())
}

/// GPU mining: epochs of random bases (RAM-locked) walked on the device; hits rebuilt, verified and
/// written on the host. Epoch generation runs on all cores while the GPU waits (~10 ms per epoch).
fn run_search_gpu(specs: Vec<String>, out: PathBuf, nb_max: Option<i32>, allow_unlocked: bool) -> Result<()> {
    const H: u32 = ov_gpu::OV_H;
    const W: i64 = 2 * H as i64 + 1;
    const L: i32 = 100; // launches per epoch (~10 s)

    let mut targets = parse_targets(&specs)?;
    if targets.len() > ov_gpu::MAX_PATTERNS {
        bail!("at most {} patterns per GPU run (got {})", ov_gpu::MAX_PATTERNS, targets.len());
    }
    prepare_out(&out)?;
    credit_existing(&out, &mut targets);

    let running = Arc::new(AtomicBool::new(true));
    {
        let r = running.clone();
        ctrlc::set_handler(move || r.store(false, Ordering::Relaxed)).ok();
    }

    let ctx = ov_gpu::open().context("opening the GPU")?;
    let n = ov_gpu::search_grid_threads(&ctx)?;
    let (ct, cg) = compressed_tables(H);
    let mut s = ov_gpu::Searcher::new(&ctx, H, n, &ct, &cg)?;

    // One fixed, RAM-locked buffer for the epoch's secret bases, reused every epoch.
    // Zeroizing so every return path (incl. `?`/bail! and panic-unwind) wipes it on drop.
    let mut a0_buf = Zeroizing::new(vec![0u8; n * 32]);
    let locked = match lock_mem(&mut a0_buf) {
        Ok(()) => true,
        Err(e) if allow_unlocked => {
            eprintln!("warning: could not lock key memory ({e}); --allow-unlocked set, continuing");
            false
        }
        Err(e) => bail!("could not lock key memory in RAM ({e}); keys could be paged to swap. Pass --allow-unlocked to override."),
    };
    wer_exclude(&a0_buf); // keep the secret bases out of any crash report

    // Calibrate nb on a throwaway epoch.
    let warm: Vec<u8> = (0..n).flat_map(|_| ClampedScalar::random().public_compressed()).collect();
    s.upload_epoch(&warm)?;
    s.set_patterns(&[], &[], 0, 1)?;
    for l in 0..3i32 {
        s.run_launch(1, l)?;
    }
    let t0 = Instant::now();
    s.run_launch(1, 0)?;
    let ms1 = t0.elapsed().as_secs_f64() * 1000.0;
    let mut nb = (((90.0 / ms1).floor()) as i32).clamp(1, (150.0 / ms1).floor().max(1.0) as i32);
    if let Some(cap) = nb_max {
        nb = nb.min(cap).max(1);
    }
    println!("calibrated nb={nb}"); // the `run` supervisor parses this to halve nb after a fault

    let cand_per_launch = n as u128 * nb as u128 * W as u128;
    let bits: Vec<u32> = targets.iter().map(|t| pattern::bits(&t.text)).collect();
    if let Err(e) = pattern::check_selectivity(cand_per_launch as u64, &bits) {
        bail!("pattern set too unselective for the GPU (expected >64 hits/launch): {e:?}. Use longer prefixes (>=5 chars) or --cpu.");
    }
    eprintln!("mining on {} threads, nb={nb}, {n} bases/epoch", s_name(&ctx));

    // Test hook: OV_FAULT_ONCE makes the first run simulate a sticky GPU fault (exit 75) so the
    // supervisor's restart path can be exercised without forcing a disruptive real display TDR.
    if std::env::var_os("OV_FAULT_ONCE").is_some() {
        let marker = out.join(".ov-fault-test");
        if !marker.exists() {
            std::fs::write(&marker, b"").ok();
            eprintln!("OV_FAULT_ONCE: simulating a sticky GPU fault; exiting 75");
            a0_buf.zeroize();
            std::process::exit(75);
        }
    }

    let start = Instant::now();
    let mut total_cand = 0u128;
    let mut epochs = 0u64;
    let mut failures = 0u32;
    let mut canaries = 0u64;
    let mut extra = 0u64; // sibling and over-cap hits: expected, verified, not written
    let mut comp = vec![0u8; n * 32]; // the epoch's public bases

    while running.load(Ordering::Relaxed) && targets.iter().any(|t| t.remaining > 0) {
        // --- generate a fresh epoch in place (overwrites the previous epoch's secrets) + upload ---
        gen_epoch(&mut a0_buf, &mut comp);
        if let Err(e) = s.upload_epoch(&comp) {
            eprintln!("GPU fault (upload): {e:?}; zeroizing and exiting 75 for supervisor restart");
            a0_buf.zeroize(); // exit() skips destructors
            std::process::exit(75);
        }

        // canary at a random evaluated (thread, launch, offset)
        let [tb0, tb1, lb, bb, jb]: [u64; 5] = rand5();
        let t_c = (tb0 ^ (tb1 << 1)) as usize % n;
        let l_c = (lb % L as u64) as i32;
        let b_c = (bb % nb as u64) as i64;
        let j_c = (jb % (2 * H as u64 + 1)) as i64 - H as i64; // [-H, H]
        let off_c = b_c * W + j_c;
        let m_c = l_c as i64 * nb as i64 * W + off_c;
        let canary_target = {
            let pk = base_at(&a0_buf, t_c).add_8m(m_c).unwrap().public_compressed();
            u64::from_le_bytes(pk[..8].try_into().unwrap())
        };
        scalar::scrub_stack(); // the canary's base-point mul leaves scalar digits on the stack
        let masks: Vec<u64> = targets.iter().filter(|t| t.remaining > 0).map(|t| t.mask).collect();
        let tgts: Vec<u64> = targets.iter().filter(|t| t.remaining > 0).map(|t| t.target).collect();
        if let Err(e) = s.set_patterns(&masks, &tgts, !0u64, canary_target) {
            eprintln!("GPU fault (set_patterns): {e:?}; zeroizing and exiting 75");
            a0_buf.zeroize();
            std::process::exit(75);
        }

        let mut consumed: std::collections::HashSet<i32> = std::collections::HashSet::new();
        let mut canary_seen = 0u32;
        let mut complete = true;
        let mut max_launch_ms = 0f64;

        for launch in 0..L {
            if !running.load(Ordering::Relaxed) || targets.iter().all(|t| t.remaining == 0) {
                complete = false;
                break;
            }
            let tl = Instant::now();
            let r = match s.run_launch(nb, launch) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("GPU fault (launch): {e:?}; zeroizing and exiting 75 for supervisor restart");
                    a0_buf.zeroize();
                    std::process::exit(75);
                }
            };
            max_launch_ms = max_launch_ms.max(tl.elapsed().as_secs_f64() * 1000.0);
            total_cand += n as u128 * nb as u128 * W as u128; // nb may have shrunk since calibration
            if r.status != 0 {
                eprintln!("GPU status {:#x} (invalid base or zero denominator): discarding this epoch", r.status);
                complete = false;
                break;
            }
            if r.overflow {
                eprintln!("hit buffer overflowed in launch {launch} ({} hits): hits past 1024 were lost", r.count);
            }
            if let Some(c) = r.canary {
                if (c.thread, c.launch, c.offset as i64) == (t_c as i32, l_c, off_c) {
                    canary_seen += 1;
                } else {
                    failures += 1;
                    eprintln!("canary at wrong (t,m) — {failures}/3");
                }
            }
            for hit in &r.hits {
                if consumed.contains(&hit.thread) {
                    extra += 1;
                    continue; // sibling rule: one key per (thread, epoch)
                }
                let m = launch as i64 * nb as i64 * W + hit.offset as i64;
                let rebuilt = usize::try_from(hit.thread)
                    .ok()
                    .filter(|&t| t < n)
                    .map(|t| base_at(&a0_buf, t).add_8m(m));
                let Some(Ok(a)) = rebuilt else {
                    // a corrupt record or a non-clamped rebuild (never re-clamp): a failed verification
                    failures += 1;
                    eprintln!("hit does not rebuild to a clamped scalar; {failures}/3"); // no offsets in logs
                    if failures >= 3 {
                        bail!("aborting after 3 verification failures");
                    }
                    continue;
                };
                let pubkey = a.public_compressed();
                let texts: Vec<&str> = targets.iter().map(|t| t.text.as_str()).collect();
                let nonce = verify::draw_nonce();
                let mut secret = Zeroizing::new([0u8; 64]);
                secret[..32].copy_from_slice(a.as_bytes());
                secret[32..].copy_from_slice(&nonce[..]);
                let v = match verify::verify_key(&secret, &pubkey, &texts) {
                    Ok(v) => v,
                    Err(e) => {
                        failures += 1;
                        eprintln!("verification failed ({e:?}); {failures}/3");
                        if failures >= 3 {
                            bail!("aborting after 3 verification failures");
                        }
                        continue;
                    }
                };
                let idx = v.credited.expect("a matched hit must credit a pattern");
                if targets[idx].remaining == 0 {
                    extra += 1;
                    continue; // over-cap (longest match is full): count only
                }
                let path = write_or_hold(&out, &v.address, &running, Duration::from_secs(30), || {
                    keyfile::write_key_dir(&out, a.as_bytes(), &nonce, &pubkey)
                })?;
                targets[idx].remaining -= 1;
                consumed.insert(hit.thread);
                println!("{}  ({})", v.address, path.display());
            }
            if !r.hits.is_empty() {
                scalar::scrub_stack(); // hit rebuilds and verification leave scalar digits on the stack
            }
        }

        if complete && canary_seen != 1 {
            bail!("epoch {epochs} completed but the canary was seen {canary_seen} times (expected 1): the kernel is not finding matches");
        }
        canaries += canary_seen as u64;
        epochs += 1;
        let secs = start.elapsed().as_secs_f64();
        let rate = total_cand as f64 / secs / 1e9;
        let caps: Vec<String> = targets
            .iter()
            .filter(|t| t.remaining > 0)
            .map(|t| {
                let mean = 2f64.powi(pattern::bits(&t.text) as i32) / (rate * 1e9);
                format!("{}:{} (~{})", t.text, t.remaining, fmt_eta(mean))
            })
            .collect();
        eprintln!(
            "epoch {epochs} | {rate:.2} G/s | canaries {canaries} | failures {failures} | extra {extra} | max {max_launch_ms:.0}ms | locked {} | [{}]",
            if locked { "yes" } else { "no" },
            caps.join(" ")
        );
        // Enforce the 150 ms launch cap: shrink nb for the next epoch if a launch ran long.
        if max_launch_ms > 150.0 {
            nb = ((nb as f64 * 100.0 / max_launch_ms).floor() as i32).max(1);
            eprintln!("launch exceeded 150ms; nb reduced to {nb} for the next epoch");
            println!("calibrated nb={nb}"); // so the supervisor halves the current nb after a fault
        }
    }

    a0_buf.zeroize(); // clear the last epoch's secret bases on clean exit / Ctrl-C
    if !running.load(Ordering::Relaxed) {
        eprintln!("interrupted; {} pattern(s) still open", targets.iter().filter(|t| t.remaining > 0).count());
    }
    Ok(())
}

fn fmt_eta(secs: f64) -> String {
    if secs < 90.0 {
        format!("{secs:.0}s")
    } else if secs < 5400.0 {
        format!("{:.0}m", secs / 60.0)
    } else if secs < 172_800.0 {
        format!("{:.1}h", secs / 3600.0)
    } else {
        format!("{:.1}d", secs / 86400.0)
    }
}

/// The clamped base scalar stored for thread `t` (a temporary that zeroizes on drop).
fn base_at(buf: &[u8], t: usize) -> ClampedScalar {
    ClampedScalar::from_bytes(buf[t * 32..t * 32 + 32].try_into().unwrap()).expect("clamped base")
}

/// Fill `a0` (the locked buffer) with fresh clamped secret scalars and `comp` with their compressed
/// public points, split across all cores: done serially it idled the GPU ~4% of every epoch.
fn gen_epoch(a0: &mut [u8], comp: &mut [u8]) {
    let workers = std::thread::available_parallelism().map_or(1, |w| w.get());
    let per = (a0.len() / 32).div_ceil(workers).max(1) * 32;
    std::thread::scope(|sc| {
        for (sec, public) in a0.chunks_mut(per).zip(comp.chunks_mut(per)) {
            sc.spawn(move || {
                for (s, p) in sec.chunks_mut(32).zip(public.chunks_mut(32)) {
                    let mut b = [0u8; 32];
                    getrandom::fill(&mut b).expect("CSPRNG failure");
                    s.copy_from_slice(&scalar::clamp(b));
                    b.zeroize();
                    p.copy_from_slice(&base_at(s, 0).public_compressed());
                }
                scalar::scrub_stack(); // dalek's base-point mul leaves scalar digits on this stack
            });
        }
    });
}

/// Exclude the locked secret buffer from any Windows Error Reporting crash dump. On a hard
/// crash (access violation / abort) Drop does not run, so exclusion is the defence there; on a
/// normal panic the Zeroizing buffer is wiped by unwinding.
#[cfg(windows)]
fn wer_exclude(buf: &[u8]) {
    use windows_sys::Win32::System::ErrorReporting::WerRegisterExcludedMemoryBlock;
    unsafe {
        WerRegisterExcludedMemoryBlock(buf.as_ptr() as *const core::ffi::c_void, buf.len() as u32);
    }
}

#[cfg(unix)]
fn wer_exclude(_buf: &[u8]) {}

/// Disable core dumps so a crash cannot write process memory (incl. secrets) to disk.
#[cfg(unix)]
fn harden_crash_dumps() {
    unsafe {
        libc::prctl(libc::PR_SET_DUMPABLE, 0, 0, 0, 0);
        let z = libc::rlimit { rlim_cur: 0, rlim_max: 0 };
        libc::setrlimit(libc::RLIMIT_CORE, &z);
    }
}

#[cfg(windows)]
fn harden_crash_dumps() {}

#[cfg(windows)]
fn lock_mem(buf: &mut [u8]) -> std::io::Result<()> {
    use windows_sys::Win32::System::Memory::{SetProcessWorkingSetSizeEx, VirtualLock};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    unsafe {
        // Raise the working-set quota first (default ~200 KB makes VirtualLock of ~1 MB fail).
        let need = buf.len() + (1 << 20);
        SetProcessWorkingSetSizeEx(GetCurrentProcess(), need, need + (64 << 20), 0);
        if VirtualLock(buf.as_mut_ptr() as *mut core::ffi::c_void, buf.len()) == 0 {
            return Err(std::io::Error::last_os_error());
        }
    }
    Ok(())
}

#[cfg(unix)]
fn lock_mem(buf: &mut [u8]) -> std::io::Result<()> {
    if unsafe { libc::mlock(buf.as_ptr() as *const core::ffi::c_void, buf.len()) } != 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

fn s_name(ctx: &std::sync::Arc<ov_gpu::CudaContext>) -> String {
    ov_gpu::info(ctx).map(|i| i.name).unwrap_or_else(|_| "GPU".into())
}

/// Five random u64 from the CSPRNG (canary placement; not secret).
fn rand5() -> [u64; 5] {
    let mut b = [0u8; 40];
    getrandom::fill(&mut b).expect("CSPRNG");
    core::array::from_fn(|i| u64::from_le_bytes(b[i * 8..i * 8 + 8].try_into().unwrap()))
}

/// Supervise a `search` child, restarting it only on exit code 75 (a sticky GPU fault / TDR), with
/// 30/60/120 s backoff and halved nb, giving up after 2 faults in 10 minutes.
fn run_supervisor(patterns: Vec<String>, out: PathBuf) -> Result<()> {
    let exe = std::env::current_exe().context("finding own executable")?;
    let backoff = [30u64, 60, 120];
    let mut attempt = 0usize;
    let mut losses: Vec<Instant> = Vec::new();
    let mut nb_max: Option<i32> = None;

    loop {
        let mut cmd = Command::new(&exe);
        cmd.arg("search");
        for p in &patterns {
            cmd.arg("-p").arg(p);
        }
        cmd.arg("--out").arg(&out);
        if let Some(nb) = nb_max {
            cmd.arg("--nb-max").arg(nb.to_string());
        }
        cmd.stdout(Stdio::piped()); // stderr is inherited (status line shows live)
        let mut child = cmd.spawn().context("spawning the search child")?;

        let mut last_nb: Option<i32> = None;
        if let Some(stdout) = child.stdout.take() {
            for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                println!("{line}"); // echo hostnames etc.
                if let Some(rest) = line.strip_prefix("calibrated nb=") {
                    last_nb = rest.trim().parse().ok();
                }
            }
        }
        let code = child.wait()?.code().unwrap_or(-1);
        if code == 0 {
            return Ok(());
        }
        if code != 75 {
            bail!("search exited with code {code}; not restarting");
        }

        let now = Instant::now();
        losses.retain(|t| now.duration_since(*t) < Duration::from_secs(600));
        losses.push(now);
        if losses.len() >= 2 {
            bail!("2 GPU faults within 10 minutes; stopping (check the driver/TDR, try a lower --nb-max)");
        }
        nb_max = Some((last_nb.unwrap_or(2) / 2).max(1));
        let wait = backoff[attempt.min(backoff.len() - 1)];
        attempt += 1;
        eprintln!("GPU fault (exit 75); restarting in {wait}s with --nb-max {}", nb_max.unwrap());
        std::thread::sleep(Duration::from_secs(wait));
    }
}

fn run_verify(dir: PathBuf, specs: Vec<String>) -> Result<()> {
    let texts: Vec<String> = specs
        .iter()
        .map(|s| parse_spec(s).map(|(t, _)| t))
        .collect::<Result<_>>()?;
    let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    match verify::verify_dir(&dir, &refs) {
        Ok(v) => {
            println!("OK  {}.onion", v.address);
            Ok(())
        }
        Err(e) => bail!("verification failed: {e:?}"),
    }
}

const BENCH_H: u32 = ov_gpu::OV_H;

/// Compressed jT = (8j)·B for j=1..=H and the giant step G = (2H+1)·8·B (public tables).
fn compressed_tables(h: u32) -> (Vec<u8>, Vec<u8>) {
    let mut table = Vec::with_capacity(h as usize * 32);
    for j in 1..=h as u64 {
        table.extend_from_slice(&EdwardsPoint::mul_base(&Scalar::from(8 * j)).compress().to_bytes());
    }
    let giant = EdwardsPoint::mul_base(&Scalar::from(8 * (2 * h as u64 + 1)))
        .compress()
        .to_bytes()
        .to_vec();
    (table, giant)
}

fn run_bench(secs: u64) -> Result<()> {
    let ctx = ov_gpu::open().context("opening the GPU")?;
    let n = ov_gpu::search_grid_threads(&ctx)?;
    let i = ov_gpu::info(&ctx)?;
    println!("{} cc {}.{} SMs={} | threads={n} (full occupancy), H={BENCH_H}", i.name, i.cc_major, i.cc_minor, i.sm_count);

    let (ct, cg) = compressed_tables(BENCH_H);
    let mut s = ov_gpu::Searcher::new(&ctx, BENCH_H, n, &ct, &cg)?;

    // One epoch of random public bases (the walk just continues across launches for a bench).
    let mut comp = vec![0u8; n * 32];
    for t in 0..n {
        comp[t * 32..t * 32 + 32].copy_from_slice(&ClampedScalar::random().public_compressed());
    }
    s.upload_epoch(&comp)?;
    s.set_patterns(&[0x00FF_FFFF], &[0], 0, 1)?; // 24-bit pattern, canary off

    for l in 0..3i32 {
        s.run_launch(1, l)?; // warm-up (boost clocks, prime caches)
    }
    let t0 = Instant::now();
    s.run_launch(1, 0)?;
    let ms1 = t0.elapsed().as_secs_f64() * 1000.0;
    let nb = (((90.0 / ms1).floor()) as i32).clamp(1, (150.0 / ms1).floor().max(1.0) as i32);
    println!("calibrated nb={nb} (single-batch launch {ms1:.1} ms)");

    let cand_per_launch = n as u128 * nb as u128 * (2 * BENCH_H as u128 + 1);
    let (mut launches, mut hits) = (0u64, 0u128);
    let mut max_ms = 0f64;
    let start = Instant::now();
    let mut l = 0i32;
    while start.elapsed().as_secs_f64() < secs as f64 {
        let tl = Instant::now();
        let r = s.run_launch(nb, l)?;
        max_ms = max_ms.max(tl.elapsed().as_secs_f64() * 1000.0);
        if r.overflow {
            bail!("hit buffer overflowed during bench (pattern too wide)");
        }
        hits += r.count as u128;
        launches += 1;
        l = l.wrapping_add(1);
    }
    let wall = start.elapsed().as_secs_f64();
    let total = cand_per_launch * launches as u128;
    let counted = total as f64 / wall / 1e9;
    let hit_based = hits as f64 * 2f64.powi(24) / wall / 1e9;
    let delta = (counted - hit_based).abs() / counted * 100.0;
    println!("launches={launches} wall={wall:.1}s max_launch={max_ms:.0}ms hits={hits}");
    println!("counted rate:   {counted:.2} G/s");
    println!("hit-based rate: {hit_based:.2} G/s  (cross-check {delta:.1}%)");
    Ok(())
}

fn run_gpu_info() -> Result<()> {
    let ctx = ov_gpu::open().context("opening the GPU")?;
    let i = ov_gpu::info(&ctx)?;
    println!("{} cc {}.{} SMs={}", i.name, i.cc_major, i.cc_minor, i.sm_count);
    let mismatches = ov_gpu::smoke_add_one(&ctx, 1_000_000)?;
    println!("add_one 1M mismatches: {mismatches}");
    if mismatches != 0 {
        bail!("add_one produced {mismatches} mismatches");
    }
    Ok(())
}

/// Pin the DLL search to System32 before any cudarc (and thus nvcuda.dll) call, so a
/// dropped `cuda*.dll` in the CWD/exe dir/PATH cannot be loaded into the key process.
#[cfg(windows)]
fn harden_dll_search() {
    use windows_sys::Win32::System::LibraryLoader::{
        SetDefaultDllDirectories, SetDllDirectoryW, LOAD_LIBRARY_SEARCH_SYSTEM32,
    };
    unsafe {
        SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_SYSTEM32);
        SetDllDirectoryW([0u16].as_ptr()); // L"" — remove the CWD from the search path
    }
}

#[cfg(not(windows))]
fn harden_dll_search() {}

fn main() -> Result<()> {
    harden_dll_search();
    harden_crash_dumps();
    match Cli::parse().cmd {
        Cmd::Search { cpu, patterns, out, h, nb_max, allow_unlocked } => {
            run_search(cpu, patterns, out, h, nb_max, allow_unlocked)
        }
        Cmd::Verify { dir, patterns } => run_verify(dir, patterns),
        Cmd::GpuInfo => run_gpu_info(),
        Cmd::Bench { secs } => run_bench(secs),
        Cmd::Run { patterns, out } => run_supervisor(patterns, out),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_out(label: &str) -> PathBuf {
        let mut r = [0u8; 8];
        getrandom::fill(&mut r).unwrap();
        let d = std::env::temp_dir().join(format!("ov-main-{label}-{:016x}", u64::from_le_bytes(r)));
        keyfile::create_out_dir(&d).unwrap(); // private on Unix whatever the umask, as check_parent needs
        d
    }

    fn target(text: &str, cap: u32) -> Target {
        Target { text: text.to_string(), mask: 0, target: 0, remaining: cap }
    }

    #[test]
    fn parse_targets_refuses_duplicates() {
        let specs = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let t = parse_targets(&specs(&["hello", "hellos=3"])).unwrap();
        assert_eq!((t[0].remaining, t[1].remaining), (1, 3));
        assert!(parse_targets(&specs(&["hello", "hello"])).is_err());
        assert!(parse_targets(&specs(&["hello=2", "hello=5"])).is_err());
    }

    #[test]
    fn longest_match_prefers_longest_then_lowest_index() {
        let t = vec![target("ab", 1), target("abcd", 1), target("abc", 1)];
        assert_eq!(longest_match(&t, "abcdef"), Some(1));
        assert_eq!(longest_match(&t, "abcxyz"), Some(2));
        assert_eq!(longest_match(&t, "zzz"), None);
    }

    #[test]
    fn credit_existing_credits_longest_only_and_ignores_junk() {
        let out = temp_out("credit");
        let a = ClampedScalar::random();
        let pk = a.public_compressed();
        let addr = address::encode(&pk);
        keyfile::write_key_dir(&out, a.as_bytes(), &[0x22; 32], &pk).unwrap();
        std::fs::create_dir(out.join("x.onion")).unwrap();
        std::fs::write(out.join("x.onion").join(keyfile::HOSTNAME_FILE), "garbage").unwrap();
        std::fs::write(out.join("y.onion"), "a file, not a dir").unwrap();
        std::fs::create_dir(out.join("z.onion")).unwrap();

        let mut t = vec![target(&addr[..3], 3), target(&addr[..5], 2)];
        credit_existing(&out, &mut t);
        assert_eq!((t[0].remaining, t[1].remaining), (3, 1));
        credit_existing(&out, &mut t);
        credit_existing(&out, &mut t);
        assert_eq!((t[0].remaining, t[1].remaining), (3, 0), "saturates, never underflows");
        std::fs::remove_dir_all(&out).ok();
    }

    #[test]
    fn write_or_hold_retries_until_the_write_lands() {
        let out = temp_out("hold");
        let running = AtomicBool::new(true);
        let retry = Duration::from_millis(10);

        let mut calls = 0;
        let p = write_or_hold(&out, "addr", &running, retry, || {
            calls += 1;
            if calls < 3 { Err(std::io::Error::other("disk full")) } else { Ok(out.join("addr.onion")) }
        })
        .unwrap();
        assert_eq!((calls, p), (3, out.join("addr.onion")));

        let p = write_or_hold(&out, "addr", &running, retry, || {
            Err(std::io::Error::from(std::io::ErrorKind::AlreadyExists))
        })
        .unwrap();
        assert_eq!(p, out.join("addr.onion"));

        // Ctrl-C during the hold (after the first failure) discards the key instead of looping.
        let mut calls = 0;
        let r = write_or_hold(&out, "addr", &running, retry, || {
            calls += 1;
            if calls == 2 {
                running.store(false, Ordering::Relaxed);
            }
            Err(std::io::Error::other("still failing"))
        });
        assert!(r.is_err() && calls == 2);

        // Ctrl-C before the first failure must not discard the key: it is written, then mining stops.
        running.store(false, Ordering::Relaxed);
        let mut calls = 0;
        let p = write_or_hold(&out, "addr", &running, retry, || {
            calls += 1;
            if calls < 2 { Err(std::io::Error::other("usb unplugged")) } else { Ok(out.join("addr.onion")) }
        })
        .unwrap();
        assert_eq!(p, out.join("addr.onion"));
        assert!(!running.load(Ordering::Relaxed), "the earlier Ctrl-C still stops mining");
        std::fs::remove_dir_all(&out).ok();
    }

    #[test]
    fn gen_epoch_fills_distinct_clamped_bases_with_matching_publics() {
        let n = 1003; // not a multiple of the worker count
        let (mut a0, mut comp) = (vec![0u8; n * 32], vec![0u8; n * 32]);
        gen_epoch(&mut a0, &mut comp);
        let mut seen = std::collections::HashSet::new();
        for t in 0..n {
            let a = base_at(&a0, t); // panics unless clamped
            assert_eq!(a.public_compressed(), comp[t * 32..t * 32 + 32], "thread {t}");
            assert!(seen.insert(a0[t * 32..t * 32 + 32].to_vec()), "duplicate base at {t}");
        }
        let first = a0.clone();
        gen_epoch(&mut a0, &mut comp);
        assert_ne!(first, a0);
    }
}
