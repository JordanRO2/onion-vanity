# onion-vanity

GPU vanity address generator for Tor v3 onion services, written from scratch in Rust + CUDA.
Targets native Windows and Linux with NVIDIA GPUs (developed and measured on an RTX 5080, sm_120).

Measured **~8.2 G candidates/s** on the bench and **~8.3 G/s** on a real run on an RTX 5080,
about 61x mkp224o on the same machine (see [Comparison](#comparison)). Every generated key is
independently verified (re-derived, clamp-checked, self-signed) before it is written, and the key
files are accepted by a real `tor`.

## Requirements

- NVIDIA GPU with compute capability 12.0 (sm_120), i.e. the RTX 50 series. Other architectures
  need the `-gencode` flags in `crates/ov-gpu/build.rs` changed and a rebuild (untested).
- **To build the kernel:** CUDA Toolkit 13.4 (any release that targets sm_120, 12.8 or newer,
  should do; only 13.4 is tested) and MSVC Build Tools 2022 (Windows) or a C++ toolchain (Linux).
  `build.rs` finds `nvcc` via `CUDA_PATH` and `cl.exe` via the registry.
- **To build without the CUDA Toolkit:** the committed `kernel/onion.fatbin` is used as long as the
  kernel sources and flags are unchanged (guarded by a SHA-256 srchash). Only the NVIDIA driver is
  needed to run.
- Rust (stable, MSVC toolchain on Windows).

```
cargo build --release
```

The binary is `target/release/onion-vanity` (`onion-vanity.exe` on Windows); `onion-vanity gpu-info`
checks the GPU and driver. The rest of this README writes `onion-vanity` for it.

## Quick start

From the repository root:

```
# Windows (PowerShell)
.\target\release\onion-vanity.exe run -p hello --out $HOME/onion-keys
# Linux
./target/release/onion-vanity run -p hello --out $HOME/onion-keys
```

This finds one address starting with `hello` (about a second, including startup) and writes it to
`onion-keys/<address>.onion/` in your home directory. `onion-vanity --help` and
`onion-vanity help <command>` print the full option reference.

## Commands

| Command | What it does |
|---|---|
| `run -p <prefix>[=N] ... --out <dir>` | **Use this for long searches.** Supervises `search` and restarts it after a recoverable GPU fault or display TDR (30/60/120 s backoff, halved batch size; gives up after 2 faults in 10 minutes). Takes only `-p` and `--out`: if it stops with an error that suggests `--allow-unlocked` or `--cpu`, run `search` with that flag instead. |
| `search -p <prefix>[=N] ... --out <dir> [options]` | The miner itself, with all the options below. |
| `verify <dir> [-p <prefix>]` | Re-verifies a generated `<address>.onion` directory; with `-p`, also checks the prefix. |
| `bench [--secs 20]` | Measures the candidate rate, with a hit-based cross-check. |
| `gpu-info` | Prints device info and runs a quick kernel self-test. |

`search` options:

| Option | Meaning |
|---|---|
| `-p, --pattern <prefix>[=N]` | Required. Address prefix; `=N` asks for N keys with it (default 1). Repeatable; up to 15 patterns on the GPU. |
| `--out <dir>` | Required. Output directory, created if missing; see [Output directory](#output-directory). |
| `--nb-max <n>` | Caps batches per launch, for shorter GPU launches. `run` halves it after a GPU fault; normally not needed. |
| `--allow-unlocked` | Continues if the secret bases cannot be locked into RAM (keys may then reach swap). |
| `--cpu` | Uses the slow CPU reference walker: for testing, or prefixes under 5 characters. Its key memory is not RAM-locked. |
| `--h <n>` | Walk half-width, `--cpu` only (default 256). |

## Patterns

- Lowercase `a-z` and `2-7` only: the digits `0 1 8 9` do not exist in onion addresses.
- 1 to 12 characters. The GPU needs **5 or more**: a selectivity guard rejects pattern sets that
  would overflow the hit buffer. `--cpu` (about 0.1 M candidates/s, single-threaded) finds a
  4-character prefix in ~10 s on average.
- Several patterns share one pass. The same guard caps the expected hits per launch, so two or
  three 5-character patterns fit together; with 6 or more characters all 15 fit.
- A pattern given twice is refused: use `-p hello=3` for three keys.

Examples:

```
# Three "shadow" keys and one "night7" key in the same pass
onion-vanity search -p shadow=3 -p night7 --out $HOME/onion-keys

# Tiers of one word in a single supervised run: one key per length, the longest when it comes
onion-vanity run -p nightowl -p nightow -p nighto --out $HOME/onion-keys

# Re-check a generated key
onion-vanity verify $HOME/onion-keys/<address>.onion -p hello
```

## Expected time

Each character multiplies the work by 32. Mean time to the first key with onion-vanity at ~8.3 G/s
on an RTX 5080 and with mkp224o at ~135 M/s on a Ryzen 9 9950X3D, both measured on the same machine
(the wait is exponential: p90 ≈ 2.3× the mean):

| Characters | onion-vanity | mkp224o |
|---|---|---|
| 5 | < 0.01 s | 0.25 s |
| 6 | 0.13 s | 8 s |
| 7 | 4 s | 4 min |
| 8 | 2 min | 2.3 h |
| 9 | 1.2 h | 3 days |
| 10 | 1.6 days | 97 days |
| 11 | 50 days | 8.5 years |
| 12 | 4.4 years | 271 years |

Other cards scale with their rate: `bench`, or the first status line of a run, gives your own figure.

**Do not run it next to other GPU jobs.** With another job using ~84% of the GPU, `nb` calibrated
to 4 and the rate fell to about 0.6 G/s.

## Comparison

| Tool | Hardware | Candidates/s | onion-vanity is |
|---|---|---|---|
| **onion-vanity** | RTX 5080 | **~8.2 G** (measured) | - |
| [mkp224o](https://github.com/cathugger/mkp224o) (`-B`, amd64-51-30k, 32 threads) | Ryzen 9 9950X3D | ~135 M (measured on the same machine) | ~61x faster |
| [Onionity](https://github.com/BrokenLatigo/onionity) | RTX 5080 | ~46.7 M (from its README) | ~175x faster |

mkp224o's batched mode shares one expensive key derivation across many candidates, the same idea
onion-vanity uses, but on the CPU. Onionity runs on the GPU but generates a full key pair per
attempt, so even on an RTX 5080 it is slower than mkp224o on a fast CPU.

## Output

Each key is written to `<out>/<address>.onion/` with `hostname`, `hs_ed25519_public_key` and
`hs_ed25519_secret_key`, the files Tor reads from a `HiddenServiceDir`. stdout prints each address
with its directory; stderr prints a status line every ~10 s, like this one:

```
epoch 12 | 8.31 G/s | canaries 12 | failures 0 | extra 0 | max 88ms | locked yes | [moonlight:1 (~71m)]
```

- `[pattern:keys still wanted (~mean time)]` lists what is still open, with the expected time at
  the current rate.
- `failures` must stay at 0: it counts hits that failed verification, and the third one aborts the
  run.
- `extra` counts matches that are deliberately not written: beyond the requested count, or a second
  match from the same random base (keys from one base are related, so only one is kept).
- `canaries` grows by one per epoch: a planted match the kernel must find, as a self-test.

The run exits once every pattern has its keys; Ctrl-C stops it at any time. Keys already in `--out`
count toward the `=N` counts, so a restart does not overshoot: to get another key for the same
prefix, raise N or use a new directory. If writing a key fails (disk full, antivirus), the key is
held in memory and retried every 30 s instead of being lost.

### Output directory

`--out` is checked at startup, so a bad directory fails at once instead of when the first key is
found:

- Only you may be able to write to it (plus SYSTEM and Administrators on Windows). A folder in your
  home directory works; a new folder at the root of `C:\` does not, because it inherits write
  access for Authenticated Users.
- Its filesystem must support an atomic no-replace rename, which WSL `/mnt/c` and NFS may lack.

## Using a key with Tor

Copy the three files from `<address>.onion/` into the service's `HiddenServiceDir` and point
`torrc` at it:

```
HiddenServiceDir /var/lib/tor/myservice/
HiddenServicePort 80 127.0.0.1:8080
```

On Linux the directory must be owned by the Tor user with mode `700`; Tor on Windows (the Expert
Bundle) takes the same two lines with a Windows path. Restart Tor: once it has bootstrapped and
published the service descriptor, the vanity address is reachable, and the `hostname` file in that
directory shows it.

To check a key against a real `tor` first, from the repository root:
`bash scripts/tor-check.sh ~/onion-keys/<address>.onion` on Linux, or
`wsl bash scripts/tor-check.sh /mnt/c/Users/<you>/onion-keys/<address>.onion` on Windows (needs
`tor` installed in WSL).

## Status

Working. The kernel arithmetic is differential-tested against `num-bigint` and
`curve25519-dalek`, and the search kernel against the CPU reference walker. Not yet done: a
long-run soak test and a live publish of a generated key.

## License

MIT; see `LICENSE`.
