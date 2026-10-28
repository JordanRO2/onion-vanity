//! GPU backend (cudarc driver-only). Receives public data (compressed points, pattern pairs) and
//! returns hits; it never depends on the secret-handling modules, so secrets cannot reach it at
//! compile time.

use std::sync::Arc;

use cudarc::driver::{sys, CudaFunction, CudaModule, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;

// Re-exported so downstream crates can name the handle/error (Searcher::new takes &Arc<CudaContext>).
pub use cudarc::driver::{CudaContext, DriverError};

/// The `sm_120` + `compute_120` fatbin, compiled from `kernel/onion.cu` by `build.rs`.
pub const FATBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/onion.fatbin"));

// `pub const OV_H: u32`: the batch half-width H the fatbin was compiled with (build.rs -DOV_H).
include!(concat!(env!("OUT_DIR"), "/ov_h.rs"));

/// Pattern slots in the kernel's constant table (OV_MAXPAT in kernel/onion.cu).
pub const MAX_PATTERNS: usize = 15;

#[derive(Debug)]
pub struct GpuInfo {
    pub name: String,
    pub cc_major: i32,
    pub cc_minor: i32,
    pub sm_count: i32,
}

/// Retain the primary context on device 0 and set blocking synchronization.
pub fn open() -> Result<Arc<CudaContext>, DriverError> {
    let ctx = CudaContext::new(0)?;
    ctx.set_blocking_synchronize()?;
    Ok(ctx)
}

pub fn info(ctx: &Arc<CudaContext>) -> Result<GpuInfo, DriverError> {
    let (cc_major, cc_minor) = ctx.compute_capability()?;
    let sm_count =
        ctx.attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?;
    Ok(GpuInfo {
        name: ctx.name()?,
        cc_major,
        cc_minor,
        sm_count,
    })
}

/// Run `add_one` on `n` elements initialized to 1.0; returns how many came back != 2.0 (0 = pass).
pub fn smoke_add_one(ctx: &Arc<CudaContext>, n: usize) -> Result<usize, DriverError> {
    let stream = ctx.default_stream();
    let module = ctx.load_module(Ptx::from_binary(FATBIN.to_vec()))?;
    let f = module.load_function("add_one")?;
    let mut dev = stream.clone_htod(&vec![1.0f32; n])?;
    let ni = n as i32;
    let mut b = stream.launch_builder(&f);
    b.arg(&mut dev).arg(&ni);
    unsafe { b.launch(LaunchConfig::for_num_elems(n as u32))? };
    stream.synchronize()?;
    let out: Vec<f32> = stream.clone_dtoh(&dev)?;
    Ok(out.iter().filter(|&&x| x != 2.0).count())
}

/// Run the `t_fe_ops` field kernel over `n = a.len()/8` little-endian 8-u32 elements.
/// `op` 0: a*b, 1: a², 2: a⁻¹, 3: freeze(a), 4: a+b, 5: a-b. `b` must match `a`'s length (ignored
/// for 1/2/3).
pub fn fe_ops(ctx: &Arc<CudaContext>, op: i32, a: &[u32], b: &[u32]) -> Result<Vec<u32>, DriverError> {
    assert!(a.len().is_multiple_of(8) && a.len() == b.len());
    let n = (a.len() / 8) as i32;
    let stream = ctx.default_stream();
    let module = ctx.load_module(Ptx::from_binary(FATBIN.to_vec()))?;
    let f = module.load_function("t_fe_ops")?;
    let da = stream.clone_htod(a)?;
    let db = stream.clone_htod(b)?;
    let mut dout = stream.alloc_zeros::<u32>(a.len())?;
    let block = 128u32; // t_fe_ops is register-heavy (fe_invert); 1024/block overflows the file
    let cfg = LaunchConfig {
        grid_dim: ((n as u32).div_ceil(block), 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = stream.launch_builder(&f);
    builder.arg(&op).arg(&da).arg(&db).arg(&mut dout).arg(&n);
    unsafe { builder.launch(cfg)? };
    stream.synchronize()?;
    stream.clone_dtoh(&dout)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Hit {
    pub thread: i32,
    pub launch: i32,
    pub offset: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Canary {
    pub thread: i32,
    pub launch: i32,
    pub offset: i32,
}

#[derive(Debug)]
pub struct LaunchResult {
    pub status: u32,
    pub count: u32,
    pub overflow: bool,
    pub hits: Vec<Hit>,
    pub canary: Option<Canary>,
}

/// GPU search driver. Public-data-only: compressed public points and pattern pairs in,
/// (thread, launch, offset) hits out; it never sees a secret scalar.
pub struct Searcher {
    ctx: Arc<CudaContext>,
    module: Arc<CudaModule>,
    search_fn: CudaFunction,
    decompress_fn: CudaFunction,
    state: CudaSlice<u32>,
    valid: CudaSlice<u8>,
    table: CudaSlice<u32>,
    giant: CudaSlice<u32>,
    n_threads: usize,
}

impl Searcher {
    /// Build the module and the epoch-independent tables (jT for j=1..H and G). `comp_table` is
    /// `h·32` compressed bytes (`(8j)·B`, j=1..=h); `comp_giant` is 32 bytes (`(2H+1)·8·B`).
    pub fn new(
        ctx: &Arc<CudaContext>,
        h: u32,
        n_threads: usize,
        comp_table: &[u8],
        comp_giant: &[u8],
    ) -> Result<Self, DriverError> {
        assert_eq!(h, OV_H, "host H must equal the fatbin's -DOV_H");
        assert_eq!(comp_table.len(), h as usize * 32);
        assert_eq!(comp_giant.len(), 32);
        let module = ctx.load_module(Ptx::from_binary(FATBIN.to_vec()))?;
        let search_fn = module.load_function("search")?;
        let decompress_fn = module.load_function("decompress_points")?;
        let build_fn = module.load_function("build_tables")?;
        let stream = ctx.default_stream();
        let state = stream.alloc_zeros::<u32>(n_threads * 16)?;
        let valid = stream.alloc_zeros::<u8>(n_threads)?;
        let mut table = stream.alloc_zeros::<u32>(h as usize * 24)?;
        let mut giant = stream.alloc_zeros::<u32>(24)?;
        let djt = stream.clone_htod(comp_table)?;
        let dg = stream.clone_htod(comp_giant)?;
        let hi = h as i32;
        let mut b = stream.launch_builder(&build_fn);
        b.arg(&djt).arg(&mut table).arg(&dg).arg(&mut giant).arg(&hi);
        unsafe { b.launch(cfg_128(h as usize))? };
        stream.synchronize()?;
        Ok(Self {
            ctx: ctx.clone(),
            module,
            search_fn,
            decompress_fn,
            state,
            valid,
            table,
            giant,
            n_threads,
        })
    }

    /// Decompress the epoch's public bases A0[] (`n_threads·32` bytes) into device state and set the
    /// per-thread validity flags.
    pub fn upload_epoch(&mut self, compressed_a0: &[u8]) -> Result<(), DriverError> {
        assert_eq!(compressed_a0.len(), self.n_threads * 32);
        let stream = self.ctx.default_stream();
        let din = stream.clone_htod(compressed_a0)?;
        let mut status = stream.alloc_zeros::<u8>(self.n_threads)?;
        let ni = self.n_threads as i32;
        let mut b = stream.launch_builder(&self.decompress_fn);
        b.arg(&din).arg(&mut self.state).arg(&mut status).arg(&ni);
        unsafe { b.launch(cfg_128(self.n_threads))? };
        stream.synchronize()?;
        let st = stream.clone_dtoh(&status)?;
        let valid: Vec<u8> = st.iter().map(|&s| (s == 0) as u8).collect();
        self.valid = stream.clone_htod(&valid)?;
        Ok(())
    }

    /// Install the pattern table and canary into constant memory (per epoch).
    pub fn set_patterns(
        &self,
        masks: &[u64],
        targets: &[u64],
        canary_mask: u64,
        canary_target: u64,
    ) -> Result<(), DriverError> {
        assert!(masks.len() == targets.len() && masks.len() <= MAX_PATTERNS);
        let mut bytes = Vec::with_capacity(264);
        for i in 0..MAX_PATTERNS {
            bytes.extend_from_slice(&masks.get(i).copied().unwrap_or(0).to_le_bytes());
        }
        for i in 0..MAX_PATTERNS {
            bytes.extend_from_slice(&targets.get(i).copied().unwrap_or(0).to_le_bytes());
        }
        bytes.extend_from_slice(&canary_mask.to_le_bytes());
        bytes.extend_from_slice(&canary_target.to_le_bytes());
        bytes.extend_from_slice(&(masks.len() as i32).to_le_bytes());
        bytes.extend_from_slice(&0i32.to_le_bytes());
        let stream = self.ctx.default_stream();
        let mut sym = self.module.get_global("ov_c", &stream)?;
        stream.memcpy_htod(&bytes, &mut sym)?;
        stream.synchronize()?;
        Ok(())
    }

    /// Run one launch of `nb` batches, tagged `launch`. Reads back the header and hits.
    pub fn run_launch(&mut self, nb: i32, launch: i32) -> Result<LaunchResult, DriverError> {
        let stream = self.ctx.default_stream();
        let mut buf = stream.alloc_zeros::<u32>(8 + 1024 * 3)?;
        let nt = self.n_threads as i32;
        let mut b = stream.launch_builder(&self.search_fn);
        b.arg(&mut self.state)
            .arg(&self.table)
            .arg(&self.giant)
            .arg(&self.valid)
            .arg(&mut buf)
            .arg(&nb)
            .arg(&launch)
            .arg(&nt);
        unsafe { b.launch(cfg_128(self.n_threads))? };
        stream.synchronize()?;
        let h = stream.clone_dtoh(&buf)?;
        let count = h[1];
        let n = count.min(1024) as usize;
        let hits = (0..n)
            .map(|k| Hit {
                thread: h[8 + 3 * k] as i32,
                launch: h[9 + 3 * k] as i32,
                offset: h[10 + 3 * k] as i32,
            })
            .collect();
        let canary = (h[3] == 1).then(|| Canary {
            thread: h[4] as i32,
            launch: h[6] as i32,
            offset: h[5] as i32,
        });
        Ok(LaunchResult {
            status: h[0],
            count,
            overflow: h[2] != 0,
            hits,
            canary,
        })
    }
}

/// Launch config for the register-heavy group/field kernels (128 threads/block).
fn cfg_128(n: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((n as u32).div_ceil(128), 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Decompress `compressed` (n·32 bytes) to affine (x,y). Returns (n·16 u32 limbs, n status bytes).
pub fn decompress_points(ctx: &Arc<CudaContext>, compressed: &[u8]) -> Result<(Vec<u32>, Vec<u8>), DriverError> {
    assert!(compressed.len().is_multiple_of(32));
    let n = compressed.len() / 32;
    let stream = ctx.default_stream();
    let module = ctx.load_module(Ptx::from_binary(FATBIN.to_vec()))?;
    let f = module.load_function("decompress_points")?;
    let din = stream.clone_htod(compressed)?;
    let mut dxy = stream.alloc_zeros::<u32>(n * 16)?;
    let mut dstatus = stream.alloc_zeros::<u8>(n)?;
    let ni = n as i32;
    let mut builder = stream.launch_builder(&f);
    builder.arg(&din).arg(&mut dxy).arg(&mut dstatus).arg(&ni);
    unsafe { builder.launch(cfg_128(n))? };
    stream.synchronize()?;
    Ok((stream.clone_dtoh(&dxy)?, stream.clone_dtoh(&dstatus)?))
}

/// Dual-law: for each (base, tabj) 32-byte pair, return (y(P+jT) bytes, y(P-jT) bytes, status).
pub fn ge_pair(ctx: &Arc<CudaContext>, base: &[u8], tabj: &[u8]) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>), DriverError> {
    assert!(base.len().is_multiple_of(32) && base.len() == tabj.len());
    let m = base.len() / 32;
    let stream = ctx.default_stream();
    let module = ctx.load_module(Ptx::from_binary(FATBIN.to_vec()))?;
    let f = module.load_function("t_ge_pair")?;
    let dbase = stream.clone_htod(base)?;
    let dtabj = stream.clone_htod(tabj)?;
    let mut dyp = stream.alloc_zeros::<u8>(m * 32)?;
    let mut dym = stream.alloc_zeros::<u8>(m * 32)?;
    let mut dstatus = stream.alloc_zeros::<u8>(m)?;
    let mi = m as i32;
    let mut builder = stream.launch_builder(&f);
    builder.arg(&dbase).arg(&dtabj).arg(&mut dyp).arg(&mut dym).arg(&mut dstatus).arg(&mi);
    unsafe { builder.launch(cfg_128(m))? };
    stream.synchronize()?;
    Ok((stream.clone_dtoh(&dyp)?, stream.clone_dtoh(&dym)?, stream.clone_dtoh(&dstatus)?))
}

/// Measure SM-cycles per op for a `bench_*` kernel at full occupancy. Returns the best (least
/// contended) median over 15 timed rounds, normalized per warp-op.
fn bench_one(ctx: &Arc<CudaContext>, name: &str, k: i32) -> Result<f64, DriverError> {
    let stream = ctx.default_stream();
    let module = ctx.load_module(Ptx::from_binary(FATBIN.to_vec()))?;
    let f = module.load_function(name)?;
    let occ = f.occupancy_max_active_blocks_per_multiprocessor(128, 0, None)?; // blocks/SM
    let sms = ctx.attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)? as u32;
    let blocks = sms * occ;
    let warps_per_sm = occ * 4; // 128 threads = 4 warps per block
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut best = f64::INFINITY;
    for round in 0..16 {
        let mut out = stream.alloc_zeros::<i64>((blocks * 128) as usize)?;
        let mut builder = stream.launch_builder(&f);
        builder.arg(&mut out).arg(&k);
        unsafe { builder.launch(cfg)? };
        stream.synchronize()?;
        if round == 0 {
            continue; // warm-up round boosts clocks / primes caches
        }
        let mut d = stream.clone_dtoh(&out)?;
        d.sort_unstable();
        let median = d[d.len() / 2] as f64;
        best = best.min(median);
    }
    // Per-thread SM-cycles: a warp's span amortized over its 32 lanes.
    // A dependent chain is latency-influenced, so this is a conservative upper bound on throughput.
    Ok(best / (warps_per_sm as f64 * 32.0 * k as f64))
}

/// (fe_mul, fe_sqr) SM-cycles per op at full occupancy.
pub fn bench_cycles(ctx: &Arc<CudaContext>, k: i32) -> Result<(f64, f64), DriverError> {
    Ok((bench_one(ctx, "bench_fe_mul", k)?, bench_one(ctx, "bench_fe_sqr", k)?))
}

/// Full-occupancy thread count for the search kernel: SMs × resident blocks/SM × 128 (a whole
/// number of waves, so there is no partial tail wave).
pub fn search_grid_threads(ctx: &Arc<CudaContext>) -> Result<usize, DriverError> {
    let module = ctx.load_module(Ptx::from_binary(FATBIN.to_vec()))?;
    let f = module.load_function("search")?;
    let occ = f.occupancy_max_active_blocks_per_multiprocessor(128, 0, None)?;
    let sms = ctx.attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?;
    Ok(sms as usize * occ as usize * 128)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Requires the RTX 5080; run with `cargo test -p ov-gpu -- --ignored`.
    #[test]
    #[ignore = "needs a CUDA GPU"]
    fn device_and_add_one() {
        let ctx = open().unwrap();
        let i = info(&ctx).unwrap();
        assert_eq!((i.cc_major, i.cc_minor), (12, 0));
        assert_eq!(i.sm_count, 84);
        assert!(i.name.contains("5080"));
        assert_eq!(smoke_add_one(&ctx, 1_000_000).unwrap(), 0);
    }

    // Field arithmetic vs a num-bigint mod-p oracle. Run with `-- --ignored`.
    #[test]
    #[ignore = "needs a CUDA GPU"]
    fn fe_differential() {
        use num_bigint::BigUint;

        let big = |n: u32| BigUint::from(n);
        let one = big(1);
        let p: BigUint = (one.clone() << 255usize) - &big(19);
        let two256: BigUint = one.clone() << 256usize;
        let e255: BigUint = one.clone() << 255usize;
        let edges: Vec<BigUint> = vec![
            big(0),
            one.clone(),
            &p - &one,
            p.clone(),
            &p + &one,
            &e255 - &one,
            &two256 - &one,
            &two256 - &big(38),
            e255.clone(),
            big(19),
            big(38),
            &e255 + &big(19),
        ];

        let mut st = 0x1234_5678_9abc_def0u64;
        let mut next = || {
            let mut bytes = [0u8; 32];
            for c in bytes.chunks_mut(8) {
                st ^= st << 13;
                st ^= st >> 7;
                st ^= st << 17;
                c.copy_from_slice(&st.to_le_bytes());
            }
            BigUint::from_bytes_le(&bytes)
        };
        let rnd: Vec<BigUint> = (0..1_000_000).map(|_| next()).collect(); // 1M random inputs

        let to_limbs = |v: &BigUint| -> [u32; 8] {
            let mut le = v.to_bytes_le();
            le.resize(32, 0);
            let mut o = [0u32; 8];
            for (i, w) in o.iter_mut().enumerate() {
                *w = u32::from_le_bytes(le[4 * i..4 * i + 4].try_into().unwrap());
            }
            o
        };
        let from_limbs = |s: &[u32]| BigUint::new(s.to_vec());

        let ctx = open().unwrap();

        // Binary ops (mul/add/sub) — all edge pairs plus random pairs.
        let (mut a, mut b) = (Vec::new(), Vec::new());
        let (mut o_mul, mut o_add, mut o_sub) = (Vec::new(), Vec::new(), Vec::new());
        let mut push_pair = |x: &BigUint, y: &BigUint, a: &mut Vec<u32>, b: &mut Vec<u32>| {
            a.extend_from_slice(&to_limbs(x));
            b.extend_from_slice(&to_limbs(y));
            o_mul.push((x * y) % &p);
            o_add.push((x + y) % &p);
            o_sub.push(((x % &p) + &p - (y % &p)) % &p);
        };
        for x in &edges {
            for y in &edges {
                push_pair(x, y, &mut a, &mut b);
            }
        }
        for i in 0..rnd.len() {
            let (x, y) = (rnd[i].clone(), rnd[(i + 1) % rnd.len()].clone());
            push_pair(&x, &y, &mut a, &mut b);
        }
        for (op, oracle, name) in [(0i32, &o_mul, "mul"), (4, &o_add, "add"), (5, &o_sub, "sub")] {
            let out = fe_ops(&ctx, op, &a, &b).unwrap();
            for (i, o) in oracle.iter().enumerate() {
                let g = from_limbs(&out[i * 8..i * 8 + 8]);
                assert_eq!(&g % &p, *o, "{name} #{i}");
                assert!(g < two256, "{name} bound #{i}");
            }
        }

        // unary inputs for sqr / freeze.
        let unary: Vec<BigUint> = edges.iter().chain(rnd.iter()).cloned().collect();
        let flat: Vec<u32> = unary.iter().flat_map(to_limbs).collect();

        let out = fe_ops(&ctx, 1, &flat, &flat).unwrap();
        for (i, v) in unary.iter().enumerate() {
            assert_eq!(&from_limbs(&out[i * 8..i * 8 + 8]) % &p, (v * v) % &p, "sqr #{i}");
        }

        let out = fe_ops(&ctx, 3, &flat, &flat).unwrap();
        for (i, v) in unary.iter().enumerate() {
            let g = from_limbs(&out[i * 8..i * 8 + 8]);
            assert_eq!(g, v % &p, "freeze #{i}");
            assert!(g < p, "freeze<p #{i}");
        }

        // Inversion on the edges + 2000 randoms: the CPU modpow oracle is too slow for 1M.
        let inv_in: Vec<&BigUint> = edges.iter().chain(rnd.iter().take(2000)).collect();
        let inv_flat: Vec<u32> = inv_in.iter().flat_map(|&v| to_limbs(v)).collect();
        let out = fe_ops(&ctx, 2, &inv_flat, &inv_flat).unwrap();
        let pm2 = &p - &big(2);
        for (i, v) in inv_in.iter().enumerate() {
            if (*v % &p) == big(0) {
                continue; // 0 has no inverse
            }
            assert_eq!(&from_limbs(&out[i * 8..i * 8 + 8]) % &p, v.modpow(&pm2, &p), "inv #{i}");
        }
    }

    #[test]
    #[ignore = "needs a CUDA GPU"]
    fn fe_throughput_gate() {
        let ctx = open().unwrap();
        let (mul, sqr) = bench_cycles(&ctx, 2000).unwrap();
        println!("fe_mul = {mul:.3} SM-cycles/op, fe_sqr = {sqr:.3} SM-cycles/op (gate 3.3 / 2.7)");
        assert!(mul <= 3.3, "fe_mul {mul:.3} > 3.3 SM-cycles");
        assert!(sqr <= 2.7, "fe_sqr {sqr:.3} > 2.7 SM-cycles");
    }

    fn xorshift_bytes(st: &mut u64) -> [u8; 32] {
        let mut b = [0u8; 32];
        for c in b.chunks_mut(8) {
            *st ^= *st << 13;
            *st ^= *st >> 7;
            *st ^= *st << 17;
            c.copy_from_slice(&st.to_le_bytes());
        }
        b
    }

    // 100k decompress round trips vs dalek.
    #[test]
    #[ignore = "needs a CUDA GPU"]
    fn ge_decompress_roundtrip() {
        use curve25519_dalek::{edwards::EdwardsPoint, scalar::Scalar};
        use num_bigint::BigUint;

        let n = 100_000usize;
        let mut st = 0xabcd_ef01_2345_6789u64;
        let mut comp = Vec::with_capacity(n * 32);
        let mut inputs: Vec<[u8; 32]> = Vec::with_capacity(n);
        for _ in 0..n {
            let a = Scalar::from_bytes_mod_order(xorshift_bytes(&mut st));
            let bytes = EdwardsPoint::mul_base(&a).compress().to_bytes();
            inputs.push(bytes);
            comp.extend_from_slice(&bytes);
        }

        let ctx = open().unwrap();
        let (xy, status) = decompress_points(&ctx, &comp).unwrap();

        let p: BigUint = (BigUint::from(1u32) << 255usize) - 19u32;
        let canon = |limbs: &[u32]| -> BigUint { BigUint::new(limbs.to_vec()) % &p };
        for i in 0..n {
            assert_eq!(status[i], 0, "pt {i} reported invalid");
            let x = canon(&xy[i * 16..i * 16 + 8]);
            let y = canon(&xy[i * 16 + 8..i * 16 + 16]);
            let mut out = [0u8; 32];
            let yb = y.to_bytes_le();
            out[..yb.len()].copy_from_slice(&yb);
            if x.bit(0) {
                out[31] |= 0x80;
            }
            assert_eq!(out, inputs[i], "roundtrip {i}");
        }
    }

    // 100k dual-law ± values vs dalek compress((a±8j)·B).
    #[test]
    #[ignore = "needs a CUDA GPU"]
    fn ge_dual_law() {
        use curve25519_dalek::{edwards::EdwardsPoint, scalar::Scalar};

        let n = 100_000usize;
        let mut st = 0x1357_9bcf_0246_8aceu64; // distinct seed
        let compress = |s: &Scalar| EdwardsPoint::mul_base(s).compress().to_bytes();

        let (mut base, mut tabj) = (Vec::with_capacity(n * 32), Vec::with_capacity(n * 32));
        let mut exp_yp: Vec<[u8; 32]> = Vec::with_capacity(n);
        let mut exp_ym: Vec<[u8; 32]> = Vec::with_capacity(n);
        for k in 0..n {
            let a = Scalar::from_bytes_mod_order(xorshift_bytes(&mut st));
            let j = (k as u64 % 256) + 1; // j in 1..=256 (H)
            let step = Scalar::from(8u64 * j); // jT with T = 8B
            base.extend_from_slice(&compress(&a));
            tabj.extend_from_slice(&compress(&step));
            exp_yp.push(compress(&(a + step)));
            exp_ym.push(compress(&(a - step)));
        }

        let ctx = open().unwrap();
        let (yp, ym, status) = ge_pair(&ctx, &base, &tabj).unwrap();
        for i in 0..n {
            assert_eq!(status[i], 0, "pair {i} invalid");
            let mut ep = exp_yp[i];
            let mut em = exp_ym[i];
            ep[31] &= 0x7f; // compare y only; GPU freeze already clears bit 255
            em[31] &= 0x7f;
            assert_eq!(&yp[i * 32..i * 32 + 32], &ep, "y+ {i}");
            assert_eq!(&ym[i * 32..i * 32 + 32], &em, "y- {i}");
        }
    }
}
