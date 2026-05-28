//! Thin cudarc-backed runtime for the Spectral Sieve kernel.

use super::{CudaError, READ_BYTES_MAX};
use cudarc::driver::{CudaContext, CudaModule, CudaStream, LaunchConfig, PushKernelArg};
use std::sync::Arc;

/// Compiled PTX, embedded into the binary at build time.
const SPECTRAL_PTX: &str = include_str!(env!("KIRA_SPECTRAL_PTX"));

/// `true` when `build.rs` failed to invoke nvcc and emitted a stub PTX.
const SPECTRAL_PTX_IS_STUB: bool = option_env!("KIRA_SPECTRAL_PTX_STUB").is_some();

/// One job submitted to the GPU: a single read against its already-padded reference window.
#[derive(Clone, Debug)]
pub struct CudaJob {
    /// 2-bit packed read, length `(read_nucs + 3) / 4` bytes.
    pub read_packed: Vec<u8>,
    /// Number of *nucleotides* in `read_packed` (not bytes).
    pub read_nucs: u32,
    /// 4 bit-phase shifts of the reference window, all the same length.
    pub ref_shifted: [Vec<u8>; 4],
    /// Number of nucleotides in the reference window (matches CPU `ref_len`).
    pub ref_nucs: u32,
    /// Maximum mismatches the caller will accept before falling through to WFA / banded SW.
    pub max_mismatches: u32,
}

/// Result for one job.
#[derive(Clone, Copy, Debug)]
pub struct CudaResult {
    pub shift: i32,
    pub mismatches: i32,
}

/// Probe whether a CUDA-capable device is reachable from this process.
pub fn cuda_runtime_available() -> bool {
    CudaContext::new(0).is_ok()
}

/// The GPU backend handle. Hold one per server session.
pub struct CudaBackend {
    /// Owning handle to the CUDA context.
    #[allow(dead_code)]
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    /// Reusable device buffer for packed reads; grows as needed.
    reads_buf: Option<cudarc::driver::CudaSlice<u8>>,
    read_lens_buf: Option<cudarc::driver::CudaSlice<i32>>,
    /// Reusable device buffer for the 4 ref phases, flattened.
    ref_buf: Option<cudarc::driver::CudaSlice<u8>>,
    ref_lens_buf: Option<cudarc::driver::CudaSlice<i32>>,
    ref_offsets_buf: Option<cudarc::driver::CudaSlice<i32>>,
    out_shifts_buf: Option<cudarc::driver::CudaSlice<i32>>,
    out_mism_buf: Option<cudarc::driver::CudaSlice<i32>>,
}

impl CudaBackend {
    /// Initialise CUDA, JIT-compile the PTX, return a ready-to-launch handle.
    pub fn new() -> Result<Self, CudaError> {
        if SPECTRAL_PTX_IS_STUB {
            return Err(CudaError::Driver(
                "This binary contains a STUB PTX — the CUDA kernel was \
                 not compiled at build time. Look at your `cargo build \
                 --features cuda` output for `cargo:warning=` lines \
                 starting with `nvcc:` — they show why nvcc failed. \
                 Common causes on Windows: cl.exe missing or version \
                 mismatch with CUDA toolkit. Fix and rebuild."
                    .to_string(),
            ));
        }
        let ctx = CudaContext::new(0).map_err(|e| CudaError::Driver(e.to_string()))?;
        let stream = ctx.default_stream();
        let module = ctx
            .load_module(SPECTRAL_PTX.into())
            .map_err(|e| CudaError::Driver(e.to_string()))?;
        Ok(Self {
            ctx,
            stream,
            module,
            reads_buf: None,
            read_lens_buf: None,
            ref_buf: None,
            ref_lens_buf: None,
            ref_offsets_buf: None,
            out_shifts_buf: None,
            out_mism_buf: None,
        })
    }

    /// Process a batch of jobs in one kernel launch.
    pub fn run_batch(&mut self, jobs: &[CudaJob]) -> Result<Vec<CudaResult>, CudaError> {
        if jobs.is_empty() {
            return Ok(Vec::new());
        }

        for j in jobs {
            let bytes = j.read_packed.len();
            if bytes > READ_BYTES_MAX {
                return Err(CudaError::ReadTooLong {
                    len: bytes,
                    max: READ_BYTES_MAX,
                });
            }
            // All 4 phase buffers must be identical in length per job.
            let l = j.ref_shifted[0].len();
            if !j.ref_shifted.iter().all(|p| p.len() == l) {
                return Err(CudaError::Runtime(
                    "ref_shifted phases have different lengths".to_string(),
                ));
            }
        }

        let n_reads = jobs.len();
        let read_bytes_max = READ_BYTES_MAX;

        let mut host_reads: Vec<u8> = vec![0u8; n_reads * read_bytes_max];
        let mut host_read_lens: Vec<i32> = Vec::with_capacity(n_reads);
        for (i, j) in jobs.iter().enumerate() {
            let dst = &mut host_reads[i * read_bytes_max..i * read_bytes_max + j.read_packed.len()];
            dst.copy_from_slice(&j.read_packed);
            host_read_lens.push(j.read_nucs as i32);
        }

        let ref_bytes_per_window: Vec<usize> =
            jobs.iter().map(|j| j.ref_shifted[0].len()).collect();
        let total_ref_bytes_per_phase: usize = ref_bytes_per_window.iter().sum();
        let ref_bytes_per_phase_i32 = total_ref_bytes_per_phase as i32;

        let mut host_ref_flat: Vec<u8> = vec![0u8; 4 * total_ref_bytes_per_phase];
        let mut host_ref_offsets: Vec<i32> = Vec::with_capacity(n_reads);
        let mut host_ref_lens: Vec<i32> = Vec::with_capacity(n_reads);

        let mut offset_acc = 0usize;
        for (i, j) in jobs.iter().enumerate() {
            let window_bytes = ref_bytes_per_window[i];
            for phase in 0..4usize {
                let dst_lo = phase * total_ref_bytes_per_phase + offset_acc;
                host_ref_flat[dst_lo..dst_lo + window_bytes]
                    .copy_from_slice(&j.ref_shifted[phase]);
            }
            host_ref_offsets.push(offset_acc as i32);
            host_ref_lens.push(j.ref_nucs as i32);
            offset_acc += window_bytes;
        }

        let stream = &self.stream;
        ensure_capacity_u8(stream, &mut self.reads_buf, host_reads.len())?;
        ensure_capacity_i32(stream, &mut self.read_lens_buf, host_read_lens.len())?;
        ensure_capacity_u8(stream, &mut self.ref_buf, host_ref_flat.len())?;
        ensure_capacity_i32(stream, &mut self.ref_lens_buf, host_ref_lens.len())?;
        ensure_capacity_i32(stream, &mut self.ref_offsets_buf, host_ref_offsets.len())?;
        ensure_capacity_i32(stream, &mut self.out_shifts_buf, n_reads)?;
        ensure_capacity_i32(stream, &mut self.out_mism_buf, n_reads)?;

        // Copy fresh inputs into the (possibly resized) buffers.
        stream
            .memcpy_htod(&host_reads, self.reads_buf.as_mut().unwrap())
            .map_err(|e| CudaError::Runtime(e.to_string()))?;
        stream
            .memcpy_htod(&host_read_lens, self.read_lens_buf.as_mut().unwrap())
            .map_err(|e| CudaError::Runtime(e.to_string()))?;
        stream
            .memcpy_htod(&host_ref_flat, self.ref_buf.as_mut().unwrap())
            .map_err(|e| CudaError::Runtime(e.to_string()))?;
        stream
            .memcpy_htod(&host_ref_lens, self.ref_lens_buf.as_mut().unwrap())
            .map_err(|e| CudaError::Runtime(e.to_string()))?;
        stream
            .memcpy_htod(&host_ref_offsets, self.ref_offsets_buf.as_mut().unwrap())
            .map_err(|e| CudaError::Runtime(e.to_string()))?;
        // Initialise output buffers to sentinel values for this batch.
        let host_out_shifts: Vec<i32> = vec![-1i32; n_reads];
        let host_out_mism: Vec<i32> = vec![i32::MAX; n_reads];
        stream
            .memcpy_htod(&host_out_shifts, self.out_shifts_buf.as_mut().unwrap())
            .map_err(|e| CudaError::Runtime(e.to_string()))?;
        stream
            .memcpy_htod(&host_out_mism, self.out_mism_buf.as_mut().unwrap())
            .map_err(|e| CudaError::Runtime(e.to_string()))?;

        let max_mism_first = jobs[0].max_mismatches as i32;

        let cfg = LaunchConfig {
            grid_dim: (n_reads as u32, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };

        let func = self
            .module
            .load_function("spectral_scan_kernel")
            .map_err(|e| CudaError::Driver(e.to_string()))?;

        unsafe {
            let mut launcher = stream.launch_builder(&func);
            launcher
                .arg(self.reads_buf.as_ref().unwrap())
                .arg(self.read_lens_buf.as_ref().unwrap())
                .arg(self.ref_buf.as_ref().unwrap())
                .arg(&ref_bytes_per_phase_i32)
                .arg(self.ref_offsets_buf.as_ref().unwrap())
                .arg(self.ref_lens_buf.as_ref().unwrap())
                .arg(&(read_bytes_max as i32))
                .arg(&max_mism_first)
                .arg(&(n_reads as i32))
                .arg(self.out_shifts_buf.as_mut().unwrap())
                .arg(self.out_mism_buf.as_mut().unwrap())
                .launch(cfg)
                .map_err(|e| CudaError::Runtime(e.to_string()))?;
        }

        let mut out_shifts: Vec<i32> = vec![0; n_reads];
        let mut out_mism: Vec<i32> = vec![0; n_reads];
        let shifts_view = self
            .out_shifts_buf
            .as_ref()
            .unwrap()
            .slice(0..n_reads);
        let mism_view = self
            .out_mism_buf
            .as_ref()
            .unwrap()
            .slice(0..n_reads);
        stream
            .memcpy_dtoh(&shifts_view, &mut out_shifts)
            .map_err(|e| CudaError::Runtime(e.to_string()))?;
        stream
            .memcpy_dtoh(&mism_view, &mut out_mism)
            .map_err(|e| CudaError::Runtime(e.to_string()))?;

        // Synchronise before reading.
        stream
            .synchronize()
            .map_err(|e| CudaError::Runtime(e.to_string()))?;

        Ok(out_shifts
            .into_iter()
            .zip(out_mism)
            .map(|(s, m)| CudaResult {
                shift: s,
                mismatches: m,
            })
            .collect())
    }

    /// Free device buffers (called automatically on drop).
    pub fn release_buffers(&mut self) {
        self.reads_buf = None;
        self.read_lens_buf = None;
        self.ref_buf = None;
        self.ref_lens_buf = None;
        self.ref_offsets_buf = None;
        self.out_shifts_buf = None;
        self.out_mism_buf = None;
    }
}

struct _NotSendSync(std::marker::PhantomData<*const ()>);
impl CudaBackend {
    fn _enforce_pin(&self) {
        let _marker: _NotSendSync = _NotSendSync(std::marker::PhantomData);
    }
}

/// Ensure a `Vec<u8>`-typed device buffer has at least `need` capacity.
fn ensure_capacity_u8(
    stream: &Arc<CudaStream>,
    buf: &mut Option<cudarc::driver::CudaSlice<u8>>,
    need: usize,
) -> Result<(), CudaError> {
    let have = buf.as_ref().map(|s| s.len()).unwrap_or(0);
    if have >= need {
        return Ok(());
    }
    // Geometric growth: at least double, but never less than the actual need.
    let new_cap = need.max(have.saturating_mul(2));
    *buf = Some(
        stream
            .alloc_zeros::<u8>(new_cap)
            .map_err(|e| CudaError::Runtime(e.to_string()))?,
    );
    Ok(())
}

/// Same for `i32`-typed device buffers.
fn ensure_capacity_i32(
    stream: &Arc<CudaStream>,
    buf: &mut Option<cudarc::driver::CudaSlice<i32>>,
    need: usize,
) -> Result<(), CudaError> {
    let have = buf.as_ref().map(|s| s.len()).unwrap_or(0);
    if have >= need {
        return Ok(());
    }
    let new_cap = need.max(have.saturating_mul(2));
    *buf = Some(
        stream
            .alloc_zeros::<i32>(new_cap)
            .map_err(|e| CudaError::Runtime(e.to_string()))?,
    );
    Ok(())
}
