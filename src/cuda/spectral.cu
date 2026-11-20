// Spectral Sieve on CUDA — 2-bit packed multi-shift Hamming.
//
// Target architecture: Pascal (SM_61) and newer.
// Designed for GTX 1060 (1280 cores, 192 GB/s HBM, no tensor cores) and up.
//
// =============================================================================
// Algorithm parallelization
// =============================================================================
// Each CUDA block processes ONE read against its precomputed reference window.
// Within the block, 32 threads (one warp) sweep candidate shifts in tiles of
// 32 at a time:
//
//   * Thread `lane` in iteration `iter` evaluates shift `iter*32 + lane`.
//   * Each thread computes the full Hamming distance for its shift
//     (read_bytes ≤ 64 → 8 u64 SWAR ops max).
//   * Each lane tracks its two best shifts. Warp reduction then selects the
//     two best distinct shifts across the window, breaking ties by position.
//
// This gives us ~ (n_reads × warp_size) parallel SIMD lanes, which fills a
// GTX 1060 (10 SMs × 4 warps/SM = 40 concurrent reads) hundreds of times over
// for a 100K-read batch.
//
// =============================================================================
// Memory layout (host side guarantees)
// =============================================================================
// reads_packed:       n_reads × READ_BYTES_MAX bytes, packed 2-bit DNA
// reads_lens:         n_reads × i32   (actual nucleotide count)
// ref_shifted_flat:   4 × ref_bytes (4 bit-phases of the same window),
//                     concatenated in phase-major order: [phase0, phase1, ...]
// ref_offsets:        n_reads × i32   (byte offset *within each phase* where
//                                       this read's window starts in the
//                                       phase buffer; for batched windows
//                                       sharing a single ref this is simply
//                                       the read's allocated tile offset)
// ref_lens:           n_reads × i32   (effective ref-window length in nucs)
// max_mismatches:     i32[n_reads]     (accept threshold per read)
// out_shifts:         n_reads × i32   (chosen shift; -1 if no accept)
// out_mismatches:     2 × n_reads i32 (best and runner-up Hamming distances)
//
// =============================================================================
// Pascal-specific notes
// =============================================================================
// * No FP16/INT8 tensor ops — everything in INT32 / INT64.
// * Warp shuffles (__shfl_sync) are available since Kepler — fine.
// * No cooperative groups in pre-Volta — we stick to warp-level primitives.

#include <cstdint>
#include <cstring>  // for memcpy (device-side)

// Reads up to 256 nucleotides packed into 64 bytes (16 × u32 / 8 × u64).
// Increase if you target longer reads — the kernel does not internally
// assume a smaller bound, but the host must allocate enough.
#define READ_BYTES_MAX 64

// Unaligned u64 load.
//
// CRITICAL: CUDA on Pascal (and every other arch) strictly requires that
// `ld.global.u64` operands be 8-byte aligned. A reinterpret_cast of a
// pointer that's only byte-aligned (which is *every* `ref + byte_off`
// pointer in this kernel — byte_off can be any small integer) throws
// CUDA_ERROR_MISALIGNED_ADDRESS at runtime. memcpy is the canonical
// portable pattern: nvcc lowers it to either an aligned ld.global.u64
// when the alignment is provable, or to a sequence of byte / u32 loads
// when it isn't — both of which are correct.
__device__ __forceinline__ uint64_t load_u64_unaligned(const uint8_t* p) {
    uint64_t x;
    memcpy(&x, p, sizeof(uint64_t));
    return x;
}

// SWAR pair-OR popcount. Treats each 2-bit pair in `x` as one "lane":
// returns the number of pairs whose either bit is set.
__device__ __forceinline__ int popcount_pair_mask_u64(uint64_t x) {
    uint64_t pair_or = (x | (x >> 1)) & 0x5555555555555555ULL;
    return __popcll(pair_or);
}

// Hamming distance between read[0..n_nucs] and ref starting at byte
// `ref_byte_off` of the chosen bit-phase. Both buffers can be at *any*
// byte alignment — see `load_u64_unaligned` above.
//
// CRITICAL CORRECTNESS: When `n_nucs % 4 != 0` the last packed byte of the
// READ has padding pairs at its high end (we zero-fill in the host). But
// the ref window is *wider* than the read, so the ref's last byte
// (read_bytes - 1) contains REAL nucleotide data in those same high pair
// positions. A naive XOR + popcount would then count read-padding (00) vs
// ref-real (any non-A) as a mismatch — up to `4 - tail_pairs` spurious
// mismatches per Hamming call. For reads right at the accept threshold
// that bumps them above the cutoff and the GPU silently rejects them.
//
// Mirror the CPU `bitpacked::mismatch_count_scalar`: process whole bytes
// in bulk, then mask the partial last byte to keep only the valid pairs.
__device__ __forceinline__ int hamming_2bit(
    const uint8_t* __restrict__ read,
    const uint8_t* __restrict__ ref,
    int read_bytes,
    int n_nucs)
{
    int mism = 0;
    int b = 0;
    int full_bytes = n_nucs / 4;       // bytes whose all 4 pairs are valid
    int tail_pairs = n_nucs % 4;       // 0..3 valid pairs in the next byte

    // Bulk 8-byte SWAR, bounded by `full_bytes` so we never absorb the
    // partial last byte into the bulk path.
    #pragma unroll 8
    while (b + 8 <= full_bytes) {
        uint64_t a = load_u64_unaligned(read + b);
        uint64_t r = load_u64_unaligned(ref + b);
        mism += popcount_pair_mask_u64(a ^ r);
        b += 8;
    }
    // Single-byte loop for the rest of the full bytes (1..7 leftover).
    while (b < full_bytes) {
        uint8_t a = read[b];
        uint8_t r = ref[b];
        uint8_t xor_v = a ^ r;
        uint8_t pair_or = (xor_v | (xor_v >> 1)) & 0x55;
        mism += __popc(pair_or);
        b += 1;
    }
    // Partial last byte: only `tail_pairs` pairs are valid in the read;
    // mask the rest so padding doesn't false-mismatch against real ref
    // bytes. `keep` holds the LSB of each *valid* pair position.
    if (tail_pairs != 0 && b < read_bytes) {
        uint8_t a = read[b];
        uint8_t r = ref[b];
        uint8_t xor_v = a ^ r;
        uint8_t pair_or = (xor_v | (xor_v >> 1)) & 0x55;
        uint8_t keep = (uint8_t)((1u << (tail_pairs * 2)) - 1) & 0x55;
        mism += __popc(pair_or & keep);
    }
    return mism;
}

extern "C" __global__ void spectral_scan_kernel(
    const uint8_t* __restrict__ reads_packed,
    const int32_t* __restrict__ read_lens,
    const uint8_t* __restrict__ ref_shifted_flat, // 4 phases concatenated
    int32_t ref_bytes_per_phase,
    const int32_t* __restrict__ ref_offsets,      // per-read byte offset in each phase
    const int32_t* __restrict__ ref_lens,
    int32_t read_bytes_max,
    const int32_t* __restrict__ max_mismatches,
    int32_t n_reads,
    int32_t* __restrict__ out_shifts,
    int32_t* __restrict__ out_mismatches)
{
    int read_id = blockIdx.x;
    if (read_id >= n_reads) return;

    int lane = threadIdx.x; // 0..31

    int read_len = read_lens[read_id];
    int ref_len  = ref_lens[read_id];
    if (read_len == 0 || ref_len < read_len) {
        if (lane == 0) {
            out_shifts[read_id] = -1;
            out_mismatches[read_id] = INT32_MAX;
            out_mismatches[n_reads + read_id] = INT32_MAX;
        }
        return;
    }
    int n_shifts = ref_len - read_len + 1;
    int max_mism_allowed = max_mismatches[read_id];

    int read_bytes = (read_len + 3) / 4;
    int ref_byte_off_base = ref_offsets[read_id];

    const uint8_t* my_read = reads_packed + (size_t)read_id * read_bytes_max;

    // Order by mismatch count, then by shift. Every lane visits disjoint shifts.
    const uint64_t missing = UINT64_MAX;
    uint64_t best = missing;
    uint64_t second = missing;
    for (int shift = lane; shift < n_shifts; shift += 32) {
        int phase = shift & 3;
        int byte_off = shift >> 2;
        const uint8_t* phase_base =
            ref_shifted_flat + (size_t)phase * ref_bytes_per_phase;
        const uint8_t* ref = phase_base + ref_byte_off_base + byte_off;
        uint64_t mismatches = hamming_2bit(my_read, ref, read_bytes, read_len);
        uint64_t rank = (mismatches << 32) | static_cast<uint32_t>(shift);
        if (rank < best) {
            second = best;
            best = rank;
        } else if (rank < second) {
            second = rank;
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        uint64_t peer_best = __shfl_down_sync(0xFFFFFFFFu, best, offset);
        uint64_t peer_second = __shfl_down_sync(0xFFFFFFFFu, second, offset);
        if (lane < offset) {
            if (peer_best < best) {
                second = best < peer_second ? best : peer_second;
                best = peer_best;
            } else if (peer_best < second) {
                second = peer_best;
            }
        }
    }

    if (lane == 0) {
        int mismatches = static_cast<int>(best >> 32);
        bool accepted = mismatches <= max_mism_allowed;
        out_shifts[read_id] = accepted ? static_cast<int>(static_cast<uint32_t>(best)) : -1;
        out_mismatches[read_id] = accepted ? mismatches : INT32_MAX;
        out_mismatches[n_reads + read_id] =
            second == missing ? INT32_MAX : static_cast<int>(second >> 32);
    }
}
