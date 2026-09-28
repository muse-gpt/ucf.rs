//! Lightweight compute metrics for auto-tune hooks.

/// Fused-multiply-add count for a naive dense `f32` MatMul: `2 * m * n * k`.
pub fn matmul_flops(m: u64, n: u64, k: u64) -> u64 {
    m.saturating_mul(n)
        .saturating_mul(k)
        .saturating_mul(2)
}

/// Timed MatMul sample used to derive throughput / optional MFU.
#[derive(Debug, Clone, PartialEq)]
pub struct MatmulMetrics {
    /// Rows of `A` / `out`.
    pub m: u32,
    /// Columns of `B` / `out`.
    pub n: u32,
    /// Shared dimension.
    pub k: u32,
    /// Nominal FLOPs (`2*m*n*k`).
    pub flops: u64,
    /// Wall time for the measured region (nanoseconds).
    pub elapsed_nanos: u128,
}

impl MatmulMetrics {
    /// Build metrics from shape and elapsed nanoseconds.
    pub fn new(m: u32, n: u32, k: u32, elapsed_nanos: u128) -> Self {
        Self {
            m,
            n,
            k,
            flops: matmul_flops(m as u64, n as u64, k as u64),
            elapsed_nanos,
        }
    }

    /// Achieved floating-point operations per second (`0.0` if elapsed is 0).
    pub fn flop_per_s(&self) -> f64 {
        if self.elapsed_nanos == 0 {
            return 0.0;
        }
        (self.flops as f64) * 1.0e9 / (self.elapsed_nanos as f64)
    }

    /// Model FLOPs utilization vs a caller-supplied peak (`None` if peak ≤ 0).
    pub fn mfu(&self, peak_flop_per_s: f64) -> Option<f64> {
        if peak_flop_per_s <= 0.0 {
            return None;
        }
        Some(self.flop_per_s() / peak_flop_per_s)
    }
}
