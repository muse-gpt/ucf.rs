//! Lightweight wall-clock performance samples for cross-backend reports.

/// One named timing sample.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TimedSample {
    /// Sample label (e.g. `matmul`, `raster_clear`).
    pub label: String,
    /// Elapsed nanoseconds.
    pub elapsed_nanos: u128,
}

impl TimedSample {
    /// Build a sample from a label and elapsed duration.
    pub fn new(label: impl Into<String>, elapsed_nanos: u128) -> Self {
        Self {
            label: label.into(),
            elapsed_nanos,
        }
    }

    /// Elapsed seconds as `f64`.
    pub fn elapsed_secs(&self) -> f64 {
        (self.elapsed_nanos as f64) * 1.0e-9
    }
}

/// Timing rows for one backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendPerfRow {
    /// Backend name (`dx12`, `vulkan`, …).
    pub backend: String,
    /// Ordered samples.
    pub samples: Vec<TimedSample>,
}

/// Aggregated cross-backend performance report.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PerfReport {
    /// Backend rows in probe order.
    pub rows: Vec<BackendPerfRow>,
}

impl PerfReport {
    /// Build from `(backend, samples)` pairs.
    pub fn from_rows(rows: impl IntoIterator<Item = BackendPerfRow>) -> Self {
        Self {
            rows: rows.into_iter().collect(),
        }
    }

    /// Stable text lines: `backend/label: <nanos> ns (<secs> s)`.
    pub fn lines(&self) -> Vec<String> {
        let mut out = Vec::new();
        for row in &self.rows {
            for sample in &row.samples {
                out.push(format!(
                    "{}/{}: {} ns ({:.6} s)",
                    row.backend,
                    sample.label,
                    sample.elapsed_nanos,
                    sample.elapsed_secs()
                ));
            }
        }
        out
    }

    /// Look up a sample by backend name and label.
    pub fn sample_nanos(&self, backend: &str, label: &str) -> Option<u128> {
        self.rows
            .iter()
            .find(|r| r.backend == backend)
            .and_then(|r| {
                r.samples
                    .iter()
                    .find(|s| s.label == label)
                    .map(|s| s.elapsed_nanos)
            })
    }

    /// Ratio `a / b` for the same label across two backends (`None` if missing or `b==0`).
    pub fn ratio(&self, backend_a: &str, backend_b: &str, label: &str) -> Option<f64> {
        let a = self.sample_nanos(backend_a, label)? as f64;
        let b = self.sample_nanos(backend_b, label)? as f64;
        if b == 0.0 {
            return None;
        }
        Some(a / b)
    }
}
