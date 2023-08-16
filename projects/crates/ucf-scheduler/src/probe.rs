//! Backend-emitted probe events drained by the scheduler after each submit.

/// Stable probe category (matches `ucf_runtime::event_kinds` strings).
pub type ProbeKind = &'static str;

/// One backend probe (module cache, compile, …) for runtime diagnostics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendProbeEvent {
    /// Backend label (`cuda`, `rocm`, …).
    pub backend: String,
    /// Stable kind string (`module_cache_hit`, …).
    pub kind: ProbeKind,
    /// Optional PTX/HSACO content hash key when known.
    pub shader_key: Option<u64>,
}

impl BackendProbeEvent {
    /// Construct a probe for `backend`.
    pub fn new(backend: impl Into<String>, kind: ProbeKind, shader_key: Option<u64>) -> Self {
        Self {
            backend: backend.into(),
            kind,
            shader_key,
        }
    }
}

/// Stable probe kind strings shared with [`ucf_runtime::event_kinds`].
pub mod kinds {
    /// Module / kernel compile on cache miss.
    pub const MODULE_COMPILE: &str = "module_compile";
    /// Stream-path module cache hit.
    pub const MODULE_CACHE_HIT: &str = "module_cache_hit";
    /// Stream-path module cache miss.
    pub const MODULE_CACHE_MISS: &str = "module_cache_miss";
}
