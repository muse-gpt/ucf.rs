# ucf-optimize

Declarative optimization passes for placement, tiling, and scheduling.

- [`Optimization`] hints (`apply` is currently a validate-only skeleton)
- [`MatmulMetrics`] / [`matmul_flops`] — MatMul throughput / optional MFU samples
- [`AttentionMetrics`] / [`attention_flops`] — Attention matmul-equivalent FLOPs / optional MFU
- [`PerfReport`] / [`TimedSample`] — cross-backend wall-clock comparison rows

Unit: `tests/metrics.rs`, `tests/perf.rs`. Live timing: `ucf/tests/matmul_metrics.rs`, `ucf/tests/dx_vulkan_perf.rs`.
