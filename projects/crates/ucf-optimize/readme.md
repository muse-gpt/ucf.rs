# ucf-optimize

Declarative optimization passes for placement, tiling, and scheduling.

- [`Optimization`] hints (`apply` is currently a validate-only skeleton)
- [`MatmulMetrics`] / [`matmul_flops`] — MatMul throughput / optional MFU samples

Unit: `tests/metrics.rs`. Live timing: `ucf/tests/matmul_metrics.rs`.
