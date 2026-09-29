# ucf

Public facade for Unified Compute Fabric. Most applications should depend on this crate.

```rust
use ucf::prelude::*;
```

- Re-exports `encode` / `decode` for the thin `.ucf` envelope (`UCF_MAGIC` + `UCF_WIRE_MAJOR` + JSON)
- Application contract helpers: `Runtime::{prepare, run_prepared, flush, capabilities}`, `ExecutionDiagnostics`, and (with `cpu`) `CpuSession`
- External bindings: `ExecutionBindings`, `ExternalBuffer`, `ExecStream` / `ExecEvent`, and (with `cpu`) `HostExternalBuffer`
- Stable scheduler error classes via `SchedulerErrorCode`
- Optional backends: `cpu` (default), `cuda`, `dx12`, `vulkan`, `rocm`

Reference contract: `tests/application_contract.rs` encodes a graph, opens `CpuSession`, then `prepare → write → run_prepared → flush → readback`.
External bindings: `tests/external_bindings_cpu.rs` binds host buffers and an immediate stream before the same lifecycle.
The older `tests/ucf_bytes_cpu.rs` path remains as a lower-level encode/decode + `Runtime::run` check.