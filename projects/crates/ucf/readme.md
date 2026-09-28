# ucf

Public facade for Unified Compute Fabric. Most applications should depend on this crate.

```rust
use ucf::prelude::*;
```

- Re-exports `encode` / `decode` for the thin `.ucf` envelope (`UCF_MAGIC` + `UCF_WIRE_MAJOR` + JSON)
- Optional backends: `cpu` (default), `cuda`, `dx12`, `vulkan`

End-to-end: `tests/ucf_bytes_cpu.rs` encodes the reference graph, decodes it, and runs `Copy → Fill → MatMul` on the CPU backend.