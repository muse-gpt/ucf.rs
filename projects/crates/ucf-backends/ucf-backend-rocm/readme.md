# ucf-backend-rocm

ROCm / HIP backend via dynamic `amdhip64` (or `amdhip64_6`).

- Device buffers (`prepare` / `write_f32` / `read_f32`)
- `Copy` via `hipMemcpy` device-to-device
- `Fill` / `MatMul` from [`ucf-emitter`](../../ucf-emitter/) **HSACO** loaded with `hipModuleLoadData` (no `hiprtc`)
- Soft-skips when no HIP runtime / device is available (`tests/parity_cpu.rs`)
- Advertises `UnifiedMemory`, `MatrixCore`, and `HipGraph` via `FeatureSet`
- HipGraph wiring: `Copy` / `Fill` / `MatMul` stream capture → instantiate → launch (`ucf` `tests/hip_graph_fill.rs`, `tests/hip_graph_reference.rs`)
- HipGraph replay vs stream wall-clock report (`ucf` `tests/hip_graph_perf.rs`)
- Mixed-vendor same graph via host bounce (`ucf` `tests/cuda_rocm_host_bridge.rs`)
