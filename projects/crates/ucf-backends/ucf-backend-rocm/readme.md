# ucf-backend-rocm

ROCm / HIP backend via dynamic `amdhip64` (or `amdhip64_6`).

- Device buffers (`prepare` / `write_f32` / `read_f32`)
- `Copy` via `hipMemcpy` device-to-device
- `Fill` / `MatMul` from [`ucf-emitter`](../../ucf-emitter/) **HSACO** loaded with `hipModuleLoadData` (no `hiprtc`)
- Soft-skips when no HIP runtime / device is available (`tests/parity_cpu.rs`)
- Advertises `UnifiedMemory`, `MatrixCore`, and `HipGraph` via `FeatureSet`
