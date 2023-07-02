# ucf-backend-cuda

NVIDIA CUDA Driver API backend (`nvcuda.dll` / `libcuda.so`).

Executes the stage-0 reference path (`Copy` / `Fill` / `MatMul`) plus `Custom("denoise")` RGBA8 spatial filter on device buffers with host readback. PTX comes from `ucf-emitter` (no CUDA C / nvcc).

- CUDA Graph thin gate: `Copy` / `Fill` / `MatMul` stream capture → instantiate → launch (`ucf` `tests/cuda_graph_fill.rs`, `tests/cuda_graph_reference.rs`)

Parity vs CPU: `tests/parity_cpu.rs` (soft-skips when no driver).
