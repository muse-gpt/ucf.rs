# ucf-backend-cuda

NVIDIA CUDA Driver API backend (`nvcuda.dll` / `libcuda.so`).

Executes the stage-0 reference path (`Copy` / `Fill` / `MatMul`) on device buffers with host readback. PTX comes from `ucf-emitter` (no CUDA C / nvcc).

Parity vs CPU: `tests/parity_cpu.rs` (soft-skips when no driver).
