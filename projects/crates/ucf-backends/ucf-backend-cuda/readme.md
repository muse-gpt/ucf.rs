# ucf-backend-cuda

NVIDIA backend via **CUDA Driver API** (`nvcuda.dll` / `libcuda.so`).

- Dynamic load with `libloading` — no CUDA Toolkit install required for build
- Kernels from [`ucf-emitter`](../ucf-emitter/) PTX (driver JIT → SASS)
- No `cudarc`, no CUDA C
