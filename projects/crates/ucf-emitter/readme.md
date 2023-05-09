# ucf-emitter

Built-in shader **emitter** for UCF. Maps `ShaderProgram` ops to backend-loadable bytes:

| ISA | Bytes | Loaded by |
|-----|-------|-----------|
| PTX | text | CUDA Driver (`cuModuleLoadData`) |
| DXBC | bytecode | D3D12 (system `d3dcompiler` used only inside emitter) |
| SPIR-V | words | Vulkan (`naga` used only inside emitter) |
| HSACO | ELF code object | HIP (`hipModuleLoadData`; OpenCL ICD used only inside emitter) |

Authors never write PTX / HLSL / OpenCL / HIP C. Backends only load and launch the blobs — no `hiprtc` / `nvrtc` in backends.
