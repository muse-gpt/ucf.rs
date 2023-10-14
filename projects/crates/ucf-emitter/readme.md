# ucf-emitter

UCF backend emitter. Legacy `ShaderProgram` operations remain available for compatibility.
Generic UCF `Program` values are lowered from the UCF IR directly to backend-loadable bytes.

| ISA | Bytes | Loaded by |
|-----|-------|-----------|
| PTX | text | CUDA Driver (`cuModuleLoadData`) |
| DXBC | bytecode | D3D12 (system `d3dcompiler` used only inside emitter) |
| SPIR-V | words | Vulkan (`naga` is used as an internal IR and binary writer) |
| HSACO | ELF code object | HIP (`hipModuleLoadData`; OpenCL ICD used only inside emitter) |

Authors never write PTX / HLSL / OpenCL / HIP C, and domain crates never write shader source.
The generic UCF path constructs backend IR directly. Backends only load and launch the blobs.
