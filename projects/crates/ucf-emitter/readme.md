# ucf-emitter

Built-in shader **emitter** for UCF. Maps [`ShaderProgram`](../ucf-types/src/shader.rs) ops directly to **PTX** or **DXIL container** bytes.

This is not a compiler front-end — no AST lowering, no HLSL, no CUDA C, no `D3DCompile`. Backends consume the emitted blobs as-is.
