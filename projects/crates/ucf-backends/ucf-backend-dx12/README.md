# ucf-backend-dx12

DirectX 12 backend via raw **D3D12** API (`windows` crate).

- Compute PSO from [`ucf-emitter`](../ucf-emitter/) DXIL bytes
- No HLSL source, no `D3DCompile`
- Windows only (`stub` on other targets)
