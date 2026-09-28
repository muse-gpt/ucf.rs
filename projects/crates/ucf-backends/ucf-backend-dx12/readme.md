# ucf-backend-dx12

DirectX 12 backend via raw **D3D12** API (`windows` crate).

- Device buffers (`prepare` / `write_f32` / `read_f32` / `read_u8`)
- `Copy` via `CopyBufferRegion`
- `Fill` / `MatMul` compute from [`ucf-emitter`](../../ucf-emitter/) DXBC (`cs_5_1`)
- `Raster` thin gate: RT clear / optional `draw=tri` yellow triangle → packed `RGBA8` (`read_u8`)
- Windows only (`stub` on other targets)

Parity: `tests/parity_cpu.rs` soft-skips when no D3D12 device is available.
Raster: `tests/raster_clear.rs`, `tests/raster_tri.rs` soft-skip the same way.
