# ucf-backend-vulkan

Vulkan compute backend via **ash** (`vulkan-1`).

- Device buffers (`prepare` / `write_f32` / `read_f32` / `read_u8`)
- `Copy` via `vkCmdCopyBuffer`
- `Fill` / `MatMul` compute from [`ucf-emitter`](../../ucf-emitter/) SPIR-V (WGSL → `naga`)
- `Raster` thin gate: `vkCmdClearColorImage` → packed `RGBA8` buffer (`read_u8`)
- Soft-skips when no Vulkan device is available (`tests/parity_cpu.rs`, `tests/raster_clear.rs`)
