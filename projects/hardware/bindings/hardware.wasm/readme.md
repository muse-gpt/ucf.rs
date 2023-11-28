# hardware-wasm

`wasm32-unknown-unknown` cdylib，导出稳定 C ABI 供浏览器 / Node 宿主加载。默认仅链 `ucf` + `hardware-backend-cpu`（**无 `wgpu` crate**）。

## 构建

```bash
rustup target add wasm32-unknown-unknown
cargo build -p hardware-wasm --target wasm32-unknown-unknown --release
# 产物：target/wasm32-unknown-unknown/release/hardware_wasm.wasm
```

## 导出（C ABI）

| 符号 | 说明 |
|------|------|
| `ucf_version_code` | `major * 1_000_000 + minor * 1_000 + patch` |
| `ucf_target_triple` | 只读 C 字符串 `"wasm32-unknown-unknown"` |
| `ucf_has_cpu_backend` | 恒为 `1`（本 crate 仅 CPU 后端） |
| `ucf_bind_external_buffer` | 经 `ucf_host` import 绑定宿主句柄 |
| `ucf_application_contract_matmul` | 跑 reference Copy→Fill→MatMul 图，写 4 个 `f32`，返回 `0` 或负 `ErrorCode` |
| `ucf_application_contract_diag_mask` | 诊断事件 kind 位掩码（与 `event_kinds` 子集对齐） |
| `ucf_color_fill_rgba8` | L0：UCF CPU `Fill` → RGBA8 写入线性内存 |
| `ucf_l2_color_fill_webgpu` | L2：CPU `Fill` 校验后，经 `ucf_host` 提交 WebGPU 清屏 |
| `ucf_l2_raster_clear_webgpu` | L2：CPU `Raster` 清屏校验后，经 `ucf_host` 提交 WebGPU 清屏 |
| `ucf_l2_raster_tri_webgpu` | L2：CPU `Raster` `draw=tri` 校验后，经 `ucf_host` 提交 WebGPU 黄三角 |
| `ucf_raster_clear_rgba8` | L0：CPU `Raster` 清屏读回 RGBA8 |
| `ucf_raster_tri_rgba8` | L0：CPU `Raster` `draw=tri` 读回 RGBA8 |

## 验收

```bash
cargo test -p hardware-wasm
```

原生测试会通过 `wasmtime` 加载已构建的 `.wasm` 并调用上述符号（与 `tests/application_contract.rs` 同语义）。

TypeScript 加载器不在本仓发布 npm 包；下游产品自行封装（Spark：`@game-gpt/spark-host-web`）。

设计见工作区 `规划设计/ucf/44-ucf-web-platform.md`。
