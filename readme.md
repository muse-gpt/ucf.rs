# media.rs

几何、硬件与媒体三层 Rust 库，供下游产品（OpenCut 等）以 FFI / 宿主外壳消费。

```text
projects/
├── geometry/           # 几何：值类型、仿射变换、相机、缓动与关键帧
│   ├── geometry/           # 门面
│   ├── geometry.types/
│   └── geometry.motion/
├── hardware/           # 硬件：设备/后端建模、图 IR、调度、发射、运行时
│   ├── hardware/
│   ├── hardware.types/
│   ├── hardware.ir/
│   ├── hardware.capability/
│   ├── hardware.optimize/
│   ├── hardware.scheduler/
│   ├── hardware.emitter/
│   ├── hardware.program/
│   ├── hardware.runtime/
│   ├── hardware.backends/  # cpu / cuda / dx12 / vulkan / rocm
│   └── bindings/hardware.wasm
├── media/              # 媒体：场景 IR、光栅、文字、编解码、音频、逐帧编排
│   ├── media/              # 门面
│   ├── media.types/
│   ├── media.scene/
│   ├── media.raster/
│   ├── media.text/
│   ├── media.codec/
│   ├── media.audio/
│   └── media.render/
└── examples/           # 可运行的 IR 示例（每图一个 crate）
```

目录用点分层（`media.raster`），Cargo package 名用连字符（`media-raster`）。三个门面 crate 只做再导出，不放业务逻辑。

## 三条主链

- **几何**：`geometry.types` 定义坐标语义，`geometry.motion` 定义时间曲线；相机 = 几何姿态 + 关键帧，不再由某个产品私有。
- **硬件**：`hardware.ir` → 调度 → 发射 → 后端执行。`.ucf` 仍是线格式名，`UCF_MAGIC` / `UCF_WIRE_MAJOR` 保持不变。
- **媒体**：`media.scene` 描述画面，`media.raster` / `media.text` 画，`media.codec` 解码与编码，`media.audio` 合成音轨，`media.render` 逐帧编排。

## Build

```bash
cargo check --workspace --all-targets
cargo test -p media-render
cargo run -p example-graphics-raster
```

下游依赖：

```toml
media = { path = "../media.rs/projects/media/media" }
geometry = { path = "../media.rs/projects/geometry/geometry" }
```

跨仓正式消费按完整 40 位 `rev` pin。

## License

See [License.md](./License.md).
