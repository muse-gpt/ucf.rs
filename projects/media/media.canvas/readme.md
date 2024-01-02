# media.canvas

画布积木：图层栈、混合模式、笔刷与位图文档。绘图类工具共用这一层，避免各仓各造一套。

| 模块 | 内容 |
|------|------|
| `layer` | `Layer` / `LayerStack` / `BlendMode`，按顺序合成到一张 `Frame` |
| `brush` | `Brush`（半径、硬度、流量）+ 沿线/沿路径盖章的笔画 |
| `doc` | `ImageDoc`：画布尺寸、背景、图层栈与图像放置（左上对齐、裁剪到画布、不缩放） |

## 边界

- 只做像素运算，不做工具状态、UI。
- 位图文档（尺寸 + 背景 + 图层 + 放置）归 `doc`；其余文档模型归各产品仓：OpenPaint 是矢量，OpenPixel 是索引色，OpenVoxel 是体素。
- 图层模型在 Rust 侧完整可用；宿主作者面暴露多少由各产品决定，本 crate 不为其裁剪。
- 合成读的是 `media.image::Frame`，绘制用 `media.raster` 的 `Paint`，不引入新的帧类型。

## 约定

- `opacity` 与 `flow` 都是 0..1。
- 混合按 **source-over** 的扩展：`Multiply` / `Screen` / `Add` 先算颜色再按 alpha 合成。
- 笔画盖章间距固定为半径的 1/3，保证同一路径在任何缩放下连续。
