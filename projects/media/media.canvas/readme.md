# media.canvas

画布积木：图层栈、混合模式、笔刷。四个绘图工具共用这一层，避免各仓各造一套。

| 模块 | 内容 |
|------|------|
| `layer` | `Layer` / `LayerStack` / `BlendMode`，按顺序合成到一张 `Frame` |
| `brush` | `Brush`（半径、硬度、流量）+ 沿线/沿路径盖章的笔画 |

## 边界

- 只做像素运算，不做文档、工具状态、UI。
- 文档模型归各产品仓：OpenPaint 是矢量，OpenPhoto 是栅格图层，OpenPixel 是索引色，OpenVoxel 是体素。
- 合成读的是 `media.image::Frame`，绘制用 `media.raster` 的 `Paint`，不引入新的帧类型。

## 约定

- `opacity` 与 `flow` 都是 0..1。
- 混合按 **source-over** 的扩展：`Multiply` / `Screen` / `Add` 先算颜色再按 alpha 合成。
- 笔画盖章间距固定为半径的 1/3，保证同一路径在任何缩放下连续。
