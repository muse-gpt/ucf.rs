# media.vector

矢量文档：`Shape`（路径/圆角矩形/圆）、`VectorLayer`、`VectorDoc`、宿主输入合同 `VectorSpec` 与规范化表示 `NormDoc`。

- 本 crate 不画像素、不写文件：`normalize` 产出中立规范化图元序列，光栅绘制由 `media.render` 的 `vector-export` 组合入口驱动 `media.raster`，SVG 编码由 `media.codec` 的 `svg` feature 消费同一份 `NormDoc`。
- 同一规范化表示驱动两产物：圆角矩形描边在 PNG 与 SVG 中几何一致（吸收自 OpenPaint 的有意修正之一——旧光栅路忽略 radius 按四边折线描）。
- 图层透明度在规范化时折进每个图元的 paint alpha（源实现的 `with_alpha` 替换语义），隐藏层与 `opacity <= 0.001` 的层在此剔除，两路不再各自判断。
- 输入合同住在 Rust：白背景、图层名 `layer`、opacity clamp(0..=1)、stroke width 2.0、坐标与宽度必须有限；非法 hex 颜色是显式 `invalid_input` 错误，不再静默丢 paint。
- 只依赖 `media.types`、`geometry` 与 serde；路径数学（`smooth_polyline`、`rounded_rect_outline`）归 `geometry.types`。
