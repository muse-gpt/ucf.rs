# media.pixel

索引像素画：调色板 `Palette`（默认 16 色暖色表与 hex 解析）、索引画布 `PixelCanvas`（line/rect/flood/replace 与最近邻放大）与 sprite spec 重放入口 `render_sprite`。

- 编辑全程停留在 `Vec<u8>` 索引域，RGBA 展开只发生在导出边界（`to_bitmap` 产出 `media.image` 存储）。
- 0 号索引为透明索引；越界读回 0、越界写静默——flood 的回绕邻居推进依赖这对合同，不可单方面改成报错。
- 非法 hex 颜色是显式 `invalid_input` 错误（吸收自 OpenPixel 时的唯一有意修正）；调色板槽位越界写与未知索引读保持各自的既有语义。
- 只依赖 `media.types` 与 `media.image`；PNG 等编码归 `media.codec`，由调用侧组合，本 crate 不导入。
