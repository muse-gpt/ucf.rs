# media

媒体库门面。只做再导出，按域提供可选 feature：`image` / `raster` / `canvas` / `pixel` / `text` / `scene` / `audio` / `post` / `codec` / `render`，`default` 含全部。`media.types` 是无条件的基础合同；纯音频消费者可以 `default-features = false, features = ["audio"]`，不会拉入视觉与宿主栈。

```rust
use media::{render_project, scene::Project};
```

| 子库 | 职责 |
|------|------|
| `media.types` | 错误、确定性 hash、有理时间基、项目默认采样率 |
| `media.image` | 像素/浮点图像存储与格式转换 |
| `media.scene` | 场景 IR |
| `media.raster` | 软件光栅绘制 |
| `media.pixel` | 调色板、索引画布与像素操作 |
| `media.text` | 文字排版与字形 |
| `media.codec` | 宿主解码 / 编码与 PNG / WAV 写出 |
| `media.audio` | 音轨计划、合成与混音 |
| `media.render` | 逐帧编排 |

下游产品（如 OpenCut）只依赖 `media` 与 `geometry`，自己保留 FFI / 宿主外壳。
