# media

媒体库门面。只做再导出。

```rust
use media::{render_project, scene::Project};
```

| 子库 | 职责 |
|------|------|
| `media.types` | 错误、确定性 hash、采样率 |
| `media.image` | 像素/浮点图像存储与格式转换 |
| `media.scene` | 场景 IR |
| `media.raster` | 软件光栅与 PNG |
| `media.text` | 文字排版与字形 |
| `media.codec` | 宿主解码 / 编码 |
| `media.audio` | 音频床合成与混音 |
| `media.render` | 逐帧编排 |

下游产品（如 OpenCut）只依赖 `media` 与 `geometry`，自己保留 FFI / 宿主外壳。
