# media

媒体库门面。只做再导出，按域提供可选 feature：`image` / `raster` / `canvas` / `pixel` / `text` / `scene` / `dsp` / `synthesis` / `audio` / `post` / `codec` / `render` / `sfx-export`，`default` 含全部。`media.types` 是无条件的基础合同；纯音频消费者可以 `default-features = false, features = ["synthesis"]` 或 `["audio"]`，不会拉入视觉与宿主栈。

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
| `media.dsp` | 确定性 DSP 基元：随机流、双二阶、包络、滑移、声像 |
| `media.synthesis` | 八族参数化音效、`SfxSpec` 合同与命名预设目录 |
| `media.audio` | 音轨计划与混音（音效由 `media.synthesis` 渲染） |
| `media.render` | 逐帧编排 |

下游产品（如 OpenCut）只依赖 `media` 与 `geometry`，自己保留 FFI / 宿主外壳。
