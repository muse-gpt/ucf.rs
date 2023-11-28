# geometry

几何库门面。只做再导出，不放业务逻辑。

```rust
use geometry::{Camera, Ease, Rgba, smooth_polyline};
```

| 子库 | 职责 |
|------|------|
| `geometry.types` | 点 / 尺寸 / 矩形 / 颜色 / 仿射变换 / 静态相机 / 路径 |
| `geometry.motion` | 缓动曲线 / 相机关键帧 / 淡入淡出包络 |

下游（`media.*` 与产品仓）只依赖 `geometry`，不直接依赖子 crate。
