# geometry.motion

几何运动：缓动曲线、按帧的相机关键帧曲线、淡入淡出包络。

相机是「几何 + 时间」的组合，因此 `Camera`（静态姿态）在 `geometry.types`，而**关键帧采样**在这里。下游不需要自己实现 `zoom → 画面` 的时间插值。

## 模块

| 模块 | 内容 |
|------|------|
| `ease` | `Ease` 曲线族（含 `out-expo` / `in-out-quart` / `out-back`）与 `fade` 包络 |
| `curve` | `CameraKey` / `CameraPose` / `camera_at`，以及通用分段采样 |

## 约定

- 时间单位是**帧**；秒属于作者层。
- 曲线首尾夹取，段与段之间用**出发关键帧**的缓动，保证一镜到底不跳变。
