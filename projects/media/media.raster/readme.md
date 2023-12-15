# media.raster

软件光栅：在 `media.image` 的 `Frame` 上绘制混合、形状、路径、光带、颗粒，并输出 PNG。

- 只做像素数学，不碰 GPU、不解码、不依赖宿主 API。
- 帧缓冲与解码位图的存储归 `media.image`，本层只有 `Paint` 绘制语义。
- 相机推拉由渲染层调用 `Paint::resample_from`，光栅层只负责采样质量。
