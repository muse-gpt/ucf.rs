# media.raster

软件光栅：RGBA8 帧缓冲、混合、形状、路径、光带、颗粒、PNG 输出。

- 只做像素数学，不碰 GPU、不解码、不依赖宿主 API。
- 相机推拉由渲染层调用 `Frame::resample_from`，光栅层只负责采样质量。
