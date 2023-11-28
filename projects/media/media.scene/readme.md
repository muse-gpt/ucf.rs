# media.scene

媒体场景 IR：`Project` / `Scene` / `Item` / `Background` / `AudioSpec`。

- 时间单位是**帧**，秒属于作者层。
- 相机关键帧与缓动来自 `geometry`，场景只负责把它们放进 IR。
- 不渲染、不解码：这里只描述“要画什么”。
