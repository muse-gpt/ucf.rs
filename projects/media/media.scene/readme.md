# media.scene

媒体场景 IR：`Project` / `Scene` / `Item` / `Background` / `AudioSpec`。

- 时间单位是**帧**，秒属于作者层。
- 相机关键帧与缓动来自 `geometry`，场景只负责把它们放进 IR。
- 音频床只描述作者意图：BGM 参数、音效放置（`kind` 族名或 `preset` 名 + 覆盖参数）、歌声放置（`media_voice::VoiceSpec`）、乐器音符放置（`notes`，17 件乐器）、鼓点节奏型放置（`patterns`，preset 或自定义步表）、旁白槽与导入片段；解析、合成与混音分别归 `media.audio` / `media.synthesis` / `media.voice`。
- 不渲染、不解码：这里只描述“要画什么”。
