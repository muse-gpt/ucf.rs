# media.codec

宿主编解码：WIC 解图、Media Foundation 解视频 / 音频、Media Foundation 编码 MP4。

- 解码结果统一成 RGBA8 `Bitmap` 与 44.1k 立体声 float 采样。
- 编码优先 AAC，本机没有 AAC 编码器时回落到 FLAC，不引入外部工具链。
- 路径必须绝对化：Media Foundation 不认相对路径。
