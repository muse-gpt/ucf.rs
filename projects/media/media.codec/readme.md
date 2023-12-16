# media.codec

编解码汇聚点：WIC 解图、Media Foundation 解视频 / 音频、Media Foundation 编码 MP4，以及 PNG 与 WAV 文件写出。

- 解码结果统一成 `media.image` 的 RGBA8 `Bitmap` 与按调用方请求采样率输出的立体声 float 采样。
- `write_png`、`write_wav` 与 `parse_wav_pcm16` 是唯一的图像 / 音频文件编码入口，raster 与 audio 不再内嵌编码。
- WAV 读取保留文件声明的采样率，供混音做输入合同校验，不做隐式重采样。
- 编码优先 AAC，本机没有 AAC 编码器时回落到 FLAC，不引入外部工具链。
- 路径必须绝对化：Media Foundation 不认相对路径。
