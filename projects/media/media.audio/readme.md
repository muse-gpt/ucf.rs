# media.audio

音频床合成与混音：pad / pluck、whoosh / tick / impact / riser / glitch，以及导入片段。

- 全部在进程内合成，不调用外部合成器。
- 旁白只留接口：给定 WAV 就混入，不生成语音。
