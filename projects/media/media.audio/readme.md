# media.audio

音轨计划与混音：mixer 消费中立的 `AudioPlan`（显式采样率、声道、时长、帧放置与音量），合成 pad / pluck、whoosh / tick / impact / riser / glitch / sub 内置音效，并混入旁白与导入片段。

- 不认识场景：场景 `AudioSpec` 由调用方（render 层）lowering 成 `AudioPlan`。
- 全部在进程内合成，不调用外部合成器。
- 采样率与声道是输入合同：合成与写出之间不做隐式重采样，旁白采样率不符即报错。
- 旁白只留接口：给定 WAV 就混入，不生成语音。
