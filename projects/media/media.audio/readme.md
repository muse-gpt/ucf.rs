# media.audio

音轨计划与混音：mixer 消费中立的 `AudioPlan`（显式采样率、声道、时长、帧放置与音量），合成 pad / pluck BGM，渲染 `media.synthesis` 的八族音效与 `media.voice` 的歌声，并混入旁白与带 fade 的导入片段。

- 不认识场景：场景 `AudioSpec` 由调用方（render 层）lowering 成 `AudioPlan`。
- 音效算法不在本 crate：`SfxRequest` 只做「family 或 preset + 覆盖参数」的解析（`kind` 与 `preset` 同时给出是显式错误，未知名称是 `invalid_input`），渲染由 `media.synthesis` 完成。
- 歌声算法不在本 crate：`VoicePlacement` 只带 `media.voice::VoiceSpec` 与放置位置/轨音量，渲染由 `media.voice` 完成。
- 随机流唯一合同是 `media.dsp::Rng`：`synth_plucks` 与全部音效共用，不存在第二套随机设施。
- 采样率与声道是输入合同：合成与写出之间不做隐式重采样，旁白采样率不符即报错。
- 旁白只留接口：给定 WAV 就混入，不生成语音。导入片段支持 `fade_in_frames` / `fade_out_frames` 线性淡入淡出（帧基，经 `TimeBase` 推导）。
