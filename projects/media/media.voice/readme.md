# media.voice

歌声（人声）合成领域 crate——与 `media.synthesis`（参数化音效）分立的人声职责。

- 输入合同：[`VoiceSpec`] = 音节序列（起音辅音类 + 元音 + 音高 Hz + 时值 beats）+ tempo + vibrato + breath + volume + seed。
- 引擎：共鸣峰（formant）合成——声门脉冲源（相位连续 + 颤音 + 气息）→ 三共振峰元色（2-pole 谐振器）→ 起音事件（擦音噪声 / 爆音间隙+click / 鼻音哼鸣 / R·W·Y 共振峰滑入）→ 攻击/释放包络。
- 合同：范围校验在 Rust（`VoiceSpec::validate`，`invalid_input`）；呼吸随机流为 `media.dsp::Rng`（seed 缺省 1）；同 spec+seed 字节级可复现；元音/辅音名 kebab-case（`a|e|i|o|u`、`s|t|n|r|w|y|f`）。
- 诚实边界：这是类别合成（categorical synthesis）的最小真实现——不是神经声库，不伪装说话人音色参数；声库/模型/推理合同属后续独立建设。
- 文件 IO 不在本 crate：`media.render` 的 `voice-export` feature 组合本渲染与 `media.codec` 写 WAV。
