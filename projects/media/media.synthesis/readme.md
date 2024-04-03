# media.synthesis

共享声音合成核心，三条并列的目录：

1. **音效**：八个 family（`Family`）、输入合同 `SfxSpec`（serde wire：family kebab-case tag、字段为 Rust snake_case）、命名目录 `PRESETS` 与 `render(spec, sample_rate)`。
2. **乐器**：17 件可演奏乐器（鼓组 9 + 旋律 8）、`InstrumentSpec`（MIDI 音高 + 力度 + 衰减）、`INSTRUMENT_PRESETS` 与 `render_instrument(spec, sample_rate)`。
3. **节奏型**：`DrumPattern`（BPM、每小节步数、小节数、音轨步进力度）、`pattern_presets()`（8 种常用鼓点）与 `render_pattern(pattern, sample_rate)`；每一步复用 `render_instrument`。

共同合同：

- 全部程序化合成：没有采样、没有波表资产、没有学习权重；振荡器用 PolyBLEP 带限，拨弦用 Karplus-Strong 延迟线，金属类用非谐波方波堆叠。
- 随机流唯一合同为 `media.dsp::Rng`（xorshift32），本 crate 不含第二套随机设施；同 spec+seed 字节级复现，测试以 FNV-1a 波形指纹锁定。
- 参数范围合同在 Rust 域：音效 `duration >= 0.01`、`pitch >= 0.05`、`brightness`/`intensity` ∈ `[0,1]`；乐器 `duration` ∈ `[0.02,8]`、`note` ∈ `[21,108]`、`decay`/`tone`/`gain` ∈ `[0,1]`；节奏型 `bpm` ∈ `[20,300]`、每小节 1..32 步、1..32 小节。越界与非有限值都是显式 `Error::invalid_input`，不静默截断。
- 名称解析（`Family::parse`、`Instrument::parse`）与预设查找（`preset`、`instrument_preset`、`pattern_preset`）都是唯一正式入口，绑定只做值映射。
- 只依赖 `media.dsp`、`media.types` 与 serde；不认识 scene、audio 与 codec，文件写出由调用侧组合 `media.codec`（`media.render` 的 `sfx-export`）。
