# media.synthesis

参数化音效合成：八个 family（`Family`）、输入合同 `SfxSpec`（serde wire：family kebab-case tag、字段 camelCase）、命名目录 `PRESETS` 与确定性渲染入口 `render(spec, sample_rate)`。

- 正式语义吸收自 OpenSFX 核心（Apache-2.0，© OpenSFX Authors），逐值保住全部常数与每族随机消耗次序（glitch 每网格 3 次、sparkle 每 ping 2 次）；同 spec+seed 字节级复现是合同，测试以 FNV-1a 波形指纹锁定。
- 随机流唯一合同为 `media.dsp::Rng`（xorshift32），本 crate 不含第二套随机设施。
- 参数范围合同在 Rust 域：`duration >= 0.01`、`pitch >= 0.05`、`brightness`/`intensity` 属于 `[0,1]`、`pan` 属于 `[-1,1]`、seed 缺省统一为 1；越界与非有限值都是显式 `Error::invalid_input`，不静默截断。
- 只依赖 `media.dsp`、`media.types` 与 serde；不认识 scene、audio 与 codec，文件写出由调用侧组合 `media.codec` 完成。
