# media.dsp

确定性 DSP 基元：xorshift32 随机流 `Rng`、双二阶滤波器 `Biquad`（低通/高通/带通，常数裙边增益）、attack/decay 包络 `env_ad`、指数音高滑移 `glide` 与等功率声像 `pan_gains`。

- `Rng` 是声音域的**唯一正式随机流合同**：`new(seed)` 归一为 `seed.max(1) | 1`（0 与 1 同流），`unit()` 取高 24 位除以 2^24，`bipolar()` 为 `unit*2-1`。逐声音播种，无全局状态；同 seed 恒同流，这是音效字节级复现的地基。
- 滤波系数、包络曲线、滑移与声像公式逐值吸收自 OpenSFX 核心（Apache-2.0，© OpenSFX Authors）。每 64 样本重建扫频滤波器等用法合同归消费者（`media.synthesis`）所有，本 crate 只提供原语本身。
- 零依赖叶子 crate：不认识 scene、audio 与 codec，任何声音消费者可单独引入。
