//! 乐器：鼓组与旋律乐器的一次性发声。
//!
//! 与八族音效（[`crate::spec`]）分立：音效描述的是“环境事件”（whoosh、
//! impact、riser），乐器描述的是“可演奏的音符”（底鼓、军鼓、贝斯、铺底）。
//! 两者共用 `media.dsp` 的随机流与滤波器，但输入合同不同：乐器有 MIDI 音高、
//! 力度与衰减。
//!
//! 全部为程序化合成：没有采样、没有波表资产、没有学习权重。振荡器用
//! PolyBLEP 带限，拨弦用 Karplus-Strong 延迟线，金属类用非谐波方波堆叠；
//! 同 spec + seed 逐字节复现。

use media_dsp::{Biquad, Rng};
use media_types::Error;

/// 可演奏的乐器。前九个是鼓组，后面是旋律乐器。
#[derive(serde::Serialize, serde::Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum Instrument {
    Kick,
    Snare,
    Clap,
    HatClosed,
    HatOpen,
    Tom,
    Rim,
    Shaker,
    Crash,
    Bass,
    Pluck,
    Pad,
    Lead,
    Bell,
    Marimba,
    Keys,
    Strings,
}

impl Instrument {
    /// 全部乐器，按鼓组 → 旋律顺序（绑定与 UI 都从这里列能力）。
    pub const ALL: [Self; 17] = [
        Self::Kick,
        Self::Snare,
        Self::Clap,
        Self::HatClosed,
        Self::HatOpen,
        Self::Tom,
        Self::Rim,
        Self::Shaker,
        Self::Crash,
        Self::Bass,
        Self::Pluck,
        Self::Pad,
        Self::Lead,
        Self::Bell,
        Self::Marimba,
        Self::Keys,
        Self::Strings,
    ];

    /// kebab-case 正式拼写，与 serde tag 一致。
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Kick => "kick",
            Self::Snare => "snare",
            Self::Clap => "clap",
            Self::HatClosed => "hat-closed",
            Self::HatOpen => "hat-open",
            Self::Tom => "tom",
            Self::Rim => "rim",
            Self::Shaker => "shaker",
            Self::Crash => "crash",
            Self::Bass => "bass",
            Self::Pluck => "pluck",
            Self::Pad => "pad",
            Self::Lead => "lead",
            Self::Bell => "bell",
            Self::Marimba => "marimba",
            Self::Keys => "keys",
            Self::Strings => "strings",
        }
    }

    /// 解析 kebab-case 名称。
    #[must_use]
    pub fn parse(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|instrument| instrument.as_str() == name)
    }

    /// 是否按 MIDI 音高发声（鼓组忽略音高，只做音色微调）。
    #[must_use]
    pub const fn is_pitched(self) -> bool {
        matches!(
            self,
            Self::Bass
                | Self::Pluck
                | Self::Pad
                | Self::Lead
                | Self::Bell
                | Self::Marimba
                | Self::Keys
                | Self::Strings
        )
    }

    /// 打击类的自然尾音长度（秒），用于节奏型里没有显式时值的音符。
    #[must_use]
    pub const fn natural_duration(self) -> f32 {
        match self {
            Self::Kick | Self::Tom => 0.45,
            Self::Snare => 0.32,
            Self::Clap => 0.38,
            Self::HatClosed => 0.12,
            Self::HatOpen => 0.55,
            Self::Rim => 0.1,
            Self::Shaker => 0.16,
            Self::Crash => 1.8,
            _ => 0.6,
        }
    }
}

fn default_seed() -> u32 {
    1
}

fn default_note() -> u8 {
    60
}

fn one() -> f32 {
    1.0
}

fn half() -> f32 {
    0.5
}

/// 一次乐器发声。
///
/// `note` 是 MIDI 音高（A4 = 69 = 440 Hz），鼓组忽略音高但保留字段以统一
/// 数据形状；`decay` 控制尾音长短，`tone` 控制亮度，`gain` 是输出电平。
#[derive(serde::Serialize, serde::Deserialize, Clone, Copy, Debug, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct InstrumentSpec {
    pub instrument: Instrument,
    #[serde(default = "default_seed")]
    pub seed: u32,
    #[serde(default = "default_note")]
    pub note: u8,
    /// 发声时长（秒）。
    pub duration: f32,
    #[serde(default = "half")]
    pub decay: f32,
    #[serde(default = "half")]
    pub tone: f32,
    #[serde(default = "one")]
    pub gain: f32,
    #[serde(default)]
    pub pan: f32,
}

/// MIDI 音高 → 频率（Hz）。
#[must_use]
pub fn midi_to_hz(note: u8) -> f32 {
    440.0 * 2.0_f32.powf((f32::from(note) - 69.0) / 12.0)
}

impl InstrumentSpec {
    /// 以乐器的默认尾音长度为时长构造。
    #[must_use]
    pub fn new(instrument: Instrument, note: u8) -> Self {
        Self {
            instrument,
            seed: 1,
            note,
            duration: if instrument.is_pitched() { 0.6 } else { instrument.natural_duration() },
            decay: 0.5,
            tone: 0.5,
            gain: 1.0,
            pan: 0.0,
        }
    }

    /// 参数范围合同：时长、音高、decay/tone/gain 与有限性都显式校验。
    ///
    /// # Errors
    /// 越界或非有限值为 `invalid_input`。
    pub fn validate(&self) -> Result<(), Error> {
        if !self.duration.is_finite() || !(0.02..=8.0).contains(&self.duration) {
            return Err(Error::invalid_input(format!(
                "instrument duration {} must be within 0.02..=8 seconds",
                self.duration
            )));
        }
        if !(21..=108).contains(&self.note) {
            return Err(Error::invalid_input(format!("instrument note {} must be within 21..=108", self.note)));
        }
        for (name, value) in [("decay", self.decay), ("tone", self.tone), ("gain", self.gain)] {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(Error::invalid_input(format!("instrument {name} {value} must be within 0..=1")));
            }
        }
        if !self.pan.is_finite() {
            return Err(Error::invalid_input(format!("instrument pan {} must be finite", self.pan)));
        }
        Ok(())
    }
}

/// 渲染为立体声 f32。`sample_rate` 是显式输入合同。
///
/// # Errors
/// 越界参数或非法采样率为 `invalid_input`。
pub fn render_instrument(spec: &InstrumentSpec, sample_rate: f32) -> Result<Vec<[f32; 2]>, Error> {
    spec.validate()?;
    if !sample_rate.is_finite() || sample_rate <= 0.0 {
        return Err(Error::invalid_input(format!("instrument sample rate {sample_rate} must be finite and positive")));
    }
    let frames = (spec.duration * sample_rate).round().max(1.0) as usize;
    let mono = render_mono(spec, sample_rate, frames);
    let (left, right) = media_dsp::pan_gains(spec.pan);
    let gain = spec.gain * 0.9;
    Ok(mono
        .into_iter()
        .map(|sample| {
            let value = (sample * gain).clamp(-1.0, 1.0);
            [value * left, value * right]
        })
        .collect())
}

fn render_mono(spec: &InstrumentSpec, sample_rate: f32, frames: usize) -> Vec<f32> {
    let frequency = midi_to_hz(spec.note);
    match spec.instrument {
        Instrument::Kick => kick(spec, sample_rate, frames),
        Instrument::Snare => snare(spec, sample_rate, frames),
        Instrument::Clap => clap(spec, sample_rate, frames),
        Instrument::HatClosed => hat(spec, sample_rate, frames, 90.0),
        Instrument::HatOpen => hat(spec, sample_rate, frames, 9.0),
        Instrument::Tom => tom(spec, sample_rate, frames, frequency),
        Instrument::Rim => rim(spec, sample_rate, frames),
        Instrument::Shaker => shaker(spec, sample_rate, frames),
        Instrument::Crash => crash(spec, sample_rate, frames),
        Instrument::Bass => bass(spec, sample_rate, frames, frequency),
        Instrument::Pluck => pluck(spec, sample_rate, frames, frequency),
        Instrument::Pad => pad(spec, sample_rate, frames, frequency),
        Instrument::Lead => lead(spec, sample_rate, frames, frequency),
        Instrument::Bell => bell(spec, sample_rate, frames, frequency),
        Instrument::Marimba => marimba(spec, sample_rate, frames, frequency),
        Instrument::Keys => keys(spec, sample_rate, frames, frequency),
        Instrument::Strings => strings(spec, sample_rate, frames, frequency),
    }
}

// ---- 鼓组 ------------------------------------------------------------------

/// 底鼓：正弦音高包络（约 120 → 45 Hz）+ 短促 click。
fn kick(spec: &InstrumentSpec, sample_rate: f32, frames: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(frames);
    let mut phase = 0.0_f32;
    let mut click = Biquad::high_pass(1400.0, 0.8, sample_rate);
    let mut rng = Rng::new(spec.seed);
    let rate = 4.5 + spec.decay * 9.0;
    for index in 0..frames {
        let t = index as f32 / sample_rate;
        let frequency = 44.0 + (120.0 - 44.0) * (-t * 28.0).exp();
        phase = (phase + std::f32::consts::TAU * frequency / sample_rate) % std::f32::consts::TAU;
        let body = phase.sin() * (-t * rate).exp();
        let transient = click.process(rng.bipolar()) * (-t * 180.0).exp() * spec.tone * 0.35;
        out.push(body * 0.95 + transient);
    }
    out
}

/// 军鼓：两个音高体 + 带通噪声。
fn snare(spec: &InstrumentSpec, sample_rate: f32, frames: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(frames);
    let mut rng = Rng::new(spec.seed);
    let mut noise = Biquad::band_pass(1500.0 + spec.tone * 2500.0, 0.7, sample_rate);
    let mut high = Biquad::high_pass(4000.0, 0.7, sample_rate);
    let noise_rate = 9.0 + (1.0 - spec.decay) * 14.0;
    for index in 0..frames {
        let t = index as f32 / sample_rate;
        let body = (std::f32::consts::TAU * 185.0 * t).sin() * (-t * 26.0).exp() * 0.5
            + (std::f32::consts::TAU * 330.0 * t).sin() * (-t * 34.0).exp() * 0.2;
        let hiss = noise.process(rng.bipolar()) * 0.75 + high.process(rng.bipolar()) * 0.25;
        out.push(body + hiss * (-t * noise_rate).exp());
    }
    out
}

/// 拍手：三次短噪声爆发 + 尾巴。
fn clap(spec: &InstrumentSpec, sample_rate: f32, frames: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(frames);
    let mut rng = Rng::new(spec.seed);
    let mut band = Biquad::band_pass(1100.0 + spec.tone * 1200.0, 1.4, sample_rate);
    let mut high = Biquad::high_pass(2500.0, 0.7, sample_rate);
    let bursts = [0.0_f32, 0.011, 0.023];
    for index in 0..frames {
        let t = index as f32 / sample_rate;
        let mut envelope = 0.0_f32;
        for offset in bursts {
            if t >= offset {
                envelope += (-(t - offset) * 110.0).exp();
            }
        }
        let tail = (-t * 12.0).exp() * 0.35;
        let noise = band.process(rng.bipolar()) * 0.7 + high.process(rng.bipolar()) * 0.3;
        out.push(noise * (envelope * 0.5 + tail));
    }
    out
}

/// 踩镲：非谐波方波堆叠 + 高通噪声，`rate` 决定开/闭。
fn hat(spec: &InstrumentSpec, sample_rate: f32, frames: usize, rate: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(frames);
    let mut rng = Rng::new(spec.seed);
    let ratios = [1.0_f32, 1.483, 1.932, 2.546, 3.117, 4.271];
    let base = 6200.0 + spec.tone * 2200.0;
    let mut phases = [0.0_f32; 6];
    let mut high = Biquad::high_pass(6000.0, 0.8, sample_rate);
    for index in 0..frames {
        let t = index as f32 / sample_rate;
        let mut metallic = 0.0_f32;
        for (slot, ratio) in ratios.iter().enumerate() {
            let frequency = base * ratio;
            if frequency >= sample_rate * 0.45 {
                continue;
            }
            phases[slot] = (phases[slot] + frequency / sample_rate) % 1.0;
            metallic += if phases[slot] < 0.5 { 1.0 } else { -1.0 };
        }
        metallic /= ratios.len() as f32;
        let noise = high.process(rng.bipolar());
        let envelope = (-t * rate).exp();
        out.push((metallic * 0.55 + noise * 0.45) * envelope * 0.5);
    }
    out
}

/// 通鼓：正弦下滑 + 噪声攻击。
fn tom(spec: &InstrumentSpec, sample_rate: f32, frames: usize, frequency: f32) -> Vec<f32> {
    let base = frequency.clamp(70.0, 240.0);
    let mut out = Vec::with_capacity(frames);
    let mut phase = 0.0_f32;
    let mut rng = Rng::new(spec.seed);
    let mut noise = Biquad::low_pass(2500.0, 0.7, sample_rate);
    for index in 0..frames {
        let t = index as f32 / sample_rate;
        let swept = base * (1.0 + 0.6 * (-t * 14.0).exp());
        phase = (phase + std::f32::consts::TAU * swept / sample_rate) % std::f32::consts::TAU;
        let body = phase.sin() * (-t * (4.0 + spec.decay * 6.0)).exp();
        let attack = noise.process(rng.bipolar()) * (-t * 90.0).exp() * 0.25;
        out.push(body * 0.9 + attack);
    }
    out
}

/// 边击：极短的带通噪声 + 木质感。
fn rim(spec: &InstrumentSpec, sample_rate: f32, frames: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(frames);
    let mut rng = Rng::new(spec.seed);
    let mut band = Biquad::band_pass(2400.0 + spec.tone * 1800.0, 3.0, sample_rate);
    for index in 0..frames {
        let t = index as f32 / sample_rate;
        let wooden = (std::f32::consts::TAU * 820.0 * t).sin() * (-t * 220.0).exp() * 0.4;
        let click = band.process(rng.bipolar()) * (-t * 260.0).exp();
        out.push(wooden + click * 0.8);
    }
    out
}

/// 沙锤：高通噪声 + 短包络。
fn shaker(spec: &InstrumentSpec, sample_rate: f32, frames: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(frames);
    let mut rng = Rng::new(spec.seed);
    let mut high = Biquad::high_pass(5000.0 + spec.tone * 2000.0, 0.9, sample_rate);
    for index in 0..frames {
        let t = index as f32 / sample_rate;
        let attack = (t / 0.004).min(1.0);
        let envelope = attack * (-t * (20.0 + (1.0 - spec.decay) * 40.0)).exp();
        out.push(high.process(rng.bipolar()) * envelope * 0.55);
    }
    out
}

/// 吊镲：宽带噪声 + 非谐波金属尾音。
fn crash(spec: &InstrumentSpec, sample_rate: f32, frames: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(frames);
    let mut rng = Rng::new(spec.seed);
    let ratios = [1.0_f32, 1.341, 1.767, 2.229, 2.872, 3.519, 4.211, 5.107];
    let base = 3200.0 + spec.tone * 1800.0;
    let mut phases = [0.0_f32; 8];
    let mut high = Biquad::high_pass(3500.0, 0.7, sample_rate);
    for index in 0..frames {
        let t = index as f32 / sample_rate;
        let mut metallic = 0.0_f32;
        for (slot, ratio) in ratios.iter().enumerate() {
            let frequency = base * ratio;
            if frequency >= sample_rate * 0.45 {
                continue;
            }
            phases[slot] = (phases[slot] + frequency / sample_rate) % 1.0;
            metallic += (std::f32::consts::TAU * phases[slot]).sin();
        }
        metallic /= ratios.len() as f32;
        let noise = high.process(rng.bipolar());
        let envelope = (-t * (1.2 + (1.0 - spec.decay) * 2.0)).exp();
        out.push((noise * 0.75 + metallic * 0.25) * envelope * 0.55);
    }
    out
}

// ---- 旋律乐器 --------------------------------------------------------------

/// PolyBLEP 修正项：去除锯齿/方波在跳变处的混叠。
fn poly_blep(t: f32, dt: f32) -> f32 {
    if t < dt {
        let x = t / dt;
        x + x - x * x - 1.0
    } else if t > 1.0 - dt {
        let x = (t - 1.0) / dt;
        x * x + x + x + 1.0
    } else {
        0.0
    }
}

fn saw(phase: f32, dt: f32) -> f32 {
    (2.0 * phase - 1.0) - poly_blep(phase, dt)
}

fn square(phase: f32, dt: f32) -> f32 {
    let raw = if phase < 0.5 { 1.0 } else { -1.0 };
    raw + poly_blep(phase, dt) - poly_blep((phase + 0.5) % 1.0, dt)
}

/// 力度包络：短攻 + 保持 + 释放，`duration` 是整段长度。
fn shaped(t: f32, duration: f32, attack: f32, release: f32) -> f32 {
    let attack_gain = if attack <= 0.0 { 1.0 } else { (t / attack).min(1.0) };
    let remaining = duration - t;
    let release_gain = if remaining < release { (remaining / release.max(1e-4)).max(0.0) } else { 1.0 };
    attack_gain * release_gain
}

/// 贝斯：锯齿 + 低八度正弦，低通随音高变化。
fn bass(spec: &InstrumentSpec, sample_rate: f32, frames: usize, frequency: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(frames);
    let mut phase = 0.0_f32;
    let mut sub = 0.0_f32;
    let cutoff = (frequency * (2.0 + spec.tone * 8.0)).clamp(80.0, sample_rate * 0.45);
    let mut low = Biquad::low_pass(cutoff, 0.8, sample_rate);
    let dt = frequency / sample_rate;
    let duration = spec.duration;
    for index in 0..frames {
        let t = index as f32 / sample_rate;
        phase = (phase + dt) % 1.0;
        sub = (sub + frequency * 0.5 / sample_rate) % 1.0;
        let tone = low.process(saw(phase, dt));
        let envelope = shaped(t, duration, 0.008, 0.06) * (-t * (1.2 - spec.decay * 0.8)).exp();
        out.push((tone * 0.7 + (std::f32::consts::TAU * sub).sin() * 0.45) * envelope);
    }
    out
}

/// 拨弦：Karplus-Strong 延迟线（噪声激励 + 平均滤波反馈）。
fn pluck(spec: &InstrumentSpec, sample_rate: f32, frames: usize, frequency: f32) -> Vec<f32> {
    let mut rng = Rng::new(spec.seed);
    let length = ((sample_rate / frequency).round() as usize).clamp(2, frames.max(2));
    let mut buffer: Vec<f32> = (0..length).map(|_| rng.bipolar()).collect();
    let damping = 0.9 + spec.decay * 0.09;
    let mut cursor = 0_usize;
    let mut out = Vec::with_capacity(frames);
    for index in 0..frames {
        let current = buffer[cursor];
        let next_index = (cursor + 1) % length;
        let filtered = 0.5 * (current + buffer[next_index]) * damping;
        buffer[cursor] = filtered;
        cursor = next_index;
        let t = index as f32 / sample_rate;
        out.push(current * shaped(t, spec.duration, 0.002, 0.05));
    }
    out
}

/// 铺底：三个失谐锯齿 + 慢起音。
fn pad(spec: &InstrumentSpec, sample_rate: f32, frames: usize, frequency: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(frames);
    let mut phases = [0.0_f32; 3];
    let detune = [0.997_f32, 1.0, 1.004];
    let dt = frequency / sample_rate;
    let cutoff = (600.0 + spec.tone * 2600.0).clamp(200.0, sample_rate * 0.45);
    let mut low = Biquad::low_pass(cutoff, 0.6, sample_rate);
    let attack = 0.25;
    for index in 0..frames {
        let t = index as f32 / sample_rate;
        let mut mix = 0.0_f32;
        for slot in 0..3 {
            phases[slot] = (phases[slot] + dt * detune[slot]) % 1.0;
            mix += saw(phases[slot], dt);
        }
        let envelope = shaped(t, spec.duration, attack, 0.3) * 0.5;
        out.push(low.process(mix / 3.0) * envelope);
    }
    out
}

/// 主音：锯齿 + 方波 + 轻微颤音。
fn lead(spec: &InstrumentSpec, sample_rate: f32, frames: usize, frequency: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(frames);
    let mut phase = 0.0_f32;
    let mut phase_square = 0.0_f32;
    let duration = spec.duration;
    for index in 0..frames {
        let t = index as f32 / sample_rate;
        let vibrato = 1.0 + 0.004 * (std::f32::consts::TAU * 5.2 * t).sin();
        let dt = frequency * vibrato / sample_rate;
        phase = (phase + dt) % 1.0;
        phase_square = (phase_square + dt * 0.5) % 1.0;
        let mix = saw(phase, dt) * 0.7 + square(phase_square, dt * 0.5) * 0.3;
        let envelope = shaped(t, duration, 0.012, 0.08);
        out.push(mix * envelope * 0.55);
    }
    out
}

/// 铃：FM（非整数调制比）+ 长衰减。
fn bell(spec: &InstrumentSpec, sample_rate: f32, frames: usize, frequency: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(frames);
    let mut carrier = 0.0_f32;
    let mut modulator = 0.0_f32;
    let ratio = 3.5_f32;
    let duration = spec.duration;
    for index in 0..frames {
        let t = index as f32 / sample_rate;
        let index_amount = 3.0 * (-t * 3.0).exp();
        modulator = (modulator + frequency * ratio / sample_rate) % 1.0;
        let mod_signal = (std::f32::consts::TAU * modulator).sin() * index_amount;
        carrier = (carrier + frequency / sample_rate) % 1.0;
        let tone = (std::f32::consts::TAU * carrier + mod_signal).sin();
        let envelope = shaped(t, duration, 0.002, 0.25) * (-t * 1.8).exp();
        out.push(tone * envelope * 0.55);
    }
    out
}

/// 马林巴：基音 + 四次谐波 + 槌击噪声。
fn marimba(spec: &InstrumentSpec, sample_rate: f32, frames: usize, frequency: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(frames);
    let mut rng = Rng::new(spec.seed);
    let mut mallet = Biquad::band_pass(3000.0, 1.0, sample_rate);
    let duration = spec.duration;
    for index in 0..frames {
        let t = index as f32 / sample_rate;
        let fundamental = (std::f32::consts::TAU * frequency * t).sin() * (-t * 12.0).exp();
        let overtone = (std::f32::consts::TAU * frequency * 4.0 * t).sin() * (-t * 30.0).exp() * 0.3;
        let strike = mallet.process(rng.bipolar()) * (-t * 260.0).exp() * 0.2 * spec.tone;
        out.push((fundamental + overtone + strike) * shaped(t, duration, 0.002, 0.08) * 0.6);
    }
    out
}

/// 电钢：双算子 FM（tine 音叉）+ 中长衰减。
fn keys(spec: &InstrumentSpec, sample_rate: f32, frames: usize, frequency: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(frames);
    let mut carrier = 0.0_f32;
    let mut modulator = 0.0_f32;
    let duration = spec.duration;
    for index in 0..frames {
        let t = index as f32 / sample_rate;
        let index_amount = 2.0 * (-t * 8.0).exp();
        modulator = (modulator + frequency * 14.0 / sample_rate) % 1.0;
        let mod_signal = (std::f32::consts::TAU * modulator).sin() * index_amount;
        carrier = (carrier + frequency / sample_rate) % 1.0;
        let tone = (std::f32::consts::TAU * carrier + mod_signal).sin()
            + (std::f32::consts::TAU * carrier * 2.0).sin() * 0.2;
        let envelope = shaped(t, duration, 0.004, 0.12) * (-t * 2.6).exp();
        out.push(tone * envelope * 0.5);
    }
    out
}

/// 弦乐：四个失谐锯齿 + 慢起音。
fn strings(spec: &InstrumentSpec, sample_rate: f32, frames: usize, frequency: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(frames);
    let mut phases = [0.0_f32; 4];
    let detune = [0.994_f32, 0.999, 1.001, 1.006];
    let dt = frequency / sample_rate;
    let cutoff = (1400.0 + spec.tone * 2600.0).clamp(400.0, sample_rate * 0.45);
    let mut low = Biquad::low_pass(cutoff, 0.7, sample_rate);
    for index in 0..frames {
        let t = index as f32 / sample_rate;
        let mut mix = 0.0_f32;
        for slot in 0..4 {
            phases[slot] = (phases[slot] + dt * detune[slot]) % 1.0;
            mix += saw(phases[slot], dt);
        }
        let envelope = shaped(t, spec.duration, 0.18, 0.25);
        out.push(low.process(mix / 4.0) * envelope * 0.5);
    }
    out
}

/// 命名乐器预设：稳定的名字 + 默认参数，宿主存名字而不是下标。
pub struct InstrumentPreset {
    pub name: &'static str,
    pub description: &'static str,
    pub spec: InstrumentSpec,
}

const fn preset_spec(instrument: Instrument, note: u8, duration: f32, decay: f32, tone: f32, pan: f32) -> InstrumentSpec {
    InstrumentSpec { instrument, seed: 1, note, duration, decay, tone, gain: 1.0, pan }
}

/// 常用乐器预设目录。
pub const INSTRUMENT_PRESETS: &[InstrumentPreset] = &[
    InstrumentPreset { name: "kick-808", description: "长尾 808 底鼓", spec: preset_spec(Instrument::Kick, 36, 0.9, 0.85, 0.4, 0.0) },
    InstrumentPreset { name: "kick-tight", description: "短促的 house 底鼓", spec: preset_spec(Instrument::Kick, 36, 0.32, 0.15, 0.6, 0.0) },
    InstrumentPreset { name: "snare-crack", description: "明亮军鼓", spec: preset_spec(Instrument::Snare, 60, 0.3, 0.35, 0.75, 0.0) },
    InstrumentPreset { name: "snare-soft", description: "柔和军鼓", spec: preset_spec(Instrument::Snare, 60, 0.28, 0.6, 0.35, 0.0) },
    InstrumentPreset { name: "clap-wide", description: "宽拍手", spec: preset_spec(Instrument::Clap, 60, 0.4, 0.5, 0.55, 0.0) },
    InstrumentPreset { name: "hat-closed", description: "闭镲", spec: preset_spec(Instrument::HatClosed, 60, 0.1, 0.4, 0.55, 0.15) },
    InstrumentPreset { name: "hat-open", description: "开镲", spec: preset_spec(Instrument::HatOpen, 60, 0.5, 0.6, 0.5, -0.15) },
    InstrumentPreset { name: "tom-low", description: "低通鼓", spec: preset_spec(Instrument::Tom, 43, 0.45, 0.5, 0.4, -0.2) },
    InstrumentPreset { name: "tom-high", description: "高通鼓", spec: preset_spec(Instrument::Tom, 55, 0.4, 0.45, 0.6, 0.2) },
    InstrumentPreset { name: "rim-click", description: "边击", spec: preset_spec(Instrument::Rim, 60, 0.1, 0.3, 0.6, 0.1) },
    InstrumentPreset { name: "shaker-soft", description: "轻沙锤", spec: preset_spec(Instrument::Shaker, 60, 0.16, 0.4, 0.5, 0.25) },
    InstrumentPreset { name: "crash-bright", description: "明亮吊镲", spec: preset_spec(Instrument::Crash, 60, 1.8, 0.6, 0.7, 0.0) },
    InstrumentPreset { name: "bass-sub", description: "低音贝斯", spec: preset_spec(Instrument::Bass, 36, 0.9, 0.5, 0.25, 0.0) },
    InstrumentPreset { name: "bass-pick", description: "拨片贝斯", spec: preset_spec(Instrument::Bass, 40, 0.6, 0.3, 0.6, 0.0) },
    InstrumentPreset { name: "pluck-nylon", description: "尼龙拨弦", spec: preset_spec(Instrument::Pluck, 60, 0.8, 0.55, 0.5, 0.0) },
    InstrumentPreset { name: "pad-warm", description: "暖铺底", spec: preset_spec(Instrument::Pad, 57, 2.0, 0.6, 0.4, 0.0) },
    InstrumentPreset { name: "lead-square", description: "方波主音", spec: preset_spec(Instrument::Lead, 72, 0.8, 0.5, 0.6, 0.0) },
    InstrumentPreset { name: "bell-glass", description: "玻璃铃", spec: preset_spec(Instrument::Bell, 84, 2.0, 0.7, 0.6, 0.0) },
    InstrumentPreset { name: "marimba-wood", description: "木质马林巴", spec: preset_spec(Instrument::Marimba, 72, 1.0, 0.5, 0.5, 0.0) },
    InstrumentPreset { name: "keys-electric", description: "电钢", spec: preset_spec(Instrument::Keys, 60, 1.6, 0.55, 0.5, 0.0) },
    InstrumentPreset { name: "strings-swell", description: "弦乐渐强", spec: preset_spec(Instrument::Strings, 57, 2.0, 0.6, 0.5, 0.0) },
];

/// 按名字取乐器预设。
#[must_use]
pub fn instrument_preset(name: &str) -> Option<InstrumentSpec> {
    INSTRUMENT_PRESETS.iter().find(|preset| preset.name == name).map(|preset| preset.spec)
}

/// 全部乐器预设名，按目录顺序。
#[must_use]
pub fn instrument_preset_names() -> Vec<&'static str> {
    INSTRUMENT_PRESETS.iter().map(|preset| preset.name).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fnv1a64(bytes: &[u8]) -> u64 {
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        for byte in bytes {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        hash
    }

    #[test]
    fn names_round_trip_and_kebab_matches_the_serde_tag() {
        for instrument in Instrument::ALL {
            assert_eq!(Instrument::parse(instrument.as_str()), Some(instrument));
            let json = serde_json::to_string(&instrument).expect("serialisable");
            assert_eq!(json, format!("\"{}\"", instrument.as_str()));
        }
        assert_eq!(Instrument::parse("theremin"), None);
    }

    #[test]
    fn midi_note_69_is_concert_a() {
        assert!((midi_to_hz(69) - 440.0).abs() < 1e-3);
        assert!((midi_to_hz(60) - 261.625_55).abs() < 0.01);
    }

    #[test]
    fn every_instrument_renders_audible_finite_samples_at_the_contract_length() {
        for instrument in Instrument::ALL {
            let spec = InstrumentSpec::new(instrument, if instrument.is_pitched() { 60 } else { 36 });
            let samples = render_instrument(&spec, 44_100.0).expect("valid spec");
            assert_eq!(samples.len(), (spec.duration * 44_100.0).round() as usize, "{} length", instrument.as_str());
            let peak = samples.iter().fold(0.0_f32, |peak, frame| peak.max(frame[0].abs()));
            assert!(peak > 0.01, "{} is silent (peak {peak})", instrument.as_str());
            assert!(peak <= 1.0 + 1e-6, "{} clips (peak {peak})", instrument.as_str());
            assert!(samples.iter().all(|frame| frame[0].is_finite() && frame[1].is_finite()));
        }
    }

    #[test]
    fn same_spec_and_seed_reproduces_byte_identically() {
        for instrument in [Instrument::Kick, Instrument::Snare, Instrument::Pluck, Instrument::Pad] {
            let spec = InstrumentSpec::new(instrument, 57);
            let a = render_instrument(&spec, 48_000.0).expect("a");
            let b = render_instrument(&spec, 48_000.0).expect("b");
            assert_eq!(a, b, "{} must be deterministic", instrument.as_str());
        }
    }

    #[test]
    fn different_seeds_change_noise_but_not_length() {
        let mut a = InstrumentSpec::new(Instrument::Snare, 60);
        a.seed = 3;
        let mut b = a;
        b.seed = 11;
        let first = render_instrument(&a, 44_100.0).expect("a");
        let second = render_instrument(&b, 44_100.0).expect("b");
        assert_eq!(first.len(), second.len());
        assert_ne!(first, second, "the noise stream must follow the seed");
    }

    #[test]
    fn range_contract_rejects_out_of_bounds() {
        let mut spec = InstrumentSpec::new(Instrument::Kick, 36);
        spec.duration = 0.001;
        assert_eq!(render_instrument(&spec, 44_100.0).expect_err("short").code(), "invalid_input");

        let mut spec = InstrumentSpec::new(Instrument::Kick, 36);
        spec.note = 12;
        assert_eq!(render_instrument(&spec, 44_100.0).expect_err("note").code(), "invalid_input");

        let mut spec = InstrumentSpec::new(Instrument::Kick, 36);
        spec.gain = 1.5;
        assert_eq!(render_instrument(&spec, 44_100.0).expect_err("gain").code(), "invalid_input");

        assert_eq!(
            render_instrument(&InstrumentSpec::new(Instrument::Pluck, 60), f32::NAN).expect_err("rate").code(),
            "invalid_input"
        );
    }

    #[test]
    fn instrument_presets_are_unique_valid_and_named_stably() {
        let names = instrument_preset_names();
        let unique: std::collections::BTreeSet<_> = names.iter().collect();
        assert_eq!(unique.len(), names.len());
        for preset in INSTRUMENT_PRESETS {
            preset.spec.validate().expect("every preset passes the range contract");
            assert_eq!(instrument_preset(preset.name), Some(preset.spec));
        }
        assert_eq!(instrument_preset("theremin"), None);
    }

    #[test]
    fn golden_fingerprint_pins_the_instrument_engine() {
        let mut bytes = Vec::new();
        for instrument in [Instrument::Kick, Instrument::Snare, Instrument::HatClosed, Instrument::Bass, Instrument::Pluck, Instrument::Bell] {
            let spec = InstrumentSpec::new(instrument, 57);
            for frame in render_instrument(&spec, 44_100.0).expect("render") {
                bytes.extend_from_slice(&frame[0].to_bits().to_ne_bytes());
            }
        }
        assert_eq!(fnv1a64(&bytes), 992265080340798733, "instrument constants are pinned");
    }
}
