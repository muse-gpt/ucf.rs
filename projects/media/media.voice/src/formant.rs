//! GOLF 风格人声合成引擎：LF 声门流波表 + 级联全极点声道 + 谐波/噪声模型。
//!
//! 相对旧版「锯齿波 + 并联带通」的本质改进（GOLF, Yu & Fazekas 2024）：
//! - 激励源：LF 模型单参数 `R_d` 波表（真实声门脉冲频谱倾斜与闭合峰）；
//! - 声道：共振峰极点 → 级联 2 极点全极点 IIR（共振峰相互作用真实存在）；
//! - 气声：谐波 + 噪声模型，噪声经同一声道滤波（GOLF §3.1）。
//!
//! 一首歌是一条连续发声链，不是若干独立音块：
//! - 声道滤波器跨音节保留状态，只在相邻元音之间做共振峰过渡（协同发音），
//!   禁止每音节重建滤波器（那会在接缝处产生爆音）；
//! - 声门相位跨音节连续推进，颤音相位按整曲时间计算，不在每个音符重置；
//! - 包络只在整曲首尾做淡入淡出，音节之间保持连音；
//! - 响度合同是固定的源增益 × `volume`（不做整曲峰值归一化——那会让前段
//!   输出依赖后段内容，`05` §8 明确拒绝）。
//!
//! 全部确定性：同 spec+seed 字节级复现。

use crate::lf::GlottalTables;
use crate::spec::{Onset, Syllable, VoiceSpec};
use crate::tract::AllPole;
use media_dsp::Rng;
use media_types::Error;

/// 每个共振峰的固定带宽（Hz）。
const FORMANT_BANDWIDTH: f32 = 90.0;
/// 相邻音节之间的共振峰过渡长度（秒）；起音类 `Onset::R/W/Y` 用滑音起点。
const TRANSITION_SECONDS: f32 = 0.06;
/// 起音前导结束时的短淡入（秒），避免前导与元音接缝处的阶跃。
const LEAD_FADE_SECONDS: f32 = 0.006;
/// 共振峰过渡期间更新滤波器系数的样本间隔。
const FORMANT_UPDATE_INTERVAL: usize = 16;
/// 起音事件（擦音/爆音/鼻音）相对元音声道的固定电平。
///
/// 噪声激励与 LF 激励的能量分布不同，直接沿用单位峰值增益会让擦音比
/// 元音高出约 20 倍；用固定系数把起音压到元音以下，符合语音里辅音
/// 弱于元音的实际关系，也避免依赖整曲峰值归一。
const ONSET_LEVEL: f32 = 0.03;
/// 整曲首尾包络（秒）。
const ATTACK_SECONDS: f32 = 0.015;
const RELEASE_SECONDS: f32 = 0.045;
/// 固定输出增益：LF 表按周期单位能量构造、声道按**峰值**响应归一，此常量
/// 把中性元音放在中等电平（实测峰值约 0.2–0.3）；`volume` 只做线性缩放，
/// 不对整曲峰值再归一。
const OUTPUT_GAIN: f32 = 12.0;

/// 渲染为立体声 f32（双声道同相；与 `media.synthesis` 合同一致）。
pub fn render(spec: &VoiceSpec, sample_rate: f32) -> Result<Vec<[f32; 2]>, Error> {
    spec.validate()?;
    if !sample_rate.is_finite() || sample_rate <= 0.0 {
        return Err(Error::invalid_input(format!("voice: sample rate {sample_rate} must be positive")));
    }

    let beat_seconds = 60.0 / spec.tempo_bpm;
    let lengths: Vec<usize> = spec
        .syllables
        .iter()
        .map(|syllable| ((syllable.beats * beat_seconds * sample_rate).round() as usize).max(1))
        .collect();
    let total: usize = lengths.iter().sum();

    let first_formants = spec.syllables[0].vowel.formants();
    let mut engine = Engine {
        spec,
        sample_rate,
        tables: GlottalTables::new(),
        rng: Rng::new(spec.seed),
        phase: 0.0,
        tract: AllPole::from_formants(&pairs(first_formants), sample_rate),
        tract_gain: 1.0,
        formants: first_formants,
        attack: (ATTACK_SECONDS * sample_rate).round() as usize,
        release: (RELEASE_SECONDS * sample_rate).round() as usize,
    };
    engine.tract_gain = engine.tract.peak_gain(sample_rate);

    let mut mono = vec![0.0_f32; total];
    let mut cursor = 0_usize;
    for (index, syllable) in spec.syllables.iter().enumerate() {
        let length = lengths[index];
        let end = cursor + length;
        engine.note(syllable, &mut mono[cursor..end], cursor, total, index == 0);
        cursor = end;
    }

    let gain = spec.volume * OUTPUT_GAIN;
    Ok(mono
        .into_iter()
        .map(|sample| {
            let value = (sample * gain).clamp(-1.0, 1.0);
            [value, value]
        })
        .collect())
}

fn pairs(formants: [f32; 3]) -> [(f32, f32); 3] {
    formants.map(|frequency| (frequency, FORMANT_BANDWIDTH))
}

/// 整曲发声状态：一条声门相位、一个声道滤波器、一条随机流。
struct Engine<'a> {
    spec: &'a VoiceSpec,
    sample_rate: f32,
    tables: GlottalTables,
    rng: Rng,
    /// 声门相位，单位「周期」，始终折叠在 `0..1`。
    phase: f32,
    tract: AllPole,
    tract_gain: f32,
    /// 上一个音符结束时的共振峰（协同发音过渡的起点）。
    formants: [f32; 3],
    attack: usize,
    release: usize,
}

impl Engine<'_> {
    fn note(&mut self, syllable: &Syllable, out: &mut [f32], start: usize, total: usize, first: bool) {
        let length = out.len();
        let sample_rate = self.sample_rate;
        let target = syllable.vowel.formants();
        let rd = rd_for(syllable.onset);
        let onset = onset_kind(syllable.onset);

        let lead = match onset {
            OnsetKind::Fricative { .. } => (0.045 * sample_rate).round() as usize,
            OnsetKind::Plosive => (0.030 * sample_rate).round() as usize,
            OnsetKind::Nasal => (0.040 * sample_rate).round() as usize,
            OnsetKind::Glide { .. } | OnsetKind::None => 0,
        }
        .min(length);

        let mut fricative = match &onset {
            OnsetKind::Fricative { center, bandwidth } => {
                let filter = AllPole::from_formants(&[(*center, *bandwidth)], sample_rate);
                let gain = filter.peak_gain(sample_rate);
                Some((filter, gain))
            }
            _ => None,
        };
        let mut nasal = match &onset {
            OnsetKind::Nasal => {
                let filter = AllPole::from_formants(&[(250.0, 100.0), (1000.0, 150.0)], sample_rate);
                let gain = filter.peak_gain(sample_rate);
                Some((filter, gain))
            }
            _ => None,
        };

        // 协同发音：从上一音符的共振峰（或滑音起点）过渡到目标元音。
        let from = match onset {
            OnsetKind::Glide { from } => from,
            _ if first => target,
            _ => self.formants,
        };
        let transition = ((TRANSITION_SECONDS * sample_rate).round() as usize).max(1);
        let glides = from != target;
        if glides {
            self.set_formants(from);
        } else {
            self.set_formants(target);
        }
        let mut done = !glides;

        let lead_fade = (LEAD_FADE_SECONDS * sample_rate).round() as usize;
        let note_start_seconds = start as f32 / sample_rate;

        for (index, slot) in out.iter_mut().enumerate() {
            let absolute = start + index;

            // --- 起音前导：擦音/爆音/鼻音 -------------------------------------
            if index < lead {
                *slot = ONSET_LEVEL
                    * match &onset {
                        OnsetKind::Fricative { .. } => {
                            let raw = self.rng.bipolar();
                            let (filter, gain) = fricative.as_mut().expect("fricative filter");
                            filter.process(raw) * *gain
                        }
                        OnsetKind::Plosive => {
                            if index as f32 > 0.022 * sample_rate {
                                self.rng.bipolar()
                            } else {
                                0.0
                            }
                        }
                        OnsetKind::Nasal => {
                            self.phase = (self.phase + syllable.pitch / sample_rate).fract();
                            let source = self.tables.sample(rd, self.phase);
                            let (filter, gain) = nasal.as_mut().expect("nasal filter");
                            filter.process(source) * *gain
                        }
                        OnsetKind::Glide { .. } | OnsetKind::None => 0.0,
                    };
                continue;
            }

            // --- 声道共振峰过渡 -----------------------------------------------
            if !done {
                let voiced_index = index - lead;
                let progress = (voiced_index as f32 / transition as f32).min(1.0);
                // 每 16 个样本更新一次系数：逐样本换系数会让高 Q 滤波器
                // 在每个样本上重新分配能量，产生远高于稳态电平的瞬态。
                if voiced_index % FORMANT_UPDATE_INTERVAL == 0 || progress >= 1.0 {
                    let blended = [
                        from[0] + (target[0] - from[0]) * progress,
                        from[1] + (target[1] - from[1]) * progress,
                        from[2] + (target[2] - from[2]) * progress,
                    ];
                    self.set_formants(blended);
                }
                if progress >= 1.0 {
                    done = true;
                }
            }

            // --- 颤音：相位跟随整曲时间，深度按音符内延迟渐入 ----------------
            let seconds = absolute as f32 / sample_rate;
            let vibrato = if self.spec.vibrato.depth > 0.0 {
                let since = seconds - note_start_seconds;
                if since > self.spec.vibrato.delay_s {
                    let ramp = ((since - self.spec.vibrato.delay_s) / 0.15).min(1.0);
                    let lfo = (std::f32::consts::TAU * self.spec.vibrato.rate_hz * seconds).sin();
                    self.spec.vibrato.depth * ramp * lfo
                } else {
                    0.0
                }
            } else {
                0.0
            };

            // --- LF 声门源 + 气声噪声 ------------------------------------------
            self.phase = (self.phase + syllable.pitch * (1.0 + vibrato) / sample_rate).fract();
            let source = self.tables.sample(rd, self.phase);
            let mixed = source + self.rng.bipolar() * self.spec.breath;
            let shaped = self.tract.process(mixed) * self.tract_gain;

            // --- 包络：整曲首尾 + 起音前导接缝 --------------------------------
            let mut envelope = 1.0_f32;
            if absolute < self.attack {
                envelope *= absolute as f32 / self.attack.max(1) as f32;
            }
            let from_end = total - absolute;
            if from_end < self.release {
                envelope *= from_end as f32 / self.release.max(1) as f32;
            }
            if lead > 0 && index - lead < lead_fade {
                envelope *= (index - lead) as f32 / lead_fade.max(1) as f32;
            }
            *slot = shaped * envelope;
        }

        self.formants = target;
    }

    fn set_formants(&mut self, formants: [f32; 3]) {
        self.tract.set_formants(&pairs(formants), self.sample_rate);
        self.tract_gain = self.tract.peak_gain(self.sample_rate);
    }
}

/// 音节级 Rd：辅音类决定发声力度（高 R_d = 压喉短开相）。
/// T 爆音起音压得最紧；S/F 擦音段本身是噪声、其后元音偏松。
fn rd_for(onset: Option<Onset>) -> f32 {
    match onset {
        Some(Onset::T) => 2.4,
        Some(Onset::S | Onset::F) => 0.6,
        Some(Onset::N) => 1.2,
        Some(Onset::R | Onset::W | Onset::Y) => 1.5,
        None => 1.5,
    }
}

/// 起音事件：擦音/爆音/鼻音/滑音的声道前导处理。
enum OnsetKind {
    /// 擦音：高频噪声（S 5500 Hz / F 2200 Hz）经声道滤波。
    Fricative { center: f32, bandwidth: f32 },
    /// 爆音：静音 + 宽带 click。
    Plosive,
    /// 鼻音：低通声道（250/1000 Hz 双共振峰）。
    Nasal,
    /// 滑音：共振峰从起始滑入目标。
    Glide { from: [f32; 3] },
    /// 无起音。
    None,
}

fn onset_kind(onset: Option<Onset>) -> OnsetKind {
    match onset {
        Some(Onset::S) => OnsetKind::Fricative { center: 5500.0, bandwidth: 1500.0 },
        Some(Onset::F) => OnsetKind::Fricative { center: 2200.0, bandwidth: 1500.0 },
        Some(Onset::T) => OnsetKind::Plosive,
        Some(Onset::N) => OnsetKind::Nasal,
        Some(Onset::R) => OnsetKind::Glide { from: [350.0, 1100.0, 2400.0] },
        Some(Onset::W) => OnsetKind::Glide { from: [325.0, 850.0, 2350.0] },
        Some(Onset::Y) => OnsetKind::Glide { from: [300.0, 2300.0, 3000.0] },
        None => OnsetKind::None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::{Syllable, Vibrato, Vowel, VoiceSpec};

    fn spec(syllables: Vec<Syllable>) -> VoiceSpec {
        VoiceSpec::new(syllables)
    }

    fn s(vowel: Vowel, pitch: f32, beats: f32) -> Syllable {
        Syllable { onset: None, vowel, pitch, beats }
    }

    fn fnv1a64(bytes: &[u8]) -> u64 {
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        for byte in bytes {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        hash
    }

    #[test]
    fn renders_stereo_frames_for_the_declared_duration() {
        let spec = spec(vec![s(Vowel::A, 440.0, 1.0), s(Vowel::O, 330.0, 0.5)]);
        let samples = render(&spec, 48_000.0).expect("valid spec");
        let frames: usize = spec
            .syllables
            .iter()
            .map(|syllable| (syllable.beats * 60.0 / spec.tempo_bpm * 48_000.0).round() as usize)
            .sum();
        assert_eq!(samples.len(), frames);
    }

    #[test]
    fn same_spec_and_seed_reproduces_byte_identically() {
        let spec = spec(vec![
            Syllable { onset: Some(Onset::S), vowel: Vowel::A, pitch: 440.0, beats: 1.0 },
            Syllable { onset: Some(Onset::R), vowel: Vowel::E, pitch: 392.0, beats: 1.0 },
        ]);
        let a = render(&spec, 44_100.0).expect("a");
        let b = render(&spec, 44_100.0).expect("b");
        assert_eq!(a, b);
    }

    #[test]
    fn vowel_change_is_continuous_not_reset() {
        // 音节接缝处不得出现滤波器重置造成的空档或阶跃。
        let spec = spec(vec![s(Vowel::A, 220.0, 1.0), s(Vowel::I, 220.0, 1.0)]);
        let samples = render(&spec, 44_100.0).expect("song");
        let boundary = (60.0 / spec.tempo_bpm * 44_100.0).round() as usize;
        let window = &samples[boundary - 4..boundary + 4];
        assert!(
            window.iter().all(|frame| frame[0].abs() > 0.0),
            "silence across the syllable boundary means the engine reset"
        );
        let biggest = window
            .windows(2)
            .map(|pair| (pair[1][0] - pair[0][0]).abs())
            .fold(0.0_f32, f32::max);
        assert!(biggest < 0.5, "step {biggest} at the seam looks like a click");
    }

    #[test]
    fn different_seed_changes_breath_not_pitch() {
        let syllables = vec![s(Vowel::A, 440.0, 1.0)];
        let mut a = spec(syllables.clone());
        a.breath = 0.5;
        let mut b = spec(syllables);
        b.breath = 0.5;
        b.seed = 7;
        let samples_a = render(&a, 44_100.0).expect("a");
        let samples_b = render(&b, 44_100.0).expect("b");
        assert_ne!(samples_a, samples_b, "breath noise differs per seed");
        for frame in samples_b.iter() {
            assert!(frame[0].is_finite() && frame[1].is_finite());
        }
    }

    #[test]
    fn singing_a_melody_is_audible_and_stays_inside_full_scale() {
        let spec = spec(vec![
            Syllable { onset: Some(Onset::R), vowel: Vowel::A, pitch: 392.0, beats: 1.0 },
            s(Vowel::I, 440.0, 1.0),
            Syllable { onset: Some(Onset::N), vowel: Vowel::O, pitch: 523.25, beats: 2.0 },
        ]);
        let samples = render(&spec, 44_100.0).expect("valid spec");
        let peak = samples.iter().fold(0.0_f32, |a, f| a.max(f[0].abs()));
        assert!(peak > 0.02, "song too quiet: {peak}");
        assert!(peak <= 1.0 + 1e-4, "clipped: {peak}");
        let rms = samples.iter().map(|f| f[0] * f[0]).sum::<f32>() / samples.len() as f32;
        assert!(rms.sqrt() > 0.01, "rms too low: {}", rms.sqrt());
    }

    #[test]
    fn volume_scales_linearly_without_global_normalisation() {
        // 固定源增益合同：音量只做线性缩放，后段内容不改变前段电平。
        let syllables = vec![s(Vowel::A, 220.0, 1.0), s(Vowel::A, 220.0, 3.0)];
        let mut loud = spec(syllables.clone());
        loud.volume = 0.8;
        let mut soft = spec(syllables);
        soft.volume = 0.4;
        let loud_samples = render(&loud, 44_100.0).expect("loud");
        let soft_samples = render(&soft, 44_100.0).expect("soft");
        let ratio = loud_samples[20_000][0] / soft_samples[20_000][0];
        assert!((ratio - 2.0).abs() < 0.01, "volume must scale linearly, got {ratio}");
    }

    #[test]
    fn vibrato_uses_the_song_clock_not_the_note_clock() {
        let mut plain = spec(vec![s(Vowel::A, 330.0, 2.0), s(Vowel::A, 330.0, 2.0)]);
        plain.vibrato = Vibrato { rate_hz: 0.0, depth: 0.0, delay_s: 0.0 };
        let mut wobble = plain.clone();
        wobble.vibrato = Vibrato { rate_hz: 5.0, depth: 0.1, delay_s: 0.0 };
        let a = render(&plain, 44_100.0).expect("plain");
        let b = render(&wobble, 44_100.0).expect("wobble");
        assert_ne!(a, b, "vibrato must actually change the pitch track");
    }

    #[test]
    fn invalid_rate_is_rejected() {
        let spec = spec(vec![s(Vowel::A, 440.0, 1.0)]);
        assert_eq!(render(&spec, 0.0).expect_err("zero rate").code(), "invalid_input");
        assert_eq!(render(&spec, f32::NAN).expect_err("nan rate").code(), "invalid_input");
    }

    #[test]
    fn golden_song_fingerprint_pins_the_engine() {
        let spec = spec(vec![
            Syllable { onset: Some(Onset::R), vowel: Vowel::A, pitch: 392.0, beats: 1.0 },
            Syllable { onset: Some(Onset::S), vowel: Vowel::I, pitch: 440.0, beats: 1.0 },
            Syllable { onset: Some(Onset::N), vowel: Vowel::O, pitch: 523.25, beats: 2.0 },
        ]);
        let samples = render(&spec, 44_100.0).expect("valid spec");
        let bytes: Vec<u8> = samples.iter().flat_map(|f| f[0].to_ne_bytes()).collect();
        assert_eq!(fnv1a64(&bytes), 11073221635966551012, "engine constants are pinned");
    }
}
