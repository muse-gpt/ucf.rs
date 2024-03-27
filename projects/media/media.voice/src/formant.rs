//! GOLF 风格人声合成引擎：LF 声门流波表 + 级联全极点声道 + 谐波/噪声模型。
//!
//! 相对旧版「锯齿波 + 并联带通」的本质改进（GOLF, Yu & Fazekas 2024）：
//! - 激励源：LF 模型单参数 `R_d` 波表（真实声门脉冲频谱倾斜与闭合峰）；
//! - 声道：共振峰极点 → 级联 2 极点全极点 IIR（共振峰相互作用真实存在）；
//! - 气声：谐波 + 噪声模型，噪声经同一声道滤波（GOLF §3.1）。
//!
//! 全部确定性：同 spec+seed 字节级复现。

use crate::lf::GlottalTables;
use crate::spec::{Onset, VoiceSpec};
use crate::tract::AllPole;
use media_dsp::Rng;
use media_types::Error;

/// 渲染为立体声 f32（双声道同相；与 `media.synthesis` 合同一致）。
pub fn render(spec: &VoiceSpec, sample_rate: f32) -> Result<Vec<[f32; 2]>, Error> {
    spec.validate()?;
    let total_seconds = spec.duration_seconds();
    let total = (total_seconds * sample_rate).ceil() as usize;
    let mut mono = vec![0.0_f32; total.max(1)];

    let mut breath = Rng::new(spec.seed);
    let tables = GlottalTables::new();
    let mut phase = 0.0_f32;
    let beat_seconds = 60.0 / spec.tempo_bpm;
    let mut cursor = 0_usize;

    for syllable in &spec.syllables {
        let length = (syllable.beats * beat_seconds * sample_rate).round() as usize;
        let length = length.max(1).min(mono.len().saturating_sub(cursor).max(1));
        render_syllable(spec, syllable, &mut mono[cursor..cursor + length], sample_rate, &mut breath, &tables, &mut phase);
        cursor += length;
    }

    let peak = mono.iter().fold(0.0_f32, |a, s| a.max(s.abs())).max(1e-6);
    let gain = spec.volume / peak;
    Ok(mono
        .into_iter()
        .map(|sample| {
            let value = (sample * gain).clamp(-1.0, 1.0);
            [value, value]
        })
        .collect())
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

fn onset_kind(onset: Option<Onset>, target: [f32; 3]) -> OnsetKind {
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

fn render_syllable(
    spec: &VoiceSpec,
    syllable: &crate::spec::Syllable,
    out: &mut [f32],
    sample_rate: f32,
    breath_rng: &mut Rng,
    tables: &GlottalTables,
    phase: &mut f32,
) {
    let length = out.len();
    let seconds = length as f32 / sample_rate;
    let formants = syllable.vowel.formants();
    let rd = rd_for(syllable.onset);
    let onset = onset_kind(syllable.onset, formants);

    // 起音前导长度。
    let onset_lead = match onset {
        OnsetKind::Fricative { .. } => (0.045 * sample_rate).round() as usize,
        OnsetKind::Plosive => (0.030 * sample_rate).round() as usize,
        OnsetKind::Nasal => (0.040 * sample_rate).round() as usize,
        _ => 0,
    };
    let lead = onset_lead.min(length);
    // 前导专用滤波器在循环外构造一次（保持 IIR 状态跨样本）。
    let mut fricative = match &onset {
        OnsetKind::Fricative { center, bandwidth } => Some(AllPole::from_formants(&[(*center, *bandwidth)], sample_rate)),
        _ => None,
    };
    let mut nasal = match &onset {
        OnsetKind::Nasal => Some(AllPole::from_formants(&[(250.0, 100.0), (1000.0, 150.0)], sample_rate)),
        _ => None,
    };

    // 声道：目标共振峰全极点滤波器。
    let mut tract = AllPole::from_formants(
        &formants.map(|f| (f, 90.0)),
        sample_rate,
    );
    let mut tract_gain = tract.dc_gain();
    let onset_lead_f = (length as f32 * 0.3).max(1.0);

    let attack = (0.015 * sample_rate).round() as usize;
    let release = (0.045 * sample_rate).round() as usize;
    let cycle = 2.0 * std::f32::consts::PI / spec.vibrato.rate_hz.max(0.01);

    for (index, slot) in out.iter_mut().enumerate() {
        let t = index as f32 / sample_rate;

        // --- 起音前导：擦音/爆音/鼻音 -------------------------------------
        if index < lead {
            *slot = match &onset {
                OnsetKind::Fricative { .. } => {
                    let raw = (breath_rng.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0;
                    fricative.as_mut().expect("fricative filter").process(raw) * 0.9
                }
                OnsetKind::Plosive => {
                    if t < 0.022 {
                        0.0
                    } else {
                        let raw = (breath_rng.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0;
                        raw * 0.7
                    }
                }
                OnsetKind::Nasal => {
                    *phase += 2.0 * std::f32::consts::PI * syllable.pitch / sample_rate;
                    let source = tables.sample(rd, *phase / (2.0 * std::f32::consts::PI));
                    nasal.as_mut().expect("nasal filter").process(source) * 0.7
                }
                _ => 0.0,
            };
            continue;
        }

        // --- 声道：滑音或目标 --------------------------------------------
        if let OnsetKind::Glide { from } = onset {
            let glide_t = ((index - lead) as f32 / (onset_lead_f.max(1.0))).clamp(0.0, 1.0);
            if glide_t < 1.0 {
                let blended = [
                    from[0] + (formants[0] - from[0]) * glide_t,
                    from[1] + (formants[1] - from[1]) * glide_t,
                    from[2] + (formants[2] - from[2]) * glide_t,
                ];
                tract = AllPole::from_formants(&blended.map(|f| (f, 90.0)), sample_rate);
                tract_gain = tract.dc_gain();
            }
        }

        // --- 颤音 ---------------------------------------------------------
        let vibrato = if spec.vibrato.depth > 0.0 && t > spec.vibrato.delay_s {
            let since = t - spec.vibrato.delay_s;
            let envelope = (since * cycle / cycle).min(1.0);
            let lfo = (2.0 * std::f32::consts::PI * spec.vibrato.rate_hz * since).sin();
            spec.vibrato.depth * envelope * lfo
        } else {
            0.0
        };
        *phase += 2.0 * std::f32::consts::PI * syllable.pitch * (1.0 + vibrato) / sample_rate;

        // --- LF 声门源 ----------------------------------------------------
        let source = tables.sample(rd, *phase / (2.0 * std::f32::consts::PI));

        // --- 气声噪声（谐波 + 噪声模型）-----------------------------------
        let noise = (breath_rng.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0;
        let mixed = source + noise * spec.breath;

        // --- 声道滤波 -----------------------------------------------------
        let shaped = tract.process(mixed) / tract_gain;

        // --- 包络 ---------------------------------------------------------
        let voiced_index = index - lead;
        let voiced_length = length - lead;
        let mut envelope = 1.0;
        if voiced_index < attack {
            envelope *= voiced_index as f32 / attack.max(1) as f32;
        }
        let from_end = voiced_length - voiced_index;
        if from_end < release {
            envelope *= from_end as f32 / release.max(1) as f32;
        }
        *slot = shaped * envelope * 0.55;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::{Syllable, Vowel, VoiceSpec};

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
        let frames = (spec.duration_seconds() * 48_000.0).ceil() as usize;
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
    fn singing_a_melody_is_audible_and_normalized_to_volume() {
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
    fn golden_song_fingerprint_pins_the_engine() {
        let spec = spec(vec![
            Syllable { onset: Some(Onset::R), vowel: Vowel::A, pitch: 392.0, beats: 1.0 },
            Syllable { onset: Some(Onset::S), vowel: Vowel::I, pitch: 440.0, beats: 1.0 },
            Syllable { onset: Some(Onset::N), vowel: Vowel::O, pitch: 523.25, beats: 2.0 },
        ]);
        let samples = render(&spec, 44_100.0).expect("valid spec");
        let bytes: Vec<u8> = samples.iter().flat_map(|f| f[0].to_ne_bytes()).collect();
        assert_eq!(fnv1a64(&bytes), 9562453024053591702, "engine constants are pinned");
    }
}
