//! 节奏型：把鼓组编排成可循环的 step 序列。
//!
//! 一个节奏型是纯数据：BPM、每小节步数（16 = 十六分音符）、小节数与若干
//! 音轨；每个音轨是一串力度值（0 = 休止）。渲染时每一步调用一次
//! [`render_instrument`]，所以节奏型与单次乐器发声共享同一套合成实现、同一
//! 条随机流与同一条确定性合同。

use media_types::Error;

use crate::instruments::{Instrument, InstrumentSpec, render_instrument};

/// 一条节奏型音轨。
#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct PatternTrack {
    pub instrument: Instrument,
    /// MIDI 音高；鼓组忽略（但保留以便统一数据形状）。
    #[serde(default = "default_track_note")]
    pub note: u8,
    /// 每步力度 0..1，0 表示休止。
    pub steps: Vec<f32>,
    #[serde(default = "half")]
    pub tone: f32,
    #[serde(default = "half")]
    pub decay: f32,
    #[serde(default)]
    pub pan: f32,
}

/// 一个可循环的节奏型。
#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct DrumPattern {
    pub bpm: f32,
    #[serde(default = "default_steps_per_bar")]
    pub steps_per_bar: u32,
    #[serde(default = "default_bars")]
    pub bars: u32,
    pub tracks: Vec<PatternTrack>,
}

fn half() -> f32 {
    0.5
}

fn default_track_note() -> u8 {
    60
}

fn default_steps_per_bar() -> u32 {
    16
}

fn default_bars() -> u32 {
    1
}

impl PatternTrack {
    /// 由稀疏的 (步, 力度) 列表构造一整小节音轨。
    #[must_use]
    pub fn sparse(instrument: Instrument, steps_per_bar: usize, hits: &[(usize, f32)]) -> Self {
        let mut steps = vec![0.0_f32; steps_per_bar];
        for (index, velocity) in hits {
            if let Some(slot) = steps.get_mut(*index) {
                *slot = *velocity;
            }
        }
        Self { instrument, note: 60, steps, tone: half(), decay: half(), pan: 0.0 }
    }

    /// 构造一个铺满每一步的音轨（用于踩镲）。
    #[must_use]
    pub fn every(instrument: Instrument, steps_per_bar: usize, stride: usize, velocity: f32) -> Self {
        let mut steps = vec![0.0_f32; steps_per_bar];
        for index in (0..steps_per_bar).step_by(stride.max(1)) {
            steps[index] = velocity;
        }
        Self { instrument, note: 60, steps, tone: half(), decay: half(), pan: 0.0 }
    }

    fn with(mut self, note: u8, tone: f32, decay: f32, pan: f32) -> Self {
        self.note = note;
        self.tone = tone;
        self.decay = decay;
        self.pan = pan;
        self
    }
}

impl DrumPattern {
    /// 总步数。
    #[must_use]
    pub fn total_steps(&self) -> usize {
        self.steps_per_bar.max(1) as usize * self.bars.max(1) as usize
    }

    /// 一小节时长（秒，按 4/4）。
    #[must_use]
    pub fn bar_seconds(&self) -> f32 {
        240.0 / self.bpm.max(1.0)
    }

    /// 总时长（秒）。
    #[must_use]
    pub fn duration_seconds(&self) -> f32 {
        self.bar_seconds() * self.bars.max(1) as f32
    }

    /// 参数范围与数据形状合同。
    ///
    /// # Errors
    /// BPM、步数、小节数、音轨为空、力度越界或步数与总步数不符均为
    /// `invalid_input`。
    pub fn validate(&self) -> Result<(), Error> {
        if !self.bpm.is_finite() || !(20.0..=300.0).contains(&self.bpm) {
            return Err(Error::invalid_input(format!("pattern bpm {} must be within 20..=300", self.bpm)));
        }
        if !(1..=32).contains(&self.steps_per_bar) {
            return Err(Error::invalid_input(format!(
                "pattern steps per bar {} must be within 1..=32",
                self.steps_per_bar
            )));
        }
        if !(1..=32).contains(&self.bars) {
            return Err(Error::invalid_input(format!("pattern bars {} must be within 1..=32", self.bars)));
        }
        if self.tracks.is_empty() {
            return Err(Error::invalid_input("pattern must have at least one track"));
        }
        let total = self.total_steps();
        for (index, track) in self.tracks.iter().enumerate() {
            if track.steps.len() > total {
                return Err(Error::invalid_input(format!(
                    "pattern track {index} has {} steps, more than the {total} contracted steps",
                    track.steps.len()
                )));
            }
            for (step, velocity) in track.steps.iter().enumerate() {
                if !velocity.is_finite() || !(0.0..=1.0).contains(velocity) {
                    return Err(Error::invalid_input(format!(
                        "pattern track {index} step {step} velocity {velocity} must be within 0..=1"
                    )));
                }
            }
            for (name, value) in [("tone", track.tone), ("decay", track.decay)] {
                if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                    return Err(Error::invalid_input(format!(
                        "pattern track {index} {name} {value} must be within 0..=1"
                    )));
                }
            }
            if !track.pan.is_finite() {
                return Err(Error::invalid_input(format!("pattern track {index} pan must be finite")));
            }
        }
        Ok(())
    }

    /// 按步展开为全部音符（供宿主检查或编辑，不参与渲染）。
    #[must_use]
    pub fn events(&self) -> Vec<PatternEvent> {
        let step_seconds = self.bar_seconds() / self.steps_per_bar.max(1) as f32;
        let mut events = Vec::new();
        for (track_index, track) in self.tracks.iter().enumerate() {
            for (step, velocity) in track.steps.iter().enumerate() {
                if *velocity > 0.0 {
                    events.push(PatternEvent {
                        step,
                        at_seconds: step as f32 * step_seconds,
                        track: track_index,
                        instrument: track.instrument,
                        note: track.note,
                        velocity: *velocity,
                    });
                }
            }
        }
        events.sort_by(|a, b| a.step.cmp(&b.step).then(a.track.cmp(&b.track)));
        events
    }
}

/// 展开后的单个音符事件。
#[derive(Clone, Copy, Debug)]
pub struct PatternEvent {
    pub step: usize,
    pub at_seconds: f32,
    pub track: usize,
    pub instrument: Instrument,
    pub note: u8,
    pub velocity: f32,
}

/// 渲染节奏型为立体声 f32。
///
/// # Errors
/// 数据形状或参数越界为 `invalid_input`；非法采样率同样拒绝。
pub fn render_pattern(pattern: &DrumPattern, sample_rate: f32) -> Result<Vec<[f32; 2]>, Error> {
    pattern.validate()?;
    if !sample_rate.is_finite() || sample_rate <= 0.0 {
        return Err(Error::invalid_input(format!("pattern sample rate {sample_rate} must be finite and positive")));
    }
    let step_seconds = pattern.bar_seconds() / pattern.steps_per_bar as f32;
    let frames = (pattern.duration_seconds() * sample_rate).ceil() as usize;
    let mut out = vec![[0.0_f32; 2]; frames];

    for (track_index, track) in pattern.tracks.iter().enumerate() {
        for (step, velocity) in track.steps.iter().enumerate() {
            if *velocity <= 0.0 {
                continue;
            }
            let duration = if track.instrument.is_pitched() {
                (step_seconds * 0.9).max(0.06)
            } else {
                track.instrument.natural_duration()
            };
            let spec = InstrumentSpec {
                instrument: track.instrument,
                seed: 1 + track_index as u32 * 101 + step as u32,
                note: track.note,
                duration: duration.min(8.0),
                decay: track.decay,
                tone: track.tone,
                gain: *velocity,
                pan: track.pan,
            };
            let hit = render_instrument(&spec, sample_rate)?;
            let start = (step as f32 * step_seconds * sample_rate).round() as usize;
            for (offset, frame) in hit.iter().enumerate() {
                let at = start + offset;
                if at >= out.len() {
                    break;
                }
                out[at][0] = (out[at][0] + frame[0]).clamp(-1.0, 1.0);
                out[at][1] = (out[at][1] + frame[1]).clamp(-1.0, 1.0);
            }
        }
    }
    Ok(out)
}

/// 命名节奏型预设。
pub struct PatternPreset {
    pub name: &'static str,
    pub description: &'static str,
    pub pattern: DrumPattern,
}

fn pattern(bpm: f32, tracks: Vec<PatternTrack>) -> DrumPattern {
    DrumPattern { bpm, steps_per_bar: 16, bars: 1, tracks }
}

/// 常用鼓点预设（每小节 16 步，一小节一循环）。
pub fn pattern_presets() -> Vec<PatternPreset> {
    vec![
        PatternPreset {
            name: "four-on-floor",
            description: "House 四拍底鼓 + 2/4 拍手 + 十六分踩镲",
            pattern: pattern(
                124.0,
                vec![
                    PatternTrack::every(Instrument::Kick, 16, 4, 1.0).with(36, 0.5, 0.2, 0.0),
                    PatternTrack::sparse(Instrument::Clap, 16, &[(4, 0.9), (12, 0.9)]).with(60, 0.55, 0.5, 0.0),
                    PatternTrack::every(Instrument::HatClosed, 16, 2, 0.5).with(60, 0.6, 0.35, 0.2),
                    PatternTrack::sparse(Instrument::HatOpen, 16, &[(14, 0.45)]).with(60, 0.5, 0.6, -0.2),
                ],
            ),
        },
        PatternPreset {
            name: "breakbeat",
            description: "放克碎拍：反拍军鼓 + 切分底鼓",
            pattern: pattern(
                130.0,
                vec![
                    PatternTrack::sparse(Instrument::Kick, 16, &[(0, 1.0), (6, 0.85), (10, 0.9)]).with(36, 0.5, 0.2, 0.0),
                    PatternTrack::sparse(Instrument::Snare, 16, &[(4, 0.95), (12, 0.95)]).with(60, 0.7, 0.35, 0.0),
                    PatternTrack::every(Instrument::HatClosed, 16, 2, 0.45).with(60, 0.6, 0.35, 0.2),
                ],
            ),
        },
        PatternPreset {
            name: "trap",
            description: "Trap：稀疏底鼓 + 8 步军鼓 + 密集踩镲",
            pattern: pattern(
                140.0,
                vec![
                    PatternTrack::sparse(Instrument::Kick, 16, &[(0, 1.0), (6, 0.9), (10, 0.8)]).with(36, 0.4, 0.3, 0.0),
                    PatternTrack::sparse(Instrument::Snare, 16, &[(8, 0.9)]).with(60, 0.6, 0.4, 0.0),
                    PatternTrack::every(Instrument::HatClosed, 16, 1, 0.32).with(60, 0.7, 0.3, 0.25),
                ],
            ),
        },
        PatternPreset {
            name: "rock",
            description: "摇滚 2/4 底鼓 + 2/4 军鼓 + 八分踩镲 + 吊镲起头",
            pattern: pattern(
                110.0,
                vec![
                    PatternTrack::sparse(Instrument::Kick, 16, &[(0, 1.0), (8, 0.9)]).with(36, 0.5, 0.25, 0.0),
                    PatternTrack::sparse(Instrument::Snare, 16, &[(4, 0.95), (12, 0.95)]).with(60, 0.65, 0.4, 0.0),
                    PatternTrack::every(Instrument::HatClosed, 16, 2, 0.5).with(60, 0.55, 0.35, 0.2),
                    PatternTrack::sparse(Instrument::Crash, 16, &[(0, 0.5)]).with(60, 0.7, 0.6, -0.15),
                ],
            ),
        },
        PatternPreset {
            name: "half-time",
            description: "半速：底鼓在 1 与 3 拍后、军鼓只在第 3 拍",
            pattern: pattern(
                90.0,
                vec![
                    PatternTrack::sparse(Instrument::Kick, 16, &[(0, 1.0), (10, 0.85)]).with(36, 0.5, 0.25, 0.0),
                    PatternTrack::sparse(Instrument::Snare, 16, &[(8, 1.0)]).with(60, 0.6, 0.45, 0.0),
                    PatternTrack::every(Instrument::HatClosed, 16, 4, 0.4).with(60, 0.5, 0.4, 0.2),
                ],
            ),
        },
        PatternPreset {
            name: "bossa",
            description: "Bossa：切分底鼓 + 边击 + 沙锤",
            pattern: pattern(
                100.0,
                vec![
                    PatternTrack::sparse(Instrument::Kick, 16, &[(0, 0.9), (6, 0.7), (8, 0.8), (14, 0.6)]).with(36, 0.45, 0.3, 0.0),
                    PatternTrack::sparse(Instrument::Rim, 16, &[(4, 0.7), (12, 0.7)]).with(60, 0.55, 0.35, 0.1),
                    PatternTrack::every(Instrument::Shaker, 16, 2, 0.35).with(60, 0.5, 0.4, 0.3),
                ],
            ),
        },
        PatternPreset {
            name: "lofi",
            description: "Lo-fi：迟滞的底鼓与军鼓、轻踩镲",
            pattern: pattern(
                85.0,
                vec![
                    PatternTrack::sparse(Instrument::Kick, 16, &[(0, 0.9), (7, 0.75), (10, 0.7)]).with(36, 0.4, 0.35, 0.0),
                    PatternTrack::sparse(Instrument::Snare, 16, &[(4, 0.7), (12, 0.75)]).with(60, 0.35, 0.55, 0.0),
                    PatternTrack::every(Instrument::HatClosed, 16, 2, 0.3).with(60, 0.4, 0.45, 0.2),
                    PatternTrack::sparse(Instrument::Rim, 16, &[(15, 0.4)]).with(60, 0.45, 0.3, -0.1),
                ],
            ),
        },
        PatternPreset {
            name: "dnb",
            description: "Drum & Bass：2/4 军鼓 + 十六分踩镲",
            pattern: pattern(
                174.0,
                vec![
                    PatternTrack::sparse(Instrument::Kick, 16, &[(0, 1.0), (10, 0.9)]).with(36, 0.5, 0.2, 0.0),
                    PatternTrack::sparse(Instrument::Snare, 16, &[(4, 0.95), (12, 0.95), (13, 0.5)]).with(60, 0.7, 0.35, 0.0),
                    PatternTrack::every(Instrument::HatClosed, 16, 2, 0.4).with(60, 0.65, 0.3, 0.2),
                    PatternTrack::sparse(Instrument::HatClosed, 16, &[(6, 0.25), (14, 0.25)]).with(60, 0.65, 0.3, 0.25),
                ],
            ),
        },
    ]
}

/// 按名字取节奏型预设。
#[must_use]
pub fn pattern_preset(name: &str) -> Option<DrumPattern> {
    pattern_presets().into_iter().find(|preset| preset.name == name).map(|preset| preset.pattern)
}

/// 全部节奏型预设名。
#[must_use]
pub fn pattern_preset_names() -> Vec<&'static str> {
    pattern_presets().into_iter().map(|preset| preset.name).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_are_unique_named_and_valid() {
        let presets = pattern_presets();
        let names: Vec<_> = presets.iter().map(|preset| preset.name).collect();
        let unique: std::collections::BTreeSet<_> = names.iter().collect();
        assert_eq!(unique.len(), names.len());
        for preset in &presets {
            preset.pattern.validate().expect("every shipped groove passes the contract");
            assert_eq!(pattern_preset(preset.name).as_ref().map(|p| p.total_steps()), Some(16));
            assert!(preset.pattern.tracks.len() >= 2);
        }
        assert!(pattern_preset("no-such-groove").is_none());
        assert_eq!(pattern_preset_names().len(), presets.len());
    }

    #[test]
    fn renders_one_bar_of_audible_deterministic_audio() {
        let pattern = pattern_preset("four-on-floor").expect("preset");
        let samples = render_pattern(&pattern, 44_100.0).expect("render");
        assert_eq!(samples.len(), (pattern.duration_seconds() * 44_100.0).ceil() as usize);
        let peak = samples.iter().fold(0.0_f32, |peak, frame| peak.max(frame[0].abs()));
        assert!(peak > 0.05, "groove is silent (peak {peak})");
        assert!(samples.iter().all(|frame| frame[0].is_finite()));
        let again = render_pattern(&pattern, 44_100.0).expect("render again");
        assert_eq!(samples, again, "patterns must be deterministic");
    }

    #[test]
    fn events_are_sorted_and_match_the_contracted_steps() {
        let pattern = pattern_preset("rock").expect("preset");
        let events = pattern.events();
        assert!(!events.is_empty());
        assert!(events.windows(2).all(|pair| pair[0].step <= pair[1].step));
        assert!(events.iter().all(|event| event.at_seconds >= 0.0 && event.velocity > 0.0));
        let step_seconds = pattern.bar_seconds() / 16.0;
        let first = &events[0];
        assert!((first.at_seconds - first.step as f32 * step_seconds).abs() < 1e-5);
    }

    #[test]
    fn contract_rejects_bad_shapes() {
        let mut bad = pattern_preset("rock").expect("preset");
        bad.bpm = 5.0;
        assert_eq!(render_pattern(&bad, 44_100.0).expect_err("bpm").code(), "invalid_input");

        let mut bad = pattern_preset("rock").expect("preset");
        bad.tracks[0].steps[0] = 1.5;
        assert_eq!(render_pattern(&bad, 44_100.0).expect_err("velocity").code(), "invalid_input");

        let mut bad = pattern_preset("rock").expect("preset");
        bad.tracks.clear();
        assert_eq!(render_pattern(&bad, 44_100.0).expect_err("empty").code(), "invalid_input");

        assert_eq!(
            render_pattern(&pattern_preset("rock").expect("preset"), 0.0).expect_err("rate").code(),
            "invalid_input"
        );
    }

    #[test]
    fn pitched_tracks_follow_their_note() {
        let pattern = DrumPattern {
            bpm: 120.0,
            steps_per_bar: 16,
            bars: 1,
            tracks: vec![PatternTrack::sparse(Instrument::Bass, 16, &[(0, 0.9), (8, 0.9)]).with(36, 0.4, 0.3, 0.0)],
        };
        let samples = render_pattern(&pattern, 44_100.0).expect("bass groove");
        let peak = samples.iter().fold(0.0_f32, |peak, frame| peak.max(frame[0].abs()));
        assert!(peak > 0.05, "bass line is silent (peak {peak})");
    }
}
