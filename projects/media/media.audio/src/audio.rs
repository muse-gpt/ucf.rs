//! Audio track planning and mixing.
//!
//! The mixer consumes a neutral [`AudioPlan`]: an explicit sample rate,
//! channel count, duration and placements in project frames. It knows nothing
//! about scenes — the scene audio bed IR is lowered into a plan by the caller.
//!
//! Sound effects are rendered by `media.synthesis` (the single formal synth
//! core, OpenSFX semantics) and mixed into the bed; the mixer owns none of
//! the voice algorithms. The BGM pad and plucks stay here as orchestration,
//! and every random choice in this crate consumes the same `media.dsp::Rng`
//! stream as the voices — one formal random contract for all sound paths.
//! Narration is a reserved slot: if the plan points at a PCM WAV file it is
//! decoded and mixed in; otherwise the slot stays silent. Sample rate and
//! channels are input contracts: nothing is resampled implicitly between
//! synthesis and the written buffer. An unknown `sfx` kind is an explicit
//! `invalid_input` error, never a silent skip.

use media_codec::DecodeLimits;
use media_codec::MediaLibrary;
use media_codec::parse_wav_pcm16;
use media_dsp::Rng;
use media_synthesis::{Family, SfxSpec, render};
use media_types::Error;
use media_types::TimeBase;

/// One synthesized effect placed on the track.
///
/// `kind` is a canonical family name resolved by `media_synthesis::Family::parse`.
/// The scene IR carries no per-effect duration, pitch or seed, so placements
/// render with the unified defaults: [`SfxSpec::new`] at the per-family
/// placement length below. The same kind therefore sounds the same at every
/// position — placement-derived seeds are deliberately not a thing.
#[derive(Clone, Debug)]
pub struct SfxPlacement {
    pub kind: String,
    pub at_frames: u32,
    pub volume: f32,
}

/// Placement length in seconds per family when the plan carries no duration.
/// The six values the old built-ins used are preserved so existing films keep
/// their timing envelope; `ambience` and `sparkle` match their catalogue intent.
#[must_use]
pub fn sfx_placement_duration(family: Family) -> f32 {
    match family {
        Family::Whoosh => 0.55,
        Family::Impact => 0.35,
        Family::Riser => 1.6,
        Family::Tick => 0.06,
        Family::Glitch => 0.14,
        Family::Sub => 0.9,
        Family::Ambience => 8.0,
        Family::Sparkle => 1.6,
    }
}

/// One imported audio file placed at `at_frames`, playing from `offset_frames`.
#[derive(Clone, Debug)]
pub struct ClipPlacement {
    pub src: String,
    pub at_frames: u32,
    /// Length in frames; `None` means "until the file ends".
    pub duration_frames: Option<u32>,
    /// Source offset in frames skipped before playback starts.
    pub offset_frames: u32,
    pub volume: f32,
    /// Linear fade-in over the first N placed frames (0 means no fade).
    pub fade_in_frames: u32,
    /// Linear fade-out over the last N placed frames (0 means no fade).
    pub fade_out_frames: u32,
    /// Project-declared contract slot: a missing or undecodable source is an
    /// explicit error. Reserved scene beds keep `false` (documented silence).
    pub required: bool,
}

/// The reserved narration slot: a PCM16 WAV mixed in at a frame offset.
#[derive(Clone, Debug)]
pub struct NarrationPlacement {
    pub wav: String,
    pub delay_frames: u32,
    pub volume: f32,
}

/// Background music bed parameters.
#[derive(Clone, Debug)]
pub struct BgmPlan {
    pub volume: f32,
    /// Reserved for future moods; only `pad` exists today.
    pub mood: String,
}

/// Neutral audio track plan: the mixer's only input contract.
///
/// Placements are expressed in project frames, the output format in explicit
/// sample rate and channels, and the bed length in frames. The plan never
/// carries seconds: that value is derived through [`TimeBase`], so it cannot
/// contradict the other time fields.
#[derive(Clone, Debug)]
pub struct AudioPlan {
    pub sample_rate: u32,
    pub channels: u16,
    pub time_base: TimeBase,
    pub duration_frames: u32,
    pub bgm: Option<BgmPlan>,
    pub narration: Option<NarrationPlacement>,
    pub sfx: Vec<SfxPlacement>,
    pub clips: Vec<ClipPlacement>,
}

/// Builds the full audio bed: synthesized BGM and effects, the reserved
/// narration slot, and every imported audio clip placed on the timeline.
///
/// `limits` bounds what one decode of an imported clip may allocate; the
/// project layer passes them explicitly to the codec.
pub fn build_audio(plan: &AudioPlan, media: &mut MediaLibrary, limits: DecodeLimits) -> Result<Vec<i16>, Error> {
    if plan.channels != 2 {
        return Err(Error::invalid_input(format!("the mixer is stereo-only, got {} channels", plan.channels)));
    }
    if plan.sample_rate == 0 || plan.time_base.numerator() == 0 {
        return Err(Error::invalid_input("sample_rate and the time base must be positive"));
    }
    let frames = plan.time_base.to_samples_ceil(plan.duration_frames, plan.sample_rate);
    let mut buf = vec![0f32; frames * 2];

    if let Some(bgm) = &plan.bgm {
        synth_pad(&mut buf, bgm.volume, plan.sample_rate);
        synth_plucks(&mut buf, bgm.volume * 0.5, plan.sample_rate);
    }
    for sfx in &plan.sfx {
        let family = Family::parse(&sfx.kind).ok_or_else(|| Error::invalid_input(format!("unknown sfx kind {}", sfx.kind)))?;
        let spec = SfxSpec::new(family, sfx_placement_duration(family));
        let samples = render(&spec, plan.sample_rate as f32)?;
        let start = plan.time_base.to_samples(sfx.at_frames, plan.sample_rate);
        let total = frames_len(&buf);
        for (i, frame) in samples.iter().enumerate() {
            let at = start + i;
            if at >= total {
                break;
            }
            buf[at * 2] += frame[0] * sfx.volume;
            buf[at * 2 + 1] += frame[1] * sfx.volume;
        }
    }
    if let Some(narration) = &plan.narration {
        mix_wav(&mut buf, &narration.wav, narration.delay_frames, narration.volume, plan)?;
    }
    for clip in &plan.clips {
        let samples = match media.audio_pcm(&clip.src, plan.sample_rate, limits) {
            Ok(samples) => samples,
            Err(error) => {
                // Optional (reserved) imports must not abort the render: the
                // clip stays silent and the rest of the film still compiles.
                // A required slot is a declared contract — fail explicitly.
                if clip.required {
                    return Err(Error::invalid_input(format!(
                        "required audio clip `{}` cannot be decoded: {error}",
                        clip.src
                    )));
                }
                continue;
            }
        };
        let start = plan.time_base.to_samples(clip.at_frames, plan.sample_rate);
        let offset = plan.time_base.to_samples(clip.offset_frames, plan.sample_rate);
        let limit = clip.duration_frames.map(|d| plan.time_base.to_samples(d, plan.sample_rate)).unwrap_or(usize::MAX);
        let total = frames_len(&buf);
        let placed = limit.min(samples.len().saturating_sub(offset)).min(total.saturating_sub(start));
        for i in 0..placed {
            let gain = clip.volume * clip_gain(i, placed, clip.fade_in_frames as usize, clip.fade_out_frames as usize);
            let sample = &samples[offset + i];
            buf[(start + i) * 2] += sample[0] * gain;
            buf[(start + i) * 2 + 1] += sample[1] * gain;
        }
    }

    Ok(buf.into_iter().map(|s| (s.clamp(-1.0, 1.0) * 32767.0) as i16).collect())
}

fn frames_len(buf: &[f32]) -> usize {
    buf.len() / 2
}

/// Linear fade gain for placed sample `i` of a `placed`-long region.
///
/// The ramps run over the first `fade_in` and last `fade_out` frames of the
/// placement; a fade longer than the region simply never reaches unity.
#[must_use]
fn clip_gain(i: usize, placed: usize, fade_in: usize, fade_out: usize) -> f32 {
    let mut gain = 1.0f32;
    if fade_in > 0 {
        gain *= (i as f32 / fade_in as f32).min(1.0);
    }
    if fade_out > 0 {
        gain *= (((placed - 1 - i) as f32) / fade_out as f32).clamp(0.0, 1.0);
    }
    gain
}

/// Slow chord pad: four chords cycling, two detuned partials each.
fn synth_pad(buf: &mut [f32], volume: f32, sample_rate: u32) {
    // A minor bed: Am, F, C, G — root frequencies.
    let chords = [[110.0, 164.81, 220.0], [87.31, 130.81, 174.61], [130.81, 164.81, 196.0], [98.0, 123.47, 146.83]];
    let chord_seconds = 8.0;
    for i in 0..frames_len(buf) {
        let t = i as f64 / sample_rate as f64;
        let chord = &chords[((t / chord_seconds) as usize) % chords.len()];
        let phase_in_chord = (t % chord_seconds) / chord_seconds;
        let env = (phase_in_chord * std::f64::consts::PI).sin().powf(0.6);
        let mut s = 0.0f64;
        for f in chord {
            let w = *f * t * std::f64::consts::TAU;
            s += w.sin() * 0.5 + (w * 2.0).sin() * 0.12;
            let detuned = (*f * 1.003) * t * std::f64::consts::TAU;
            s += detuned.sin() * 0.35;
        }
        let value = (s / (chord.len() as f64 * 1.2)) * env * volume as f64;
        // Gentle stereo width via alternating polarity of the detune.
        buf[i * 2] += value as f32;
        buf[i * 2 + 1] += value as f32 * 0.92;
    }
}

/// Sparse pentatonic plucks seeded deterministically.
///
/// Each slot is a pure function of its index through the shared
/// `media.dsp::Rng` stream — the same contract the synthesis voices use, so
/// the bed has exactly one formal source of randomness.
fn synth_plucks(buf: &mut [f32], volume: f32, sample_rate: u32) {
    let scale = [440.0, 523.25, 587.33, 659.25, 783.99];
    let total = frames_len(buf);
    let mut slot = 0usize;
    let mut cursor = (sample_rate as usize) * 3; // start after 3s
    while cursor < total {
        let note = scale[(Rng::new(slot as u32).next_u32() % scale.len() as u32) as usize];
        let decay_seconds = 1.6;
        let len = ((decay_seconds * sample_rate as f64) as usize).min(total - cursor);
        for i in 0..len {
            let t = i as f64 / sample_rate as f64;
            let env = (-t * 3.2).exp();
            let s = (note * t * std::f64::consts::TAU).sin() * 0.7 + (note * 2.0 * t * std::f64::consts::TAU).sin() * 0.2;
            let v = (s * env * volume as f64 * 0.5) as f32;
            buf[(cursor + i) * 2] += v;
            buf[(cursor + i) * 2 + 1] += v * 0.9;
        }
        slot += 1;
        let gap = 2.0 + (Rng::new(slot as u32 * 7 + 13).next_u32() % 40) as f64 / 10.0;
        cursor += (gap * sample_rate as f64) as usize;
    }
}

/// Decodes a PCM16 WAV file and mixes it at the given frame offset.
///
/// A missing or undecodable file keeps the slot silent; a file whose declared
/// sample rate differs from the plan is an input-contract violation, never an
/// implicit resample.
fn mix_wav(buf: &mut [f32], path: &str, delay_frames: u32, volume: f32, plan: &AudioPlan) -> Result<(), Error> {
    let Ok(bytes) = std::fs::read(path) else {
        // The narration slot is optional input; a missing file stays silent.
        return Ok(());
    };
    let Some(wav) = parse_wav_pcm16(&bytes) else {
        return Ok(());
    };
    if wav.sample_rate != plan.sample_rate {
        return Err(Error::invalid_input(format!(
            "narration {path} is {} Hz but the plan is {} Hz, no implicit resample",
            wav.sample_rate, plan.sample_rate
        )));
    }
    let start = plan.time_base.to_samples(delay_frames, plan.sample_rate);
    let total = frames_len(buf);
    for (i, frame) in wav.frames.iter().enumerate() {
        let at = start + i;
        if at >= total {
            break;
        }
        buf[at * 2] += frame[0] * volume;
        buf[at * 2 + 1] += frame[1] * volume;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan(sfx: Vec<SfxPlacement>) -> AudioPlan {
        AudioPlan {
            sample_rate: 44_100,
            channels: 2,
            time_base: TimeBase::from_fps(30),
            duration_frames: 120,
            bgm: None,
            narration: None,
            sfx,
            clips: Vec::new(),
        }
    }

    #[test]
    fn placed_effects_are_audible() {
        let bed = build_audio(
            &plan(vec![SfxPlacement { kind: "whoosh".into(), at_frames: 10, volume: 0.5 }]),
            &mut MediaLibrary::new(),
            DecodeLimits::unbounded(),
        )
        .expect("bed");
        let start = 10 * 44_100 / 30;
        let region = &bed[start * 2..(start + 44_100 / 2) * 2];
        let peak = region.iter().fold(0i16, |a, s| a.max(s.abs()));
        assert!(peak > 100, "whoosh region too quiet: {peak}");
    }

    #[test]
    fn same_kind_sounds_identical_at_any_position() {
        let a = build_audio(
            &plan(vec![SfxPlacement { kind: "impact".into(), at_frames: 0, volume: 0.5 }]),
            &mut MediaLibrary::new(),
            DecodeLimits::unbounded(),
        )
        .expect("bed a");
        let b = build_audio(
            &plan(vec![SfxPlacement { kind: "impact".into(), at_frames: 37, volume: 0.5 }]),
            &mut MediaLibrary::new(),
            DecodeLimits::unbounded(),
        )
        .expect("bed b");
        let start_a = 0;
        let start_b = 37 * 44_100 / 30;
        let len = (sfx_placement_duration(Family::Impact) * 44_100.0) as usize - 100;
        let region_a = &a[start_a * 2..(start_a + len) * 2];
        let region_b = &b[start_b * 2..(start_b + len) * 2];
        assert_eq!(region_a, region_b, "placement position must not change the sound");
    }

    fn clip(src: &str, required: bool) -> ClipPlacement {
        ClipPlacement {
            src: src.into(),
            at_frames: 0,
            duration_frames: None,
            offset_frames: 0,
            volume: 1.0,
            fade_in_frames: 0,
            fade_out_frames: 0,
            required,
        }
    }

    #[test]
    fn required_clip_missing_file_fails_optional_stays_silent() {
        let absent = std::env::temp_dir().join("media-audio-absent-clip.wav");
        let path = absent.to_string_lossy().to_string();
        let mut media = MediaLibrary::new();
        let error = build_audio(
            &{
                let mut p = plan(Vec::new());
                p.clips = vec![clip(&path, true)];
                p
            },
            &mut media,
            DecodeLimits::unbounded(),
        )
        .expect_err("a declared contract slot must fail explicitly");
        assert_eq!(error.code(), "invalid_input");
        assert!(error.message.contains("required audio clip"), "{error}");
        let bed = build_audio(
            &{
                let mut p = plan(Vec::new());
                p.clips = vec![clip(&path, false)];
                p
            },
            &mut media,
            DecodeLimits::unbounded(),
        )
        .expect("reserved slot stays silent");
        assert!(bed.iter().all(|s| *s == 0), "optional missing clip must be silence");
    }

    #[test]
    fn unknown_kind_is_an_explicit_error() {
        let error = build_audio(
            &plan(vec![SfxPlacement { kind: "airhorn".into(), at_frames: 0, volume: 0.5 }]),
            &mut MediaLibrary::new(),
            DecodeLimits::unbounded(),
        )
        .expect_err("unknown kind must fail");
        assert_eq!(error.code(), "invalid_input");
        assert!(error.message.contains("airhorn"));
    }

    #[test]
    fn clip_fades_ramp_linearly() {
        assert_eq!(clip_gain(0, 100, 0, 0), 1.0);
        assert_eq!(clip_gain(0, 100, 10, 0), 0.0);
        assert_eq!(clip_gain(5, 100, 10, 0), 0.5);
        assert_eq!(clip_gain(50, 100, 10, 0), 1.0);
        assert_eq!(clip_gain(99, 100, 0, 10), 0.0);
        assert_eq!(clip_gain(94, 100, 0, 10), 0.5);
        assert_eq!(clip_gain(50, 100, 0, 10), 1.0);
        // A fade longer than the region never reaches unity on that side.
        assert!((clip_gain(50, 100, 200, 0) - 0.25).abs() < 1e-6);
    }
}
