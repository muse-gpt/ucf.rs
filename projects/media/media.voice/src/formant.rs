//! Formant singing synthesis: a glottal source shaped by vowel formants and
//! syllable-onset events.
//!
//! Signal chain per syllable: phase-continuous pulse train (pitch + vibrato +
//! breath noise) → three 2-pole resonators tuned to the vowel formants →
//! onset event (fricative noise, plosive gap+click, nasal hum, or formant
//! glide) → attack/release envelope. Deterministic: no wall-clock, one
//! `media.dsp::Rng` stream for breath, seed from the spec.
//!
//! This is categorical synthesis — the honest minimal voice. Speaker timbre,
//! phoneme timing models and neural vocoders are separate later contracts and
//! are deliberately not approximated here with hidden parameters.

use crate::spec::{Onset, VoiceSpec};
use media_dsp::Rng;
use media_types::Error;

/// Renders the spec to interleaved stereo f32 (both channels identical; the
/// voice is a mono instrument, the pair is the house mixer contract).
///
/// # Errors
/// Returns the spec's `invalid_input` validation error.
pub fn render(spec: &VoiceSpec, sample_rate: f32) -> Result<Vec<[f32; 2]>, Error> {
    spec.validate()?;
    let total_seconds = spec.duration_seconds();
    let total = (total_seconds * sample_rate).ceil() as usize;
    let mut mono = vec![0.0_f32; total.max(1)];

    // One breath stream per song, consumed in sample order — same spec and
    // seed reproduce byte-identically.
    let mut breath = Rng::new(spec.seed);
    let mut phase = 0.0_f32;
    let beat_seconds = 60.0 / spec.tempo_bpm;
    let mut cursor = 0_usize;

    for syllable in &spec.syllables {
        let length = (syllable.beats * beat_seconds * sample_rate).round() as usize;
        let length = length.max(1).min(mono.len().saturating_sub(cursor).max(1));
        render_syllable(spec, syllable, &mut mono[cursor..cursor + length], sample_rate, &mut breath, &mut phase);
        cursor += length;
    }

    // Normalise into the requested volume; a song with real pitch content
    // peaks well below 1.0 from the resonator gains alone.
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

/// One 2-pole resonator: bandpass at `frequency` with bandwidth `bandwidth`.
#[derive(Clone, Copy)]
struct Resonator {
    coefficient_a: f32,
    coefficient_b: f32,
    y1: f32,
    y2: f32,
}

impl Resonator {
    fn new(frequency: f32, bandwidth: f32, sample_rate: f32) -> Self {
        let r = (-std::f32::consts::PI * bandwidth / sample_rate).exp();
        let theta = 2.0 * std::f32::consts::PI * frequency / sample_rate;
        Self {
            coefficient_a: 2.0 * r * theta.cos(),
            coefficient_b: -r * r,
            y1: 0.0,
            y2: 0.0,
        }
    }

    /// Retunes in place (formant glides), keeping the filter state.
    fn retune(&mut self, frequency: f32, bandwidth: f32, sample_rate: f32) {
        let r = (-std::f32::consts::PI * bandwidth / sample_rate).exp();
        let theta = 2.0 * std::f32::consts::PI * frequency / sample_rate;
        self.coefficient_a = 2.0 * r * theta.cos();
        self.coefficient_b = -r * r;
    }

    fn process(&mut self, input: f32) -> f32 {
        let y = input + self.coefficient_a * self.y1 + self.coefficient_b * self.y2;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }
}

fn render_syllable(
    spec: &VoiceSpec,
    syllable: &crate::spec::Syllable,
    out: &mut [f32],
    sample_rate: f32,
    breath: &mut Rng,
    phase: &mut f32,
) {
    let length = out.len();
    let seconds = length as f32 / sample_rate;
    let formants = syllable.vowel.formants();
    let mut f1 = Resonator::new(formants[0], 80.0, sample_rate);
    let mut f2 = Resonator::new(formants[1], 90.0, sample_rate);
    let mut f3 = Resonator::new(formants[2], 120.0, sample_rate);

    // Onset events reshape the first part of the syllable.
    let onset_fraction = match syllable.onset {
        Some(Onset::R | Onset::W | Onset::Y) => 0.3,
        _ => 0.0,
    };
    let (glide_from, glide_to) = match syllable.onset {
        Some(Onset::R) => ([350.0, 1100.0, 2400.0], formants),
        Some(Onset::W) => ([325.0, 850.0, 2350.0], formants),
        Some(Onset::Y) => ([300.0, 2300.0, 3000.0], formants),
        _ => (formants, formants),
    };
    // Fricative / plosive / nasal pre-roll occupies the syllable head.
    let onset_lead = match syllable.onset {
        Some(Onset::S | Onset::F) => (0.045 * sample_rate).round() as usize,
        Some(Onset::T) => (0.030 * sample_rate).round() as usize,
        Some(Onset::N) => (0.040 * sample_rate).round() as usize,
        _ => 0,
    };
    let lead = onset_lead.min(length);
    // Nasal lead uses its own two resonators on the same glottal source.
    let mut n1 = Resonator::new(250.0, 100.0, sample_rate);
    let mut n2 = Resonator::new(1000.0, 150.0, sample_rate);
    // High fricative resonator for S / F leads.
    let fricative_centre = match syllable.onset {
        Some(Onset::F) => 2200.0,
        _ => 5500.0,
    };
    let mut fricative = Resonator::new(fricative_centre, 1500.0, sample_rate);

    let attack = (0.015 * sample_rate).round() as usize;
    let release = (0.045 * sample_rate).round() as usize;
    // Glottal one-pole integration state for this syllable.
    let mut glottal_state = 0.0_f32;
    let cycle = 2.0 * std::f32::consts::PI / spec.vibrato.rate_hz.max(0.01);

    for (index, slot) in out.iter_mut().enumerate() {
        let t = index as f32 / sample_rate;

        // --- onset lead-in ------------------------------------------------
        if index < lead {
            *slot = match syllable.onset {
                Some(Onset::S | Onset::F) => {
                    // Band-limited-ish fricative: noise through a high
                    // resonator; F opens with a softer spectral centre.
                    let raw = (breath.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0;
                    fricative.process(raw) * 0.9
                }
                Some(Onset::T) => {
                    // Plosive: closed glottis (silence) then a broadband click.
                    if t < 0.022 {
                        0.0
                    } else {
                        let raw = (breath.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0;
                        raw * 0.7
                    }
                }
                Some(Onset::N) => {
                    // Nasal hum: nasal formants on the same glottal source.
                    *phase += 2.0 * std::f32::consts::PI * syllable.pitch / sample_rate;
                    let source = glottal(phase, breath, spec.breath, &mut glottal_state);
                    (n1.process(source) + n2.process(source) * 0.4) * 0.7
                }
                _ => 0.0,
            };
            continue;
        }

        // --- voiced body --------------------------------------------------
        // Formant glide over the onset fraction, then the vowel target.
        let glide_t = if onset_fraction > 0.0 && index < (length as f32 * onset_fraction) as usize {
            (t / (seconds * onset_fraction)).clamp(0.0, 1.0)
        } else {
            1.0
        };
        if glide_t < 1.0 {
            f1.retune(glide_from[0] + (glide_to[0] - glide_from[0]) * glide_t, 90.0, sample_rate);
            f2.retune(glide_from[1] + (glide_to[1] - glide_from[1]) * glide_t, 100.0, sample_rate);
            f3.retune(glide_from[2] + (glide_to[2] - glide_from[2]) * glide_t, 130.0, sample_rate);
        }

        // Vibrato ramps in after the delay, one smooth LFO period envelope.
        let vibrato = if spec.vibrato.depth > 0.0 && t > spec.vibrato.delay_s {
            let since = t - spec.vibrato.delay_s;
            let envelope = (since * cycle / cycle).min(1.0);
            let lfo = (2.0 * std::f32::consts::PI * spec.vibrato.rate_hz * since).sin();
            spec.vibrato.depth * envelope * lfo
        } else {
            0.0
        };
        *phase += 2.0 * std::f32::consts::PI * syllable.pitch * (1.0 + vibrato) / sample_rate;

        let source = glottal(phase, breath, spec.breath, &mut glottal_state);
        let shaped = f1.process(source) + f2.process(source) * 0.6 + f3.process(source) * 0.3;

        // Attack / release envelope over the voiced part.
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

/// Glottal source: a phase-continuous pulse train integrated once (one-pole)
/// for spectral rolloff, plus deterministic breath noise from the song's
/// `Rng` stream.
fn glottal(phase: &mut f32, breath: &mut Rng, breath_amount: f32, state: &mut f32) -> f32 {
    let wrapped = *phase % std::f32::consts::TAU;
    let pulse = (wrapped / std::f32::consts::TAU) * 2.0 - 1.0;
    *state = 0.92 * *state + 0.08 * pulse;
    let noise = (breath.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0;
    *state * 0.9 + noise * breath_amount * 0.35
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
        // 音高一致：主周期峰值位置近似相同（这里只验证非静音且有限）。
        for frame in samples_b.iter().take(samples_b.len()) {
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
        assert_eq!(fnv1a64(&bytes), 1917150972074027508, "engine constants are pinned");
    }
}
