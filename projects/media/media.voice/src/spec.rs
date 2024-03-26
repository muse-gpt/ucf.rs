//! Voice spec: the singing-voice input contract.
//!
//! A song is a spec, not an audio asset: syllables with an onset consonant, a
//! vowel, a pitch and a duration in beats. Everything is deterministic — the
//! same spec and seed render byte-identical samples. This is categorical
//! (formant) synthesis with explicit, honest control: pitch, duration, vowel
//! quality and onset class. It is not a neural voicebank and does not pretend
//! to be one; per-speaker timbre modelling is a separate, later contract.

use media_types::Error;
use serde::{Deserialize, Serialize};

/// A sung vowel. The five cardinal vowels are the first contract; diphthongs
/// and reduced vowels are future work and must not be smuggled in as strings.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Vowel {
    /// 「あ / a / 阿」
    A,
    /// 「え / e / 诶」
    E,
    /// 「い / i / 衣」
    I,
    /// 「お / o / 哦」
    O,
    /// 「う / u / 乌」
    U,
}

impl Vowel {
    /// Every contracted vowel, in canonical order (bindings list capabilities
    /// from this one source — never a hand-copied string array).
    pub const ALL: [Self; 5] = [Self::A, Self::E, Self::I, Self::O, Self::U];

    /// kebab-case spelling, authoritative for spec parsing.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::A => "a",
            Self::E => "e",
            Self::I => "i",
            Self::O => "o",
            Self::U => "u",
        }
    }

    /// Parses the kebab-case spelling.
    ///
    /// # Errors
    /// Unknown names are an explicit `invalid_input`, never a silent vowel
    /// fallback.
    pub fn parse(raw: &str) -> Result<Self, Error> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "a" => Ok(Self::A),
            "e" => Ok(Self::E),
            "i" => Ok(Self::I),
            "o" => Ok(Self::O),
            "u" => Ok(Self::U),
            other => Err(Error::invalid_input(format!("unknown vowel `{other}` (a|e|i|o|u)"))),
        }
    }

    /// Formant centre frequencies (F1, F2, F3) in Hz for a neutral adult voice.
    #[must_use]
    pub const fn formants(self) -> [f32; 3] {
        match self {
            Self::A => [800.0, 1200.0, 2550.0],
            Self::E => [400.0, 2000.0, 2550.0],
            Self::I => [300.0, 2300.0, 3000.0],
            Self::O => [450.0, 900.0, 2450.0],
            Self::U => [325.0, 850.0, 2350.0],
        }
    }
}

/// A syllable-onset consonant class. Classes, not phones: the contract is the
/// audible event (fricative, plosive, nasal, glide), not an IPA inventory.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Onset {
    /// Noisy fricative (s / sh / し).
    S,
    /// Plosive burst (t / k / た行).
    T,
    /// Nasal hum (n / m / な行・ま行).
    N,
    /// Liquid glide (r / l / ら行).
    R,
    /// Glide from the /u/ colour (w / わ行).
    W,
    /// Glide from the /i/ colour (y / や行).
    Y,
    /// Labial fricative-ish (f / h / ふ行).
    F,
}

impl Onset {
    /// Every contracted onset class, in canonical order.
    pub const ALL: [Self; 7] = [Self::S, Self::T, Self::N, Self::R, Self::W, Self::Y, Self::F];

    /// kebab-case spelling, authoritative for spec parsing.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::S => "s",
            Self::T => "t",
            Self::N => "n",
            Self::R => "r",
            Self::W => "w",
            Self::Y => "y",
            Self::F => "f",
        }
    }

    /// Parses the kebab-case spelling.
    ///
    /// # Errors
    /// Unknown names are an explicit `invalid_input`.
    pub fn parse(raw: &str) -> Result<Self, Error> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "s" => Ok(Self::S),
            "t" => Ok(Self::T),
            "n" => Ok(Self::N),
            "r" => Ok(Self::R),
            "w" => Ok(Self::W),
            "y" => Ok(Self::Y),
            "f" => Ok(Self::F),
            other => Err(Error::invalid_input(format!(
                "unknown onset `{other}` (s|t|n|r|w|y|f)"
            ))),
        }
    }
}

/// One sung syllable.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Syllable {
    /// Onset consonant class; `None` sings the vowel alone.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub onset: Option<Onset>,
    /// The vowel the syllable settles on.
    pub vowel: Vowel,
    /// Fundamental frequency in Hz (40..1200).
    pub pitch: f32,
    /// Duration in beats at the song tempo (0.05..8).
    pub beats: f32,
}

/// Vibrato: periodic pitch modulation while a syllable sustains.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Vibrato {
    /// Modulation rate in Hz (0..10).
    pub rate_hz: f32,
    /// Pitch depth as a fraction of the carrier (0..0.3).
    pub depth: f32,
    /// Seconds after syllable onset before vibrato reaches full depth.
    pub delay_s: f32,
}

impl Default for Vibrato {
    fn default() -> Self {
        Self { rate_hz: 5.5, depth: 0.02, delay_s: 0.12 }
    }
}

/// The singing-voice input contract: one deterministic, controllable song.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct VoiceSpec {
    /// The lyric: an ordered list of syllables. Empty is invalid.
    pub syllables: Vec<Syllable>,
    /// Song tempo in beats per minute (20..400).
    pub tempo_bpm: f32,
    /// Sustained-vowel vibrato.
    pub vibrato: Vibrato,
    /// Breath noise mixed into the voiced source, 0..1 (default 0.08).
    pub breath: f32,
    /// Overall gain, 0..1 (default 0.8).
    pub volume: f32,
    /// Random seed for the breath stream (`media.dsp::Rng` contract, default 1).
    pub seed: u32,
}

impl VoiceSpec {
    /// A spec with defaults for tempo, vibrato, breath, volume and seed.
    #[must_use]
    pub fn new(syllables: Vec<Syllable>) -> Self {
        Self {
            syllables,
            tempo_bpm: 120.0,
            vibrato: Vibrato::default(),
            breath: 0.08,
            volume: 0.8,
            seed: 1,
        }
    }

    /// Total duration in seconds at the song tempo.
    #[must_use]
    pub fn duration_seconds(&self) -> f32 {
        let beats: f32 = self.syllables.iter().map(|s| s.beats.max(0.0)).sum();
        beats * 60.0 / self.tempo_bpm.max(1.0)
    }

    /// Validates every range: empty lyric, out-of-range pitch/beats/tempo,
    /// non-finite values and volume/breath bounds are explicit
    /// `invalid_input` errors — the Rust domain owns the contract, bindings
    /// only map errors.
    ///
    /// # Errors
    /// Returns `invalid_input` describing the first violated range.
    pub fn validate(&self) -> Result<(), Error> {
        if self.syllables.is_empty() {
            return Err(Error::invalid_input("voice: the lyric must have at least one syllable"));
        }
        if !(20.0..=400.0).contains(&self.tempo_bpm) || !self.tempo_bpm.is_finite() {
            return Err(Error::invalid_input(format!(
                "voice: tempo {} outside 20..400 bpm",
                self.tempo_bpm
            )));
        }
        if !(0.0..=1.0).contains(&self.volume) || !self.volume.is_finite() {
            return Err(Error::invalid_input(format!("voice: volume {} outside 0..1", self.volume)));
        }
        if !(0.0..=1.0).contains(&self.breath) || !self.breath.is_finite() {
            return Err(Error::invalid_input(format!("voice: breath {} outside 0..1", self.breath)));
        }
        if !(0.0..=10.0).contains(&self.vibrato.rate_hz) || !self.vibrato.rate_hz.is_finite() {
            return Err(Error::invalid_input(format!(
                "voice: vibrato rate {} outside 0..10 Hz",
                self.vibrato.rate_hz
            )));
        }
        if !(0.0..=0.3).contains(&self.vibrato.depth) || !self.vibrato.depth.is_finite() {
            return Err(Error::invalid_input(format!(
                "voice: vibrato depth {} outside 0..0.3",
                self.vibrato.depth
            )));
        }
        if !(0.0..=2.0).contains(&self.vibrato.delay_s) || !self.vibrato.delay_s.is_finite() {
            return Err(Error::invalid_input(format!(
                "voice: vibrato delay {} outside 0..2 s",
                self.vibrato.delay_s
            )));
        }
        for (index, syllable) in self.syllables.iter().enumerate() {
            if !(40.0..=1200.0).contains(&syllable.pitch) || !syllable.pitch.is_finite() {
                return Err(Error::invalid_input(format!(
                    "voice: syllable {index} pitch {} outside 40..1200 Hz",
                    syllable.pitch
                )));
            }
            if !(0.05..=8.0).contains(&syllable.beats) || !syllable.beats.is_finite() {
                return Err(Error::invalid_input(format!(
                    "voice: syllable {index} beats {} outside 0.05..8",
                    syllable.beats
                )));
            }
        }
        Ok(())
    }
}

impl Default for VoiceSpec {
    fn default() -> Self {
        Self::new(Vec::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn syllable(vowel: Vowel, pitch: f32, beats: f32) -> Syllable {
        Syllable { onset: None, vowel, pitch, beats }
    }

    #[test]
    fn vowel_and_onset_parse_kebab_and_reject_unknown() {
        for vowel in [Vowel::A, Vowel::E, Vowel::I, Vowel::O, Vowel::U] {
            assert_eq!(Vowel::parse(vowel.as_str()).expect("parses"), vowel);
        }
        assert_eq!(Vowel::parse("x").unwrap_err().code(), "invalid_input");
        for onset in [Onset::S, Onset::T, Onset::N, Onset::R, Onset::W, Onset::Y, Onset::F] {
            assert_eq!(Onset::parse(onset.as_str()).expect("parses"), onset);
        }
        assert_eq!(Onset::parse("q").unwrap_err().code(), "invalid_input");
    }

    #[test]
    fn formant_table_matches_reference_values() {
        assert_eq!(Vowel::A.formants(), [800.0, 1200.0, 2550.0]);
        assert_eq!(Vowel::I.formants(), [300.0, 2300.0, 3000.0]);
        assert_eq!(Vowel::U.formants(), [325.0, 850.0, 2350.0]);
    }

    #[test]
    fn validate_rejects_empty_out_of_range_and_non_finite() {
        let spec = VoiceSpec::new(vec![syllable(Vowel::A, 440.0, 1.0)]);
        assert!(spec.validate().is_ok());
        // 空歌词
        let empty = VoiceSpec::default();
        assert_eq!(empty.validate().unwrap_err().code(), "invalid_input");
        // 音高越界
        let mut bad = spec.clone();
        bad.syllables[0].pitch = 20.0;
        assert_eq!(bad.validate().unwrap_err().code(), "invalid_input");
        // 时值越界
        let mut bad = spec.clone();
        bad.syllables[0].beats = 9.0;
        assert_eq!(bad.validate().unwrap_err().code(), "invalid_input");
        // 非有限值
        let mut bad = spec.clone();
        bad.syllables[0].pitch = f32::NAN;
        assert_eq!(bad.validate().unwrap_err().code(), "invalid_input");
        // 音量越界
        let mut bad = spec.clone();
        bad.volume = 1.5;
        assert_eq!(bad.validate().unwrap_err().code(), "invalid_input");
        // 呼吸越界
        let mut bad = spec.clone();
        bad.breath = -0.1;
        assert_eq!(bad.validate().unwrap_err().code(), "invalid_input");
    }

    #[test]
    fn duration_is_derived_from_beats_and_tempo() {
        let spec = VoiceSpec::new(vec![syllable(Vowel::A, 440.0, 2.0), syllable(Vowel::O, 330.0, 1.0)]);
        assert!((spec.duration_seconds() - 1.5).abs() < 1e-5);
    }
}
