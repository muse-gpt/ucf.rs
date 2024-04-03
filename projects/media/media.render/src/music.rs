//! Instrument and groove export: the authoritative synthesis-plus-codec
//! composition for the playable instruments and drum patterns.
//!
//! One hit is rendered by `media.synthesis::render_instrument` and one groove
//! by `render_pattern`; both are quantized to PCM16 and written by
//! `media.codec`. The sample rate is an explicit input contract on every entry
//! — nothing is resampled behind the caller's back. File IO is gated behind
//! the `sfx-export` feature, the same gate the sound-effect export uses.

use media_codec::write_wav;
use media_synthesis::{DrumPattern, INSTRUMENT_PRESETS, InstrumentSpec, pattern_presets, render_instrument, render_pattern};
use media_types::Error;

fn to_pcm(samples: &[[f32; 2]]) -> Vec<i16> {
    samples
        .iter()
        .flat_map(|frame| frame.iter().map(|sample| (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16))
        .collect()
}

/// Renders one instrument hit at `sample_rate` and writes it as PCM16 stereo WAV.
///
/// # Errors
/// An invalid spec or rate is `invalid_input`; write failures keep `io`.
pub fn write_instrument(spec: &InstrumentSpec, path: &str, sample_rate: u32) -> Result<(), Error> {
    if sample_rate == 0 {
        return Err(Error::invalid_input("instrument export: sample rate must be positive"));
    }
    let samples = render_instrument(spec, sample_rate as f32)?;
    write_wav(path, &to_pcm(&samples), sample_rate)
}

/// Renders every shipped instrument preset into `directory`, one WAV per preset.
///
/// # Errors
/// Invalid rate or write failures are explicit; returns the written paths in
/// catalogue order.
pub fn write_instrument_catalogue(directory: &str, sample_rate: u32) -> Result<Vec<String>, Error> {
    std::fs::create_dir_all(directory).map_err(|e| Error::io(format!("{directory}: {e}")))?;
    let mut written = Vec::new();
    for entry in INSTRUMENT_PRESETS {
        let path = format!("{directory}/{}.wav", entry.name);
        write_instrument(&entry.spec, &path, sample_rate)?;
        written.push(path);
    }
    Ok(written)
}

/// Renders one drum groove at `sample_rate` and writes it as PCM16 stereo WAV.
///
/// # Errors
/// An invalid pattern or rate is `invalid_input`; write failures keep `io`.
pub fn write_pattern(pattern: &DrumPattern, path: &str, sample_rate: u32) -> Result<(), Error> {
    if sample_rate == 0 {
        return Err(Error::invalid_input("pattern export: sample rate must be positive"));
    }
    let samples = render_pattern(pattern, sample_rate as f32)?;
    write_wav(path, &to_pcm(&samples), sample_rate)
}

/// Renders every shipped groove preset into `directory`, one WAV per preset.
///
/// # Errors
/// Invalid rate or write failures are explicit; returns the written paths in
/// catalogue order.
pub fn write_pattern_catalogue(directory: &str, sample_rate: u32) -> Result<Vec<String>, Error> {
    std::fs::create_dir_all(directory).map_err(|e| Error::io(format!("{directory}: {e}")))?;
    let mut written = Vec::new();
    for entry in pattern_presets() {
        let path = format!("{directory}/{}.wav", entry.name);
        write_pattern(&entry.pattern, &path, sample_rate)?;
        written.push(path);
    }
    Ok(written)
}

#[cfg(test)]
mod tests {
    use super::*;
    use media_codec::parse_wav_pcm16;
    use media_synthesis::{Instrument, instrument_preset, pattern_preset};

    fn temp_dir(name: &str) -> std::path::PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!("media-render-music-test-{}-{name}", std::process::id()));
        std::fs::create_dir_all(&path).expect("temp dir");
        path
    }

    #[test]
    fn writes_decodable_instrument_and_pattern_wavs() {
        let dir = temp_dir("wavs");
        let kick = dir.join("kick.wav");
        write_instrument(&InstrumentSpec::new(Instrument::Kick, 36), kick.to_str().unwrap(), 48_000).expect("kick");
        let wav = parse_wav_pcm16(&std::fs::read(&kick).expect("read")).expect("decode");
        assert_eq!(wav.sample_rate, 48_000);
        let peak = wav.frames.iter().flat_map(|frame| frame.iter()).fold(0.0f32, |a, b| a.max(b.abs()));
        assert!(peak > 0.05, "kick too quiet: {peak}");

        let groove = dir.join("groove.wav");
        write_pattern(&pattern_preset("rock").expect("preset"), groove.to_str().unwrap(), 44_100).expect("groove");
        let wav = parse_wav_pcm16(&std::fs::read(&groove).expect("read")).expect("decode");
        assert_eq!(wav.sample_rate, 44_100);
        let peak = wav.frames.iter().flat_map(|frame| frame.iter()).fold(0.0f32, |a, b| a.max(b.abs()));
        assert!(peak > 0.05, "groove too quiet: {peak}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn catalogue_writes_match_the_upstream_names() {
        let dir = temp_dir("catalogue");
        let instruments = write_instrument_catalogue(dir.to_str().unwrap(), 44_100).expect("instruments");
        assert_eq!(instruments.len(), INSTRUMENT_PRESETS.len());
        let patterns = write_pattern_catalogue(dir.to_str().unwrap(), 44_100).expect("patterns");
        assert_eq!(patterns.len(), pattern_presets().len());
        for path in instruments.iter().chain(patterns.iter()) {
            assert!(std::fs::metadata(path).expect("written").len() > 44);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn zero_rate_is_rejected_before_io() {
        assert_eq!(
            write_instrument(&instrument_preset("kick-tight").expect("preset"), "unused.wav", 0)
                .expect_err("zero rate")
                .code(),
            "invalid_input"
        );
        assert_eq!(
            write_pattern(&pattern_preset("lofi").expect("preset"), "unused.wav", 0).expect_err("zero rate").code(),
            "invalid_input"
        );
    }
}
