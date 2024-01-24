//! Sound-effect export: the authoritative synthesis-plus-codec composition.
//!
//! A voice is rendered by `media.synthesis`, quantized to PCM16 and written
//! by `media.codec`. The sample rate is an explicit input contract on every
//! entry — the OpenSFX core's fixed 44.1 kHz default lives in `media.types`
//! as a project default, and hosts pass it (or another rate) here. File IO
//! is gated behind the `sfx-export` feature, mirroring how the scene and
//! image-pipeline entries gate theirs.

use media_codec::write_wav;
use media_synthesis::{PRESETS, SfxSpec, render};
use media_types::Error;

/// Renders `spec` at `sample_rate` and writes it as a PCM16 stereo WAV.
pub fn write_sfx(spec: &SfxSpec, path: &str, sample_rate: u32) -> Result<(), Error> {
    if sample_rate == 0 {
        return Err(Error::invalid_input("sfx export: sample rate must be positive"));
    }
    let samples = render(spec, sample_rate as f32)?;
    let pcm: Vec<i16> = samples
        .iter()
        .flat_map(|frame| frame.iter().map(|sample| (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16))
        .collect();
    write_wav(path, &pcm, sample_rate)
}

/// Renders every shipped preset into `directory`, one WAV per preset.
///
/// Returns the written paths in catalogue order.
pub fn write_sfx_catalogue(directory: &str, sample_rate: u32) -> Result<Vec<String>, Error> {
    std::fs::create_dir_all(directory).map_err(|e| Error::io(format!("{directory}: {e}")))?;
    let mut written = Vec::new();
    for entry in PRESETS {
        let path = format!("{directory}/{}.wav", entry.name);
        write_sfx(&entry.spec, &path, sample_rate)?;
        written.push(path);
    }
    Ok(written)
}

#[cfg(test)]
mod tests {
    use super::*;
    use media_codec::parse_wav_pcm16;
    use media_synthesis::{Family, preset, preset_names};

    fn temp_dir(name: &str) -> std::path::PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!("media-render-sfx-test-{}-{name}", std::process::id()));
        std::fs::create_dir_all(&path).expect("temp dir");
        path
    }

    #[test]
    fn write_renders_a_decodable_wav_at_the_declared_rate() {
        let dir = temp_dir("single");
        let path = dir.join("tick.wav");
        let spec = SfxSpec::new(Family::Tick, 0.25);
        write_sfx(&spec, path.to_str().unwrap(), 48_000).expect("write");
        let wav = parse_wav_pcm16(&std::fs::read(&path).expect("read")).expect("decode");
        assert_eq!(wav.sample_rate, 48_000, "the header must carry the requested rate");
        assert_eq!(wav.frames.len(), 12_000);
        let peak = wav.frames.iter().flat_map(|f| f.iter()).fold(0.0f32, |a, b| a.max(b.abs()));
        assert!(peak > 0.02, "exported tick too quiet: {peak}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn exports_are_reproducible_for_the_same_spec() {
        let dir = temp_dir("repro");
        let a = dir.join("a.wav");
        let b = dir.join("b.wav");
        let spec = preset("impact-deep").expect("preset");
        write_sfx(&spec, a.to_str().unwrap(), 44_100).expect("write a");
        write_sfx(&spec, b.to_str().unwrap(), 44_100).expect("write b");
        assert_eq!(std::fs::read(&a).expect("read a"), std::fs::read(&b).expect("read b"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn catalogue_writes_one_wav_per_preset_in_order() {
        let dir = temp_dir("catalogue");
        let written = write_sfx_catalogue(dir.to_str().unwrap(), 44_100).expect("catalogue");
        assert_eq!(written.len(), preset_names().len());
        for entry in PRESETS {
            let path = dir.join(format!("{}.wav", entry.name));
            let bytes = std::fs::read(&path).expect("preset wav exists");
            let wav = parse_wav_pcm16(&bytes).expect("decodes");
            let peak = wav.frames.iter().flat_map(|f| f.iter()).fold(0.0f32, |a, b| a.max(b.abs()));
            assert!(peak > 0.0, "{} silent", entry.name);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn invalid_spec_or_rate_is_rejected_before_io() {
        let mut spec = SfxSpec::new(Family::Tick, 0.1);
        spec.duration = 0.001;
        assert_eq!(write_sfx(&spec, "unused.wav", 44_100).expect_err("invalid duration").code(), "invalid_input");
        assert_eq!(
            write_sfx(&SfxSpec::new(Family::Tick, 0.1), "unused.wav", 0).expect_err("zero rate").code(),
            "invalid_input"
        );
    }
}
