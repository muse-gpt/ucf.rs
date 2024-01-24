//! PCM16 and 32-bit float WAV readers and writers.
//!
//! The two formats are explicit, separate entry points: a reader only accepts
//! the format its name declares, a writer stamps the caller's rate into the
//! header. Nothing silently reinterprets bits between the two.
//!
//! The readers report the declared sample rate instead of discarding it, so
//! mixing can reject a rate mismatch rather than play the file at the wrong
//! speed.

use media_types::Error;

/// A parsed PCM16 WAV payload.
pub struct WavPcm16 {
    /// Sample rate declared by the file's `fmt ` chunk.
    pub sample_rate: u32,
    /// Interleaved stereo frames in -1..=1 floats (mono is widened).
    pub frames: Vec<[f32; 2]>,
}

/// A parsed 32-bit float WAV payload.
pub struct WavFloat {
    /// Sample rate declared by the file's `fmt ` chunk.
    pub sample_rate: u32,
    /// Interleaved stereo frames at their stored values (mono is widened).
    pub frames: Vec<[f32; 2]>,
}

/// The chunks a WAV reader needs, located by one shared RIFF walk.
struct RiffPayload {
    format: u16,
    channels: u16,
    sample_rate: u32,
    bits: u16,
    data_start: usize,
    data_len: usize,
}

/// Minimal RIFF scan: finds the `fmt ` and `data` chunks without decoding.
/// Returns `None` for anything that is not a RIFF/WAVE container.
fn scan_riff(bytes: &[u8]) -> Option<RiffPayload> {
    let read_u32 = |i: usize| -> Option<u32> { Some(u32::from_le_bytes(bytes[i..i + 4].try_into().ok()?)) };
    let read_u16 = |i: usize| -> Option<u16> { Some(u16::from_le_bytes(bytes[i..i + 2].try_into().ok()?)) };
    if bytes.len() < 44 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return None;
    }
    let mut pos = 12usize;
    let mut format = 1u16;
    let mut channels = 1u16;
    let mut sample_rate = 0u32;
    let mut bits = 16u16;
    let mut data_start = 0usize;
    let mut data_len = 0usize;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = read_u32(pos + 4)? as usize;
        match id {
            b"fmt " => {
                format = read_u16(pos + 8)?;
                channels = read_u16(pos + 10)?;
                sample_rate = read_u32(pos + 12)?;
                bits = read_u16(pos + 22)?;
            }
            b"data" => {
                data_start = pos + 8;
                data_len = size.min(bytes.len() - data_start);
            }
            _ => {}
        }
        pos += 8 + size + (size % 2);
    }
    if channels == 0 || sample_rate == 0 || data_len == 0 {
        return None;
    }
    Some(RiffPayload { format, channels, sample_rate, bits, data_start, data_len })
}

/// PCM16 RIFF reader: interleaved stereo f32 frames out, mono is widened.
#[must_use]
pub fn parse_wav_pcm16(bytes: &[u8]) -> Option<WavPcm16> {
    let riff = scan_riff(bytes)?;
    if riff.format != 1 || riff.bits != 16 {
        return None;
    }
    let frame_bytes = riff.channels as usize * 2;
    let frames_count = riff.data_len / frame_bytes;
    let mut out = Vec::with_capacity(frames_count);
    for f in 0..frames_count {
        let base = riff.data_start + f * frame_bytes;
        let left = i16::from_le_bytes(bytes[base..base + 2].try_into().ok()?) as f32 / 32768.0;
        let right = if riff.channels >= 2 {
            i16::from_le_bytes(bytes[base + 2..base + 4].try_into().ok()?) as f32 / 32768.0
        } else {
            left
        };
        out.push([left, right]);
    }
    Some(WavPcm16 { sample_rate: riff.sample_rate, frames: out })
}

/// IEEE-float RIFF reader: only 32-bit float payloads decode, mono is widened.
/// A PCM16 file is rejected here on purpose — use `parse_wav_pcm16` for it.
#[must_use]
pub fn parse_wav_f32(bytes: &[u8]) -> Option<WavFloat> {
    let riff = scan_riff(bytes)?;
    if riff.format != 3 || riff.bits != 32 {
        return None;
    }
    let frame_bytes = riff.channels as usize * 4;
    let frames_count = riff.data_len / frame_bytes;
    let mut out = Vec::with_capacity(frames_count);
    for f in 0..frames_count {
        let base = riff.data_start + f * frame_bytes;
        let left = f32::from_le_bytes(bytes[base..base + 4].try_into().ok()?);
        let right = if riff.channels >= 2 { f32::from_le_bytes(bytes[base + 4..base + 8].try_into().ok()?) } else { left };
        out.push([left, right]);
    }
    Some(WavFloat { sample_rate: riff.sample_rate, frames: out })
}

/// The fixed 44-byte RIFF/WAVE/fmt/data header for a stereo payload.
fn wav_header(format_tag: u16, bits: u16, sample_rate: u32, data_bytes: usize) -> Vec<u8> {
    let block_align = (bits / 8) as u16 * 2;
    let mut out: Vec<u8> = Vec::with_capacity(44 + data_bytes);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&((36 + data_bytes) as u32).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&format_tag.to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&(sample_rate as u32 * block_align as u32).to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&bits.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(data_bytes as u32).to_le_bytes());
    out
}

/// Writes an interleaved s16 stereo buffer as a PCM16 WAV file (debug aid and
/// the format the narration slot expects).
///
/// `sample_rate` is the input contract: it is stamped into the header and the
/// caller must pass samples actually produced at that rate.
pub fn write_wav(path: &str, samples: &[i16], sample_rate: u32) -> Result<(), Error> {
    if sample_rate == 0 {
        return Err(Error::invalid_input("wav: sample rate must be positive"));
    }
    let mut out = wav_header(1, 16, sample_rate, samples.len() * 2);
    for s in samples {
        out.extend_from_slice(&s.to_le_bytes());
    }
    std::fs::write(path, out)?;
    Ok(())
}

/// Writes an interleaved f32 stereo buffer as a 32-bit float WAV file.
///
/// The float format keeps headroom above 1.0 and below -1.0 instead of
/// clipping, which is what preview and round-trip tooling needs. Same sample
/// rate contract as `write_wav`.
pub fn write_wav_f32(path: &str, samples: &[f32], sample_rate: u32) -> Result<(), Error> {
    if sample_rate == 0 {
        return Err(Error::invalid_input("wav: sample rate must be positive"));
    }
    let mut out = wav_header(3, 32, sample_rate, samples.len() * 4);
    for s in samples {
        out.extend_from_slice(&s.to_le_bytes());
    }
    std::fs::write(path, out)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_path(name: &str) -> std::path::PathBuf {
        let mut path = std::env::temp_dir();
        let unique = format!("media-codec-wav-test-{}-{name}", std::process::id());
        path.push(unique);
        std::fs::create_dir_all(&path).expect("temp dir");
        path.push(name);
        path
    }

    #[test]
    fn pcm16_round_trip_keeps_the_declared_rate() {
        let path = temp_path("pcm16.wav");
        let samples = [0i16, 32767, -32768, 1000];
        write_wav(path.to_str().unwrap(), &samples, 48_000).expect("write");
        let bytes = std::fs::read(&path).expect("read");
        let wav = parse_wav_pcm16(&bytes).expect("pcm16 decodes");
        assert_eq!(wav.sample_rate, 48_000);
        assert_eq!(wav.frames.len(), 2);
        assert!((wav.frames[0][0] - 0.0).abs() < 1e-9);
        assert!((wav.frames[0][1] - 32767.0 / 32768.0).abs() < 1e-6);
        assert!((wav.frames[1][0] - (-1.0)).abs() < 1e-6);
        assert!((wav.frames[1][1] - 1000.0 / 32768.0).abs() < 1e-6);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn float_round_trip_keeps_headroom_values() {
        let path = temp_path("float.wav");
        let samples = [0.25f32, -0.5, 1.75, -2.0];
        write_wav_f32(path.to_str().unwrap(), &samples, 44_100).expect("write");
        let bytes = std::fs::read(&path).expect("read");
        let wav = parse_wav_f32(&bytes).expect("float decodes");
        assert_eq!(wav.sample_rate, 44_100);
        assert_eq!(wav.frames, vec![[0.25, -0.5], [1.75, -2.0]]);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn formats_are_explicit_and_never_cross_read() {
        let pcm = temp_path("cross-pcm.wav");
        let flt = temp_path("cross-float.wav");
        write_wav(pcm.to_str().unwrap(), &[0, 1], 8_000).expect("write");
        write_wav_f32(flt.to_str().unwrap(), &[0.0, 1.0], 8_000).expect("write");
        assert!(parse_wav_f32(&std::fs::read(&pcm).expect("read")).is_none(), "PCM16 must not decode as float");
        assert!(parse_wav_pcm16(&std::fs::read(&flt).expect("read")).is_none(), "float must not decode as PCM16");
        let _ = std::fs::remove_file(&pcm);
        let _ = std::fs::remove_file(&flt);
    }

    #[test]
    fn zero_rate_is_rejected() {
        assert_eq!(write_wav("unused.wav", &[], 0).expect_err("zero rate").code(), "invalid_input");
        assert_eq!(write_wav_f32("unused.wav", &[], 0).expect_err("zero rate").code(), "invalid_input");
    }
}
