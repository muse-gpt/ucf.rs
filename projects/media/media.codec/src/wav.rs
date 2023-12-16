//! PCM16 WAV reader and writer.
//!
//! The reader reports the declared sample rate instead of discarding it, so
//! mixing can reject a rate mismatch rather than play the file at the wrong
//! speed. The writer stamps the caller's rate into the header: the sample
//! buffer and the header must always agree.

use media_types::Error;

/// A parsed PCM16 WAV payload.
pub struct WavPcm16 {
    /// Sample rate declared by the file's `fmt ` chunk.
    pub sample_rate: u32,
    /// Interleaved stereo frames in -1..=1 floats (mono is widened).
    pub frames: Vec<[f32; 2]>,
}

/// Minimal RIFF reader: PCM16 in, interleaved stereo f32 frames out.
#[must_use]
pub fn parse_wav_pcm16(bytes: &[u8]) -> Option<WavPcm16> {
    let read_u32 = |i: usize| -> Option<u32> { Some(u32::from_le_bytes(bytes[i..i + 4].try_into().ok()?)) };
    let read_u16 = |i: usize| -> Option<u16> { Some(u16::from_le_bytes(bytes[i..i + 2].try_into().ok()?)) };
    if bytes.len() < 44 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return None;
    }
    let mut pos = 12usize;
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
    if bits != 16 || data_len == 0 || channels == 0 || sample_rate == 0 {
        return None;
    }
    let frame_bytes = channels as usize * 2;
    let frames_count = data_len / frame_bytes;
    let mut out = Vec::with_capacity(frames_count);
    for f in 0..frames_count {
        let base = data_start + f * frame_bytes;
        let left = i16::from_le_bytes(bytes[base..base + 2].try_into().ok()?) as f32 / 32768.0;
        let right =
            if channels >= 2 { i16::from_le_bytes(bytes[base + 2..base + 4].try_into().ok()?) as f32 / 32768.0 } else { left };
        out.push([left, right]);
    }
    Some(WavPcm16 { sample_rate, frames: out })
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
    let data_bytes = samples.len() * 2;
    let mut out: Vec<u8> = Vec::with_capacity(44 + data_bytes);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&((36 + data_bytes) as u32).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&(sample_rate * 4).to_le_bytes());
    out.extend_from_slice(&4u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(data_bytes as u32).to_le_bytes());
    for s in samples {
        out.extend_from_slice(&s.to_le_bytes());
    }
    std::fs::write(path, out)?;
    Ok(())
}
