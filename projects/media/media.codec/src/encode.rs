//! MP4 encoding through the Media Foundation that ships with Windows.
//!
//! Video enters as NV12 samples and is encoded to H.264; audio enters as
//! PCM16 (with the WAVEFORMATEXTENSIBLE tag MF expects inside uncompressed
//! audio samples) and is encoded to AAC. No external encoder is involved.

use media_types::Error;

/// The sink-writer-backed encoder for one output file.
pub struct Encoder {
    #[cfg(windows)]
    inner: WindowsEncoder,
}

impl Encoder {
    /// Opens `out` for H.264 video at `fps` and optional audio.
    ///
    /// The audio codec is negotiated: AAC is preferred, and machines without
    /// the AAC encoder MFT registered fall back to FLAC.
    pub fn open(
        out: &std::path::Path,
        width: u32,
        height: u32,
        fps: u32,
        bitrate: u32,
        with_audio: bool,
    ) -> Result<Self, Error> {
        #[cfg(windows)]
        {
            Ok(Self { inner: WindowsEncoder::open(out, width, height, fps, bitrate, with_audio)? })
        }
        #[cfg(not(windows))]
        {
            let _ = (out, width, height, fps, bitrate, with_audio);
            Err(Error::unsupported("OpenCut encoding currently requires Windows Media Foundation"))
        }
    }

    /// Name of the negotiated audio codec (`aac`, `flac`, or empty when silent).
    #[must_use]
    pub fn audio_codec(&self) -> &str {
        #[cfg(windows)]
        {
            self.inner.audio_codec()
        }
        #[cfg(not(windows))]
        {
            ""
        }
    }

    /// Writes one NV12 frame (1.5 bytes per pixel, stride = width).
    pub fn write_video(&mut self, nv12: &[u8], frame: u32, fps: u32) -> Result<(), Error> {
        #[cfg(windows)]
        {
            self.inner.write_video(nv12, frame, fps)
        }
        #[cfg(not(windows))]
        {
            let _ = (nv12, frame, fps);
            Ok(())
        }
    }

    /// Writes one interleaved s16 stereo chunk starting at `start_sample`.
    pub fn write_audio(&mut self, pcm: &[i16], start_sample: u64) -> Result<(), Error> {
        #[cfg(windows)]
        {
            self.inner.write_audio(pcm, start_sample)
        }
        #[cfg(not(windows))]
        {
            let _ = (pcm, start_sample);
            Ok(())
        }
    }

    /// Flushes and closes the container.
    pub fn finish(mut self) -> Result<(), Error> {
        #[cfg(windows)]
        {
            self.inner.finish()
        }
        #[cfg(not(windows))]
        {
            Ok(())
        }
    }
}

#[cfg(windows)]
mod windows_encoder {
    use std::path::Path;

    use windows::Win32::Media::MediaFoundation as mf;
    use windows::Win32::System::Com::{COINIT_APARTMENTTHREADED, CoInitializeEx};
    use windows::core::{GUID, PCWSTR};

    use media_types::Error;

    /// MFInterlaceMode_Progressive
    const INTERLACE_PROGRESSIVE: u32 = 2;
    /// MFNominalRange_0_255 (full range)
    const NOMINAL_RANGE_FULL: u32 = 1;

    fn encode_error(context: &str, error: windows::core::Error) -> Error {
        Error::encode(format!("{context}: {error}"))
    }

    fn wide(path: &Path) -> Vec<u16> {
        use std::os::windows::ffi::OsStrExt;
        path.as_os_str().encode_wide().chain(std::iter::once(0)).collect()
    }

    unsafe fn set_guid(mt: &mf::IMFMediaType, key: &GUID, value: &GUID) -> Result<(), windows::core::Error> {
        unsafe { mt.SetGUID(key as *const GUID, value as *const GUID) }
    }

    unsafe fn set_u32(mt: &mf::IMFMediaType, key: &GUID, value: u32) -> Result<(), windows::core::Error> {
        unsafe { mt.SetUINT32(key as *const GUID, value) }
    }

    unsafe fn set_u64(mt: &mf::IMFMediaType, key: &GUID, value: u64) -> Result<(), windows::core::Error> {
        unsafe { mt.SetUINT64(key as *const GUID, value) }
    }

    const fn pack32(hi: u32, lo: u32) -> u64 {
        ((hi as u64) << 32) | lo as u64
    }

    pub(super) struct WindowsEncoder {
        writer: mf::IMFSinkWriter,
        video_index: u32,
        audio_index: Option<u32>,
        width: u32,
        height: u32,
        started: bool,
        audio_codec: String,
    }

    impl WindowsEncoder {
        pub fn open(out: &Path, width: u32, height: u32, fps: u32, bitrate: u32, with_audio: bool) -> Result<Self, Error> {
            unsafe {
                let _ = CoInitializeEx(None, COINIT_APARTMENTTHREADED);
                mf::MFStartup(mf::MF_VERSION, mf::MFSTARTUP_FULL).map_err(|e| encode_error("MFStartup", e))?;
            }
            let candidates: &[(&windows::core::GUID, &str)] = if with_audio {
                &[(&mf::MFAudioFormat_AAC, "aac"), (&mf::MFAudioFormat_FLAC, "flac")]
            } else {
                &[(&mf::MFAudioFormat_AAC, "none")]
            };
            let mut last_error = None;
            for (subtype, name) in candidates {
                match Self::try_open(out, width, height, fps, bitrate, with_audio, subtype, name) {
                    Ok(encoder) => return Ok(encoder),
                    Err(error) => last_error = Some(error),
                }
            }
            Err(last_error.unwrap_or_else(|| Error::encode("no audio codec negotiated")))
        }

        fn try_open(
            out: &Path,
            width: u32,
            height: u32,
            fps: u32,
            bitrate: u32,
            with_audio: bool,
            audio_subtype: &windows::core::GUID,
            codec_name: &str,
        ) -> Result<Self, Error> {
            let wide_path = wide(out);
            let writer: mf::IMFSinkWriter = unsafe {
                mf::MFCreateSinkWriterFromURL(
                    PCWSTR(wide_path.as_ptr()),
                    None::<&mf::IMFByteStream>,
                    None::<&mf::IMFAttributes>,
                )
                .map_err(|e| encode_error("MFCreateSinkWriterFromURL", e))?
            };

            // --- video stream ---
            let video_out = unsafe { mf::MFCreateMediaType().map_err(|e| encode_error("MFCreateMediaType(video out)", e))? };
            unsafe {
                set_guid(&video_out, &mf::MF_MT_MAJOR_TYPE, &mf::MFMediaType_Video)
                    .map_err(|e| encode_error("video major", e))?;
                set_guid(&video_out, &mf::MF_MT_SUBTYPE, &mf::MFVideoFormat_H264)
                    .map_err(|e| encode_error("video subtype", e))?;
                set_u32(&video_out, &mf::MF_MT_AVG_BITRATE, bitrate).map_err(|e| encode_error("video bitrate", e))?;
                set_u64(&video_out, &mf::MF_MT_FRAME_SIZE, pack32(width, height)).map_err(|e| encode_error("frame size", e))?;
                set_u64(&video_out, &mf::MF_MT_FRAME_RATE, pack32(fps, 1)).map_err(|e| encode_error("frame rate", e))?;
                set_u64(&video_out, &mf::MF_MT_PIXEL_ASPECT_RATIO, pack32(1, 1))
                    .map_err(|e| encode_error("pixel aspect", e))?;
                set_u32(&video_out, &mf::MF_MT_INTERLACE_MODE, INTERLACE_PROGRESSIVE)
                    .map_err(|e| encode_error("interlace", e))?;
            }
            let video_in = unsafe { mf::MFCreateMediaType().map_err(|e| encode_error("MFCreateMediaType(video in)", e))? };
            unsafe {
                set_guid(&video_in, &mf::MF_MT_MAJOR_TYPE, &mf::MFMediaType_Video)
                    .map_err(|e| encode_error("video in major", e))?;
                set_guid(&video_in, &mf::MF_MT_SUBTYPE, &mf::MFVideoFormat_NV12)
                    .map_err(|e| encode_error("video in subtype", e))?;
                set_u64(&video_in, &mf::MF_MT_FRAME_SIZE, pack32(width, height))
                    .map_err(|e| encode_error("frame size in", e))?;
                set_u64(&video_in, &mf::MF_MT_FRAME_RATE, pack32(fps, 1)).map_err(|e| encode_error("frame rate in", e))?;
                set_u64(&video_in, &mf::MF_MT_PIXEL_ASPECT_RATIO, pack32(1, 1))
                    .map_err(|e| encode_error("pixel aspect in", e))?;
                set_u32(&video_in, &mf::MF_MT_INTERLACE_MODE, INTERLACE_PROGRESSIVE)
                    .map_err(|e| encode_error("interlace in", e))?;
                set_u32(&video_in, &mf::MF_MT_DEFAULT_STRIDE, width).map_err(|e| encode_error("stride", e))?;
                set_u32(&video_in, &mf::MF_MT_VIDEO_NOMINAL_RANGE, NOMINAL_RANGE_FULL)
                    .map_err(|e| encode_error("nominal range", e))?;
            }
            let video_index = unsafe { writer.AddStream(&video_out).map_err(|e| encode_error("AddStream(video)", e))? };
            unsafe {
                writer
                    .SetInputMediaType(video_index, &video_in, None::<&mf::IMFAttributes>)
                    .map_err(|e| encode_error("SetInputMediaType(video)", e))?
            };

            // --- audio stream ---
            let mut audio_index = None;
            if with_audio {
                let audio_out =
                    unsafe { mf::MFCreateMediaType().map_err(|e| encode_error("MFCreateMediaType(audio out)", e))? };
                unsafe {
                    set_guid(&audio_out, &mf::MF_MT_MAJOR_TYPE, &mf::MFMediaType_Audio)
                        .map_err(|e| encode_error("audio major", e))?;
                    set_guid(&audio_out, &mf::MF_MT_SUBTYPE, audio_subtype).map_err(|e| encode_error("audio subtype", e))?;
                    set_u32(&audio_out, &mf::MF_MT_AUDIO_BITS_PER_SAMPLE, 16).map_err(|e| encode_error("audio bits", e))?;
                    set_u32(&audio_out, &mf::MF_MT_AUDIO_NUM_CHANNELS, 2).map_err(|e| encode_error("audio channels", e))?;
                    set_u32(&audio_out, &mf::MF_MT_AUDIO_SAMPLES_PER_SECOND, media_types::SAMPLE_RATE)
                        .map_err(|e| encode_error("audio rate", e))?;
                    if *audio_subtype == mf::MFAudioFormat_AAC {
                        set_u32(&audio_out, &mf::MF_MT_AUDIO_AVG_BYTES_PER_SECOND, 25_600)
                            .map_err(|e| encode_error("audio bitrate", e))?;
                    }
                }
                let audio_in = unsafe { mf::MFCreateMediaType().map_err(|e| encode_error("MFCreateMediaType(audio in)", e))? };
                unsafe {
                    set_guid(&audio_in, &mf::MF_MT_MAJOR_TYPE, &mf::MFMediaType_Audio)
                        .map_err(|e| encode_error("audio in major", e))?;
                    set_guid(&audio_in, &mf::MF_MT_SUBTYPE, &mf::MFAudioFormat_Float)
                        .map_err(|e| encode_error("audio in subtype", e))?;
                    set_u32(&audio_in, &mf::MF_MT_AUDIO_NUM_CHANNELS, 2).map_err(|e| encode_error("audio in channels", e))?;
                    set_u32(&audio_in, &mf::MF_MT_AUDIO_SAMPLES_PER_SECOND, media_types::SAMPLE_RATE)
                        .map_err(|e| encode_error("audio in rate", e))?;
                    set_u32(&audio_in, &mf::MF_MT_AUDIO_BITS_PER_SAMPLE, 32).map_err(|e| encode_error("audio in bits", e))?;
                    set_u32(&audio_in, &mf::MF_MT_AUDIO_BLOCK_ALIGNMENT, 8)
                        .map_err(|e| encode_error("audio block align", e))?;
                    set_u32(&audio_in, &mf::MF_MT_AUDIO_AVG_BYTES_PER_SECOND, media_types::SAMPLE_RATE * 8)
                        .map_err(|e| encode_error("audio in avg bytes", e))?;
                }
                audio_index = Some(unsafe { writer.AddStream(&audio_out).map_err(|e| encode_error("AddStream(audio)", e))? });
                unsafe {
                    writer
                        .SetInputMediaType(audio_index.unwrap(), &audio_in, None::<&mf::IMFAttributes>)
                        .map_err(|e| encode_error("SetInputMediaType(audio)", e))?;
                }
            }

            Ok(Self { writer, video_index, audio_index, width, height, started: false, audio_codec: codec_name.to_string() })
        }

        pub fn write_video(&mut self, nv12: &[u8], frame: u32, fps: u32) -> Result<(), Error> {
            if !self.started {
                unsafe { self.writer.BeginWriting().map_err(|e| encode_error("BeginWriting", e))? };
                self.started = true;
            }
            let expected = (self.width * self.height * 3 / 2) as usize;
            if nv12.len() != expected {
                return Err(Error::encode(format!("NV12 buffer {} bytes, expected {expected}", nv12.len())));
            }
            let unit = 10_000_000i64;
            unsafe {
                let sample = mf::MFCreateSample().map_err(|e| encode_error("MFCreateSample", e))?;
                let buffer =
                    mf::MFCreateMemoryBuffer(nv12.len() as u32).map_err(|e| encode_error("MFCreateMemoryBuffer", e))?;
                let mut ptr: *mut u8 = std::ptr::null_mut();
                buffer.Lock(&mut ptr, None, None).map_err(|e| encode_error("Lock", e))?;
                std::ptr::copy_nonoverlapping(nv12.as_ptr(), ptr, nv12.len());
                buffer.Unlock().map_err(|e| encode_error("Unlock", e))?;
                buffer.SetCurrentLength(nv12.len() as u32).map_err(|e| encode_error("SetCurrentLength", e))?;
                sample.AddBuffer(&buffer).map_err(|e| encode_error("AddBuffer", e))?;
                sample.SetSampleTime(frame as i64 * unit / fps as i64).map_err(|e| encode_error("SetSampleTime", e))?;
                sample.SetSampleDuration(unit / fps as i64).map_err(|e| encode_error("SetSampleDuration", e))?;
                self.writer.WriteSample(self.video_index, &sample).map_err(|e| encode_error("WriteSample(video)", e))?;
            }
            Ok(())
        }

        pub fn write_audio(&mut self, pcm: &[i16], start_sample: u64) -> Result<(), Error> {
            let Some(index) = self.audio_index else {
                return Ok(());
            };
            if !self.started {
                unsafe { self.writer.BeginWriting().map_err(|e| encode_error("BeginWriting", e))? };
                self.started = true;
            }
            let floats: Vec<f32> = pcm.iter().map(|s| *s as f32 / 32768.0).collect();
            let payload = floats.len() * 4;
            unsafe {
                let sample = mf::MFCreateSample().map_err(|e| encode_error("MFCreateSample(audio)", e))?;
                let buffer =
                    mf::MFCreateMemoryBuffer(payload as u32).map_err(|e| encode_error("MFCreateMemoryBuffer(audio)", e))?;
                let mut ptr: *mut u8 = std::ptr::null_mut();
                buffer.Lock(&mut ptr, None, None).map_err(|e| encode_error("Lock(audio)", e))?;
                std::ptr::copy_nonoverlapping(bytemuck_f32(&floats).as_ptr(), ptr, payload);
                buffer.Unlock().map_err(|e| encode_error("Unlock(audio)", e))?;
                buffer.SetCurrentLength(payload as u32).map_err(|e| encode_error("SetCurrentLength(audio)", e))?;
                sample.AddBuffer(&buffer).map_err(|e| encode_error("AddBuffer(audio)", e))?;
                let rate = media_types::SAMPLE_RATE as i64;
                sample
                    .SetSampleTime(start_sample as i64 * 10_000_000 / rate)
                    .map_err(|e| encode_error("SetSampleTime(audio)", e))?;
                sample
                    .SetSampleDuration(pcm.len() as i64 * 10_000_000 / 2 / rate)
                    .map_err(|e| encode_error("SetSampleDuration(audio)", e))?;
                self.writer.WriteSample(index, &sample).map_err(|e| encode_error("WriteSample(audio)", e))?;
            }
            Ok(())
        }

        pub(super) fn audio_codec(&self) -> &str {
            &self.audio_codec
        }

        pub fn finish(&mut self) -> Result<(), Error> {
            unsafe {
                self.writer.Finalize().map_err(|e| encode_error("Finalize", e))?;
                mf::MFShutdown().map_err(|e| encode_error("MFShutdown", e))?;
            }
            Ok(())
        }
    }

    fn bytemuck_f32(samples: &[f32]) -> &[u8] {
        unsafe { std::slice::from_raw_parts(samples.as_ptr() as *const u8, samples.len() * 4) }
    }
}

#[cfg(windows)]
use windows_encoder::WindowsEncoder;
