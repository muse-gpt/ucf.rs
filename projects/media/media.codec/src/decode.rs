//! Media import: still images, video frames and audio files decoded through
//! the codecs Windows already provides.
//!
//! Images go through WIC; video and audio go through a Media Foundation source
//! reader. Both are decoded into the renderer's own RGBA8 / f32 f32
//! representations, so nothing downstream knows what produced the pixels.
//! Decoded results are cached per path and video keeps its decoder position, so
//! playing a clip forward decodes each sample exactly once.

use std::collections::HashMap;
use std::rc::Rc;

use media_image::Bitmap;
use media_types::Error;

/// Per-render cache of imported media.
#[derive(Default)]
pub struct MediaLibrary {
    images: HashMap<String, Rc<Bitmap>>,
    audio: HashMap<(String, u32), Rc<Vec<[f32; 2]>>>,
    #[cfg(windows)]
    videos: HashMap<String, win::VideoReader>,
}

impl MediaLibrary {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Decodes a still image once and returns the cached bitmap.
    pub fn image(&mut self, path: &str) -> Result<Rc<Bitmap>, Error> {
        if let Some(hit) = self.images.get(path) {
            return Ok(Rc::clone(hit));
        }
        let bitmap = Rc::new(decode_image(path)?);
        self.images.insert(path.to_string(), Rc::clone(&bitmap));
        Ok(bitmap)
    }

    /// Returns the frame of `path` covering `seconds` from the start of the file.
    pub fn video_frame(&mut self, path: &str, seconds: f64) -> Result<Rc<Bitmap>, Error> {
        #[cfg(windows)]
        {
            if !self.videos.contains_key(path) {
                let reader = win::VideoReader::open(path)?;
                self.videos.insert(path.to_string(), reader);
            }
            let reader = self.videos.get_mut(path).expect("video reader just inserted");
            reader.frame_at(seconds)
        }
        #[cfg(not(windows))]
        {
            let _ = seconds;
            Err(Error::unsupported(format!("video import is unavailable on this platform: {path}")))
        }
    }

    /// Decodes an audio file to stereo f32 samples at the requested rate.
    ///
    /// The rate is an input contract: the decoder is told which rate to
    /// produce, so the mixer never has to guess where samples came from.
    pub fn audio_pcm(&mut self, path: &str, sample_rate: u32) -> Result<Rc<Vec<[f32; 2]>>, Error> {
        let key = (path.to_string(), sample_rate);
        if let Some(hit) = self.audio.get(&key) {
            return Ok(Rc::clone(hit));
        }
        let samples = Rc::new(decode_audio(path, sample_rate)?);
        self.audio.insert(key, Rc::clone(&samples));
        Ok(samples)
    }
}

#[cfg(windows)]
fn decode_image(path: &str) -> Result<Bitmap, Error> {
    win::decode_image(path)
}

#[cfg(not(windows))]
fn decode_image(path: &str) -> Result<Bitmap, Error> {
    Err(Error::unsupported(format!("image import is unavailable on this platform: {path}")))
}

#[cfg(windows)]
fn decode_audio(path: &str, sample_rate: u32) -> Result<Vec<[f32; 2]>, Error> {
    win::decode_audio(path, sample_rate)
}

#[cfg(not(windows))]
fn decode_audio(path: &str, sample_rate: u32) -> Result<Vec<[f32; 2]>, Error> {
    let _ = sample_rate;
    Err(Error::unsupported(format!("audio import is unavailable on this platform: {path}")))
}

#[cfg(windows)]
mod win {
    use std::rc::Rc;
    use std::sync::Once;

    use windows::Win32::Foundation::GENERIC_READ;
    use windows::Win32::Graphics::Imaging::{
        CLSID_WICImagingFactory, GUID_WICPixelFormat32bppRGBA, IWICFormatConverter, IWICImagingFactory,
        WICBitmapDitherTypeNone, WICBitmapPaletteTypeCustom, WICDecodeMetadataCacheOnDemand, WICRect,
    };
    use windows::Win32::Media::MediaFoundation as mf;
    use windows::Win32::System::Com::{CLSCTX_INPROC_SERVER, COINIT_MULTITHREADED, CoCreateInstance, CoInitializeEx};
    use windows::core::PCWSTR;

    use media_image::Bitmap;
    use media_types::Error;

    static COM_ONCE: Once = Once::new();
    static MF_ONCE: Once = Once::new();
    /// `MF_SOURCE_READERF_ENDOFSTREAM`
    const FLAG_END_OF_STREAM: u32 = 2;

    fn media_error(context: &str, error: windows::core::Error) -> Error {
        Error::media(format!("{context}: {error}"))
    }

    fn wide(path: &str) -> Vec<u16> {
        use std::os::windows::ffi::OsStrExt;
        std::ffi::OsStr::new(path).encode_wide().chain(std::iter::once(0)).collect()
    }

    /// Media Foundation resolves URLs, not shell-relative paths, so relative
    /// script paths are made absolute first.
    fn absolute(path: &str) -> String {
        std::path::absolute(path).map(|p| p.to_string_lossy().into_owned()).unwrap_or_else(|_| path.to_string())
    }

    fn ensure_com() {
        COM_ONCE.call_once(|| unsafe {
            let _ = CoInitializeEx(None, COINIT_MULTITHREADED);
        });
    }

    fn ensure_mf() {
        ensure_com();
        MF_ONCE.call_once(|| unsafe {
            let _ = mf::MFStartup(mf::MF_VERSION, mf::MFSTARTUP_FULL);
        });
    }

    pub(super) fn decode_image(path: &str) -> Result<Bitmap, Error> {
        ensure_com();
        unsafe {
            let factory: IWICImagingFactory = CoCreateInstance(&CLSID_WICImagingFactory, None, CLSCTX_INPROC_SERVER)
                .map_err(|e| media_error("CoCreateInstance(WIC)", e))?;
            let path_w = wide(&absolute(path));
            let decoder = factory
                .CreateDecoderFromFilename(PCWSTR(path_w.as_ptr()), None, GENERIC_READ, WICDecodeMetadataCacheOnDemand)
                .map_err(|e| media_error(&format!("decode image {path}"), e))?;
            let frame = decoder.GetFrame(0).map_err(|e| media_error("WIC GetFrame", e))?;

            let mut width = 0u32;
            let mut height = 0u32;
            frame.GetSize(&mut width, &mut height).map_err(|e| media_error("WIC GetSize", e))?;
            if width == 0 || height == 0 {
                return Err(Error::media(format!("image {path} has zero size")));
            }

            let converter: IWICFormatConverter =
                factory.CreateFormatConverter().map_err(|e| media_error("WIC CreateFormatConverter", e))?;
            converter
                .Initialize(
                    &frame,
                    &GUID_WICPixelFormat32bppRGBA,
                    WICBitmapDitherTypeNone,
                    None,
                    0.0,
                    WICBitmapPaletteTypeCustom,
                )
                .map_err(|e| media_error("WIC Initialize", e))?;

            let stride = (width as usize) * 4;
            let mut pixels = vec![0u8; stride * height as usize];
            let rect = WICRect { X: 0, Y: 0, Width: width as i32, Height: height as i32 };
            converter.CopyPixels(&rect, stride as u32, &mut pixels).map_err(|e| media_error("WIC CopyPixels", e))?;
            Ok(Bitmap::new(width, height, pixels))
        }
    }

    /// One open video source plus the position it last decoded.
    pub(super) struct VideoReader {
        reader: mf::IMFSourceReader,
        stream: u32,
        path: String,
        /// Frame size the file declares; decoded buffers may be padded larger.
        width: u32,
        height: u32,
        fps: f64,
        last_time: f64,
        last_frame: Option<Rc<Bitmap>>,
    }

    impl VideoReader {
        pub(super) fn open(path: &str) -> Result<Self, Error> {
            ensure_mf();
            let (reader, width, height, fps) = Self::configure(path)?;
            Ok(Self {
                reader,
                stream: mf::MF_SOURCE_READER_FIRST_VIDEO_STREAM.0 as u32,
                path: path.to_string(),
                width,
                height,
                fps,
                last_time: -1.0,
                last_frame: None,
            })
        }

        /// Opens a source reader and pins the video output to RGB32.
        fn configure(path: &str) -> Result<(mf::IMFSourceReader, u32, u32, f64), Error> {
            let path_w = wide(&absolute(path));
            // The reader only converts NV12 to RGB32 when video processing is
            // enabled, which is what inserts the video processor MFT.
            let attributes = unsafe {
                let mut attributes: Option<mf::IMFAttributes> = None;
                mf::MFCreateAttributes(&mut attributes, 1).map_err(|e| media_error("MFCreateAttributes", e))?;
                let attributes = attributes.ok_or_else(|| Error::media("MFCreateAttributes returned nothing"))?;
                attributes
                    .SetUINT32(&mf::MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING, 1)
                    .map_err(|e| media_error("enable video processing", e))?;
                attributes
            };
            let reader = unsafe {
                mf::MFCreateSourceReaderFromURL(PCWSTR(path_w.as_ptr()), &attributes)
                    .map_err(|e| media_error(&format!("open video {path}"), e))?
            };
            let stream = mf::MF_SOURCE_READER_FIRST_VIDEO_STREAM.0 as u32;
            unsafe {
                let _ = reader.SetStreamSelection(mf::MF_SOURCE_READER_ALL_STREAMS.0 as u32, false);
                reader
                    .SetStreamSelection(stream, true)
                    .map_err(|e| media_error(&format!("video {path} has no video stream"), e))?;
            }
            let native =
                unsafe { reader.GetNativeMediaType(stream, 0) }.map_err(|e| media_error("video native media type", e))?;
            let size = unsafe { native.GetUINT64(&mf::MF_MT_FRAME_SIZE) }.unwrap_or(0);
            let width = (size >> 32) as u32;
            let height = (size & 0xffff_ffff) as u32;
            // Rational frame rate, packed as numerator << 32 | denominator.
            let rate = unsafe { native.GetUINT64(&mf::MF_MT_FRAME_RATE) }.unwrap_or(0);
            let numerator = (rate >> 32) as u32;
            let denominator = (rate & 0xffff_ffff) as u32;
            let fps = if numerator > 0 && denominator > 0 { f64::from(numerator) / f64::from(denominator) } else { 30.0 };
            if width == 0 || height == 0 {
                return Err(Error::media(format!("video {path} reports no frame size")));
            }

            let target = unsafe { mf::MFCreateMediaType().map_err(|e| media_error("MFCreateMediaType", e))? };
            unsafe {
                target.SetGUID(&mf::MF_MT_MAJOR_TYPE, &mf::MFMediaType_Video).map_err(|e| media_error("video major", e))?;
                target.SetGUID(&mf::MF_MT_SUBTYPE, &mf::MFVideoFormat_RGB32).map_err(|e| media_error("video subtype", e))?;
                target
                    .SetUINT64(&mf::MF_MT_FRAME_SIZE, ((width as u64) << 32) | height as u64)
                    .map_err(|e| media_error("video frame size", e))?;
                reader
                    .SetCurrentMediaType(stream, None, &target)
                    .map_err(|e| media_error(&format!("video {path} cannot be decoded as RGB32"), e))?;
            }
            Ok((reader, width, height, fps))
        }

        /// Frame covering `seconds`. Requests that move forward reuse the decoder
        /// position; moving backwards reopens the source and skips ahead.
        pub(super) fn frame_at(&mut self, seconds: f64) -> Result<Rc<Bitmap>, Error> {
            let frame_span = 1.0 / self.fps;
            if self.last_time >= 0.0 && seconds < self.last_time {
                let (reader, _, _, _) = Self::configure(&self.path)?;
                self.reader = reader;
                self.last_time = -1.0;
                self.last_frame = None;
            }
            // Fast path: the frame we already hold still covers this request.
            if let Some(frame) = &self.last_frame {
                if self.last_time >= 0.0 && seconds - self.last_time < frame_span * 0.98 {
                    return Ok(Rc::clone(frame));
                }
            }

            loop {
                let mut flags = 0u32;
                let mut timestamp = 0i64;
                let mut sample: Option<mf::IMFSample> = None;
                unsafe {
                    self.reader
                        .ReadSample(self.stream, 0, None, Some(&mut flags), Some(&mut timestamp), Some(&mut sample))
                        .map_err(|e| media_error("ReadSample(video)", e))?;
                }
                if flags & FLAG_END_OF_STREAM != 0 {
                    // Hold the last decoded frame rather than emitting nothing.
                    return self
                        .last_frame
                        .clone()
                        .ok_or_else(|| Error::media(format!("video {} has no decodable frames", self.path)));
                }
                let Some(sample) = sample else { continue };
                let bitmap = Rc::new(self.to_bitmap(&sample)?);
                self.last_time = timestamp as f64 / 10_000_000.0;
                self.last_frame = Some(Rc::clone(&bitmap));
                if self.last_time + frame_span > seconds {
                    return Ok(bitmap);
                }
            }
        }

        fn to_bitmap(&self, sample: &mf::IMFSample) -> Result<Bitmap, Error> {
            let current =
                unsafe { self.reader.GetCurrentMediaType(self.stream) }.map_err(|e| media_error("video current type", e))?;
            let size = unsafe { current.GetUINT64(&mf::MF_MT_FRAME_SIZE) }.unwrap_or(0);
            let width = (size >> 32) as u32;
            let height = (size & 0xffff_ffff) as u32;
            let buffer = unsafe { sample.ConvertToContiguousBuffer() }.map_err(|e| media_error("video buffer", e))?;
            let mut ptr: *mut u8 = std::ptr::null_mut();
            let mut len = 0u32;
            unsafe {
                buffer.Lock(&mut ptr, None, Some(&mut len)).map_err(|e| media_error("Lock(video)", e))?;
            }
            let slice = unsafe { std::slice::from_raw_parts(ptr, len as usize) };
            // RGB32 buffers are BGRA and may be padded per row.
            let stride = if height > 0 && len as usize >= (width as usize) * (height as usize) * 4 {
                len as usize / height as usize
            } else {
                (width as usize) * 4
            };
            let bitmap = Bitmap::from_bgra(width, height, slice, stride);
            unsafe {
                let _ = buffer.Unlock();
            }
            Ok(self.crop(bitmap))
        }

        /// Trims macroblock padding: decoded buffers are aligned up to 16 rows,
        /// so a 120-row frame arrives with 128 rows.
        fn crop(&self, bitmap: Bitmap) -> Bitmap {
            if bitmap.width == self.width && bitmap.height == self.height {
                return bitmap;
            }
            let row = (self.width as usize) * 4;
            let mut rgba = Vec::with_capacity(row * self.height as usize);
            for y in 0..self.height as usize {
                let start = y * (bitmap.width as usize) * 4;
                let end = start + row.min(bitmap.rgba.len().saturating_sub(start));
                rgba.extend_from_slice(&bitmap.rgba[start..end]);
                // Pad short rows so the buffer always matches the declared size.
                rgba.resize(row * (y + 1), 0);
            }
            Bitmap::new(self.width, self.height, rgba)
        }
    }

    /// Decodes the first audio stream at the requested rate, stereo float.
    pub(super) fn decode_audio(path: &str, sample_rate: u32) -> Result<Vec<[f32; 2]>, Error> {
        ensure_mf();
        let path_w = wide(&absolute(path));
        let reader = unsafe {
            mf::MFCreateSourceReaderFromURL(PCWSTR(path_w.as_ptr()), None::<&mf::IMFAttributes>)
                .map_err(|e| media_error(&format!("open audio {path}"), e))?
        };
        let stream = mf::MF_SOURCE_READER_FIRST_AUDIO_STREAM.0 as u32;
        unsafe {
            let _ = reader.SetStreamSelection(mf::MF_SOURCE_READER_ALL_STREAMS.0 as u32, false);
        }
        unsafe {
            reader
                .SetStreamSelection(stream, true)
                .map_err(|e| media_error(&format!("audio {path} has no audio stream"), e))?;
        }

        let target = unsafe { mf::MFCreateMediaType().map_err(|e| media_error("MFCreateMediaType(audio)", e))? };
        unsafe {
            target.SetGUID(&mf::MF_MT_MAJOR_TYPE, &mf::MFMediaType_Audio).map_err(|e| media_error("audio major", e))?;
            target.SetGUID(&mf::MF_MT_SUBTYPE, &mf::MFAudioFormat_Float).map_err(|e| media_error("audio subtype", e))?;
            target.SetUINT32(&mf::MF_MT_AUDIO_NUM_CHANNELS, 2).map_err(|e| media_error("audio channels", e))?;
            target.SetUINT32(&mf::MF_MT_AUDIO_SAMPLES_PER_SECOND, sample_rate).map_err(|e| media_error("audio rate", e))?;
            target.SetUINT32(&mf::MF_MT_AUDIO_BITS_PER_SAMPLE, 32).map_err(|e| media_error("audio bits", e))?;
            target.SetUINT32(&mf::MF_MT_AUDIO_BLOCK_ALIGNMENT, 8).map_err(|e| media_error("audio block align", e))?;
            target
                .SetUINT32(&mf::MF_MT_AUDIO_AVG_BYTES_PER_SECOND, sample_rate * 8)
                .map_err(|e| media_error("audio avg bytes", e))?;
            reader
                .SetCurrentMediaType(stream, None, &target)
                .map_err(|e| media_error(&format!("audio {path} cannot be decoded to PCM"), e))?;
        }

        let mut out: Vec<[f32; 2]> = Vec::new();
        loop {
            let mut flags = 0u32;
            let mut sample: Option<mf::IMFSample> = None;
            unsafe {
                reader
                    .ReadSample(stream, 0, None, Some(&mut flags), None, Some(&mut sample))
                    .map_err(|e| media_error("ReadSample(audio)", e))?;
            }
            if flags & FLAG_END_OF_STREAM != 0 {
                break;
            }
            let Some(sample) = sample else { continue };
            let buffer = unsafe { sample.ConvertToContiguousBuffer() }.map_err(|e| media_error("audio buffer", e))?;
            let mut ptr: *mut u8 = std::ptr::null_mut();
            let mut len = 0u32;
            unsafe {
                buffer.Lock(&mut ptr, None, Some(&mut len)).map_err(|e| media_error("Lock(audio)", e))?;
            }
            let bytes = unsafe { std::slice::from_raw_parts(ptr, len as usize) };
            for chunk in bytes.as_chunks::<8>().0 {
                let left = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                let right = f32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]);
                out.push([left, right]);
            }
            unsafe {
                let _ = buffer.Unlock();
            }
        }
        Ok(out)
    }
}
