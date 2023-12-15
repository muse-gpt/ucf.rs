# media-post

HDR frame buffers and display post-processing for media pipelines.

- Frame storage (`Framebuffer` / `FramebufferEncoding`, linear HDR `geometry-types::LinearRgb`) lives in `media.image`; this crate owns only the processing semantics and their execution.
- `post`: CPU reference passes — ACES filmic tone map, threshold bright extraction, multi-scale bloom, and the one-time `to_rgba8` display conversion.
- `post_ucf`: the same passes executed through UCF portable programs (`hardware-program`), kept parity-tested against the CPU reference.
- Program builders: `HdrBrightExtractBuilder`, `UcfBoxBlur3x3Builder`, `UcfDownsample2xRgbBuilder`, `UcfUpsampleRgbBuilder`, `tone_map_aces_batch`, plus the shared `tone_map_aces` / `post_bloom_extract` prelude programs re-exported from `hardware-program`.

Parts of this crate originate from the TON blackhole renderer (MIT, © TON Team), redistributed under MPL-2.0.
