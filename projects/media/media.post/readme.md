# media-post

HDR frame buffers and display post-processing for media pipelines.

- `Framebuffer` / `FramebufferEncoding`: linear HDR intermediate frames (`geometry-types::LinearRgb`), distinct from the 8-bit display `Bitmap` in `media.raster`.
- `post`: CPU reference passes — ACES filmic tone map, threshold bright extraction, multi-scale bloom.
- `post_ucf`: the same passes executed through UCF portable programs (`hardware-program`), kept parity-tested against the CPU reference.
- Program builders: `HdrBrightExtractBuilder`, `UcfBoxBlur3x3Builder`, `UcfDownsample2xRgbBuilder`, `UcfUpsampleRgbBuilder`, `tone_map_aces_batch`, plus the shared `tone_map_aces` / `post_bloom_extract` prelude programs re-exported from `hardware-program`.

Parts of this crate originate from the TON blackhole renderer (MIT, © TON Team), redistributed under MPL-2.0.
