# media-post

HDR frame buffers and display post-processing for media pipelines.

- Frame storage (`Framebuffer` / `FramebufferEncoding`, linear HDR `geometry-types::LinearRgb`) lives in `media.image`; this crate owns only the processing semantics and their execution.
- `contract`: the written post-processing contract — input domains (`InputDomain`: linear HDR vs display RGBA8), transfer-function stance, straight alpha, clamp-to-edge sampling, short-side-relative blur radius and the single tone-map-at-the-output-boundary rule. Every operator declares the domain it consumes.
- `display_blur`: the one and only separable box blur for display RGBA8 frames, shared by the develop recipe and the effect-stack blur.
- `effect` + `display_ops` + `stack_spec` + `presets`: the ordered effect-stack contract (stateless per moment, `InputDomain`-annotated display-space operators `blur`/`bloom`/`chroma`/`displace`/`sweep`/`grain`), the neutral JSON stack spec (`StackSpec` → validate → compile → apply, empty stack legal, out-of-domain parameters rejected with `invalid_input`) and the shipped preset catalogue. Particles and beams evaluation lives in `media.particles`; this crate adapts them into the stack.
- `post`: CPU reference passes — ACES filmic tone map, threshold bright extraction, multi-scale bloom, and the one-time `to_rgba8` display conversion. `DEFAULT_BLOOM_THRESHOLD` / `DEFAULT_BLOOM_INTENSITY` are the single source of bloom defaults for both the CPU and the UCF path.
- `post_ucf`: the same passes executed through UCF portable programs (`hardware-program`), kept parity-tested against the CPU reference.
- Program builders: `HdrBrightExtractBuilder`, `UcfBoxBlur3x3Builder`, `UcfDownsample2xRgbBuilder`, `UcfUpsampleRgbBuilder`, `tone_map_aces_batch`, plus the shared `tone_map_aces` / `post_bloom_extract` prelude programs re-exported from `hardware-program`.

Parts of this crate originate from the TON blackhole renderer (MIT, © TON Team), redistributed under MPL-2.0.
