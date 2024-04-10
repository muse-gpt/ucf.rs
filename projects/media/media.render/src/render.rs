//! Frame orchestration: compile the scene IR into pixels and push them
//! through the encoder together with the synthesized audio bed.

use std::path::{Path, PathBuf};

use geometry::camera_at;
use geometry::smooth_polyline;
use geometry::{Rgba, parse, parse_or};
use geometry::{apply, fade};
use media_audio::{
    AudioPlan, BgmPlan, ClipPlacement, InstrumentRequest, NarrationPlacement, NotePlacement, PatternPlacement,
    PatternRequest, PatternTrackRequest, SfxPlacement, SfxRequest, VoicePlacement, build_audio,
};
use media_codec::DecodeLimits;
use media_codec::Encoder;
use media_codec::MediaLibrary;
use media_image::{Bitmap, Frame};
use media_motion::Motion;
use media_post::Vignette;
use media_raster::Paint;
use media_scene::{Align, Background, Blend, Component, Item, Node, Project, ShapeKind};
use media_text::Fonts;
use media_types::Error;
use media_types::hash;

/// Facts about a finished render.
///
/// The stats are the render's provenance: they name the exact frame range,
/// the font files actually loaded and the hash of the project that produced
/// the file, so a success can be traced back to its inputs.
#[derive(serde::Serialize, Debug)]
pub struct RenderStats {
    pub frames: u32,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub seconds: f64,
    pub out: String,
    pub bytes: u64,
    pub backend: String,
    pub audio: bool,
    /// Half-open frame range written, in project frames.
    pub range: [u32; 2],
    /// Font files the render actually loaded, in role order.
    pub fonts: Vec<String>,
    /// FNV-1a hash of the project JSON that was rendered.
    pub project_hash: String,
    /// Declared optional resources that were absent; each is documented silence.
    pub optional_missing: Vec<String>,
}

/// A half-open range of project frames to render.
///
/// Frames are renumbered from zero in the output, so an interval export is a
/// standalone file that plays from its own start.
#[derive(Clone, Copy, Debug)]
pub struct FrameRange {
    pub start: u32,
    pub end: u32,
}

/// Execution options for one render.
///
/// `range` selects an interval and `cancel` is polled between frames; both are
/// explicit so a host never has to guess whether a run can be stopped or
/// trimmed. The default is the whole film, uncancellable.
#[derive(Default)]
pub struct RenderOptions<'a> {
    pub range: Option<FrameRange>,
    pub cancel: Option<&'a dyn Fn() -> bool>,
}

impl RenderOptions<'_> {
    fn cancelled(&self) -> bool {
        self.cancel.is_some_and(|poll| poll())
    }
}

/// Removes a partially written artifact unless it was published.
///
/// Declared before the encoder so it drops *after* it: the writer releases the
/// file first, then this guard deletes it.
struct PartialOutput {
    path: PathBuf,
    published: bool,
}

impl PartialOutput {
    fn new(out: &Path) -> Self {
        // The encoder picks its container from the extension, so the partial
        // file keeps `.mp4` and only gains a distinguishing suffix.
        let name = out.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "out.mp4".to_string());
        let stem = name.strip_suffix(".mp4").unwrap_or(name.as_str());
        let path = out.with_file_name(format!("{stem}.{}.partial.mp4", std::process::id()));
        Self { path, published: false }
    }

    fn publish(&mut self, out: &Path) -> Result<(), Error> {
        std::fs::rename(&self.path, out)
            .map_err(|e| Error::io(format!("cannot publish {} -> {}: {e}", self.path.display(), out.display())))?;
        self.published = true;
        Ok(())
    }
}

impl Drop for PartialOutput {
    fn drop(&mut self) {
        if self.published {
            return;
        }
        // The sink writer may still be releasing the file when this runs, so
        // removal is retried briefly instead of silently leaving a stray file.
        for attempt in 0..12 {
            match std::fs::remove_file(&self.path) {
                Ok(()) => return,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => return,
                Err(_) => std::thread::sleep(std::time::Duration::from_millis(10 * (attempt + 1))),
            }
        }
    }
}

/// The system font candidates, per role. A project that declares `fonts` never
/// reaches this list. Order prefers Windows CJK faces, then DejaVu / Liberation
/// on Linux CI runners, then Arial / Menlo-family faces on macOS.
const SYSTEM_FONTS: [(&str, &[&str]); 3] = [
    (
        "regular",
        &[
            "C:\\Windows\\Fonts\\msyh.ttc",
            "C:\\Windows\\Fonts\\NotoSansCJK-Regular.ttc",
            "C:\\Windows\\Fonts\\arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/Library/Fonts/Arial.ttf",
        ],
    ),
    (
        "bold",
        &[
            "C:\\Windows\\Fonts\\msyhbd.ttc",
            "C:\\Windows\\Fonts\\NotoSansCJK-Bold.ttc",
            "C:\\Windows\\Fonts\\arialbd.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/Library/Fonts/Arial Bold.ttf",
        ],
    ),
    (
        "mono",
        &[
            "C:\\Windows\\Fonts\\consola.ttf",
            "C:\\Windows\\Fonts\\DejaVuSansMono.ttf",
            "C:\\Windows\\Fonts\\cour.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
            "/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf",
            "/System/Library/Fonts/Menlo.ttc",
            "/System/Library/Fonts/Monaco.ttf",
            "/System/Library/Fonts/Supplemental/Courier New.ttf",
            "/Library/Fonts/Courier New.ttf",
        ],
    ),
];

/// Loads the three faces and returns the paths that were used.
///
/// Declared project fonts are an input contract: a missing or undecodable file
/// is an error, never a silent swap to another face. Without a declaration the
/// host probes the system candidates and still reports which files it loaded.
fn resolve_fonts(project: &Project) -> Result<(Fonts, Vec<String>), Error> {
    let (bytes, used) = match &project.fonts {
        Some(spec) => {
            let mut bytes = Vec::with_capacity(3);
            let mut used = Vec::with_capacity(3);
            for (role, path) in spec.faces() {
                let data = std::fs::read(path)
                    .map_err(|e| Error::io(format!("project font `{role}` at {path} cannot be read: {e}")))?;
                bytes.push(data);
                used.push(path.to_string());
            }
            (bytes, used)
        }
        None => {
            let mut bytes = Vec::with_capacity(3);
            let mut used = Vec::with_capacity(3);
            for (role, candidates) in SYSTEM_FONTS {
                let mut hit = None;
                for path in candidates {
                    if let Ok(data) = std::fs::read(path) {
                        hit = Some((data, *path));
                        break;
                    }
                }
                let Some((data, path)) = hit else {
                    return Err(Error::io(format!(
                        "no {role} font found among: {}; declare `fonts` in the project",
                        candidates.join(", ")
                    )));
                };
                bytes.push(data);
                used.push(path.to_string());
            }
            (bytes, used)
        }
    };
    let fonts = Fonts::from_bytes(&bytes[0], &bytes[1], &bytes[2])?;
    Ok((fonts, used))
}

fn fnv1a(bytes: &[u8]) -> String {
    let mut state = 0xcbf2_9ce4_8422_2325u64;
    for byte in bytes {
        state ^= u64::from(*byte);
        state = state.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{state:016x}")
}

/// Parses, validates and renders a project JSON string into `out`.
pub fn render_project(json: &str, out: &Path, progress: &mut dyn FnMut(u32, u32)) -> Result<RenderStats, Error> {
    render_project_with(json, out, &RenderOptions::default(), progress)
}

/// Renders `out` with explicit range and cancellation.
///
/// The artifact is written to a sibling partial file and renamed only after the
/// encoder closes cleanly, so a cancelled or failed run never replaces the last
/// good output.
pub fn render_project_with(
    json: &str,
    out: &Path,
    options: &RenderOptions<'_>,
    progress: &mut dyn FnMut(u32, u32),
) -> Result<RenderStats, Error> {
    let project: Project = serde_json::from_str(json).map_err(|e| Error::invalid(format!("project json: {e}")))?;
    validate(&project)?;
    let (w, h, fps) = (project.width, project.height, project.fps);
    let (fonts, font_paths) = resolve_fonts(&project)?;
    let mut fonts = fonts;
    let mut media = MediaLibrary::new();
    // The project layer's decode budget: this renderer trusts container sizes,
    // hosts with tighter budgets pass their own limits to the codec calls.
    let limits = DecodeLimits::unbounded();
    let range = resolve_range(&project, options)?;
    let (first, last) = (range.start, range.end);
    // 暗角唯一正式语义：`media.post::Vignette`（G7 裁决，旧 raster 掩码已删）。
    // 强度完全来自 `project.post`；未声明就是不压暗。
    let vignette = project.post.vignette.map(Vignette::new);
    let seconds = project.time_base().to_seconds(project.duration_frames);
    let has_audio = project.audio.is_some();
    let plan = lower_audio(&project);
    let optional_missing = preflight(&project, &mut fonts, &mut media, limits, &plan)?;
    let audio = build_audio(&plan, &mut media, limits)?;
    let bitrate = (w * h * fps / 2).clamp(2_000_000, 20_000_000);
    let mut partial = PartialOutput::new(out);
    if let Some(parent) = partial.path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)
            .map_err(|e| Error::io(format!("cannot create output directory {}: {e}", parent.display())))?;
    }
    let mut encoder = Encoder::open(&partial.path, w, h, fps, bitrate, plan.sample_rate, has_audio)?;
    let audio_codec = encoder.audio_codec().to_string();

    let mut frame = Frame::new(w, h);
    // Rasterize the world at the largest zoom the film uses, so a push-in has
    // full-resolution detail instead of a blown-up 1:1 frame.
    let ss = supersample(&project);
    let mut pool = FramePool::new((w as f32 * ss).round() as u32, (h as f32 * ss).round() as u32);
    let total = last - first;
    let mut written_samples = 0usize;
    for f in first..last {
        if options.cancelled() {
            return Err(Error::cancelled(format!("render cancelled at frame {f} of {first}..{last}")));
        }
        paint_frame(&project, &mut fonts, &mut media, vignette.as_ref(), &mut pool, f, &mut frame, limits)?;
        encoder.write_video(&frame.to_nv12(), f - first, fps)?;
        // Interleave audio with video in one-second blocks.
        if (f + 1 - first) % fps == 0 || f + 1 == last {
            let end_samples = project.time_base().to_samples(f + 1, plan.sample_rate);
            let start_samples = project.time_base().to_samples(first, plan.sample_rate);
            let end_samples = end_samples.min(audio.len() / 2).max(start_samples);
            if end_samples > written_samples + start_samples {
                let chunk = &audio[(written_samples + start_samples) * 2..end_samples * 2];
                encoder.write_audio(chunk, written_samples as u64)?;
                written_samples = end_samples - start_samples;
            }
        }
        progress(f + 1 - first, total);
    }
    encoder.finish()?;
    partial.publish(out)?;
    let bytes = std::fs::metadata(out).map(|m| m.len()).unwrap_or(0);
    Ok(RenderStats {
        frames: total,
        width: w,
        height: h,
        fps,
        seconds,
        out: out.display().to_string(),
        bytes,
        backend: if has_audio { format!("media-foundation+h264+{audio_codec}") } else { "media-foundation+h264".to_string() },
        audio: has_audio,
        range: [first, last],
        fonts: font_paths,
        project_hash: fnv1a(json.as_bytes()),
        optional_missing,
    })
}

/// Facts about one previewed frame.
#[derive(serde::Serialize, Debug)]
pub struct PreviewStats {
    pub frame: u32,
    pub width: u32,
    pub height: u32,
    /// Font files the preview loaded, in role order.
    pub fonts: Vec<String>,
    pub project_hash: String,
}

/// Evaluates one project frame and writes it as a PNG.
///
/// This is the preview path. It resolves fonts, runs the same contract checks
/// and paints through the same [`paint_frame`] as an export, so reviewing frame
/// 137 is reviewing the frame the encoder will receive. It does not synthesize
/// the audio bed: the preview is a picture.
pub fn preview_frame(json: &str, frame: u32, out: &Path) -> Result<PreviewStats, Error> {
    let project: Project = serde_json::from_str(json).map_err(|e| Error::invalid(format!("project json: {e}")))?;
    validate(&project)?;
    if frame >= project.duration_frames {
        return Err(Error::invalid(format!("frame {frame} is outside 0..{}", project.duration_frames)));
    }
    let (fonts, font_paths) = resolve_fonts(&project)?;
    let mut fonts = fonts;
    let mut media = MediaLibrary::new();
    let limits = DecodeLimits::unbounded();
    let plan = lower_audio(&project);
    preflight(&project, &mut fonts, &mut media, limits, &plan)?;
    let (w, h) = (project.width, project.height);
    let ss = supersample(&project);
    let mut pool = FramePool::new((w as f32 * ss).round() as u32, (h as f32 * ss).round() as u32);
    let mut frame_buffer = Frame::new(w, h);
    let vignette = project.post.vignette.map(Vignette::new);
    paint_frame(&project, &mut fonts, &mut media, vignette.as_ref(), &mut pool, frame, &mut frame_buffer, limits)?;
    if let Some(parent) = out.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).map_err(|e| Error::io(format!("cannot create {}: {e}", parent.display())))?;
    }
    media_codec::write_png(out, w, h, &frame_buffer.pixels)?;
    Ok(PreviewStats {
        frame,
        width: w,
        height: h,
        fonts: font_paths,
        project_hash: fnv1a(json.as_bytes()),
    })
}

/// Validates the requested interval against the film.
fn resolve_range(project: &Project, options: &RenderOptions<'_>) -> Result<FrameRange, Error> {
    let Some(range) = options.range else {
        return Ok(FrameRange { start: 0, end: project.duration_frames });
    };
    if range.start >= range.end {
        return Err(Error::invalid(format!("frame range {range:?} is empty (start must be below end)")));
    }
    if range.end > project.duration_frames {
        return Err(Error::invalid(format!(
            "frame range ends at {} but the film is {} frames",
            range.end, project.duration_frames
        )));
    }
    Ok(range)
}

/// Lowers the scene's `AudioSpec` into the mixer's neutral track plan.
///
/// The plan carries the project default sample rate; `media.audio` never sees
/// the scene IR, so the audio crate stays free of scene semantics. This is
/// the authoritative lowering: hosts and tests reuse it instead of building
/// plans by hand.
#[must_use]
pub fn lower_audio(project: &Project) -> AudioPlan {
    let spec = project.audio.as_ref();
    AudioPlan {
        sample_rate: media_types::SAMPLE_RATE,
        channels: 2,
        time_base: project.time_base(),
        duration_frames: project.duration_frames,
        bgm: spec.and_then(|a| a.bgm.as_ref()).map(|b| BgmPlan { volume: b.volume, mood: b.mood.clone() }),
        narration: spec.and_then(|a| a.narration.as_ref()).map(|n| NarrationPlacement {
            wav: n.wav.clone(),
            delay_frames: n.delay_frames,
            volume: n.volume,
        }),
        sfx: spec
            .map(|a| {
                a.sfx
                    .iter()
                    .map(|s| SfxPlacement {
                        request: SfxRequest {
                            family: s.kind.clone(),
                            preset: s.preset.clone(),
                            duration: s.duration,
                            pitch: s.pitch,
                            brightness: s.brightness,
                            intensity: s.intensity,
                            pan: s.pan,
                            seed: s.seed,
                        },
                        at_frames: s.at,
                        volume: s.volume,
                    })
                    .collect()
            })
            .unwrap_or_default(),
        voice: spec
            .map(|a| {
                a.voice
                    .iter()
                    .map(|v| VoicePlacement { song: v.song.clone(), at_frames: v.at, volume: v.volume })
                    .collect()
            })
            .unwrap_or_default(),
        notes: spec
            .map(|a| {
                a.notes
                    .iter()
                    .map(|n| NotePlacement {
                        request: InstrumentRequest {
                            instrument: n.instrument.clone(),
                            note: n.note,
                            duration: n.duration,
                            velocity: n.velocity,
                            tone: n.tone,
                            decay: n.decay,
                            pan: n.pan,
                            seed: n.seed,
                        },
                        at_frames: n.at,
                        volume: 1.0,
                    })
                    .collect()
            })
            .unwrap_or_default(),
        patterns: spec
            .map(|a| {
                a.patterns
                    .iter()
                    .map(|p| PatternPlacement {
                        request: PatternRequest {
                            preset: p.preset.clone(),
                            bpm: p.bpm,
                            bars: p.bars,
                            tracks: p
                                .tracks
                                .iter()
                                .map(|t| PatternTrackRequest {
                                    instrument: t.instrument.clone(),
                                    note: t.note,
                                    steps: t.steps.clone(),
                                    tone: t.tone,
                                    decay: t.decay,
                                    pan: t.pan,
                                })
                                .collect(),
                        },
                        at_frames: p.at,
                        volume: p.volume,
                    })
                    .collect()
            })
            .unwrap_or_default(),
        clips: spec
            .map(|a| {
                a.clips
                    .iter()
                    .map(|c| ClipPlacement {
                        src: c.src.clone(),
                        at_frames: c.at,
                        duration_frames: c.dur,
                        offset_frames: c.offset_frames,
                        volume: c.volume,
                        fade_in_frames: 0,
                        fade_out_frames: 0,
                        // Imported audio is a required input unless declared optional.
                        required: c.required,
                    })
                    .collect()
            })
            .unwrap_or_default(),
    }
}

/// Largest zoom the film asks for, clamped to a sane raster budget. The world
/// is drawn at this scale so camera pushes never invent pixels.
fn supersample(project: &Project) -> f32 {
    let mut max_zoom = 1.0f32;
    for key in &project.camera {
        max_zoom = max_zoom.max(key.zoom);
    }
    for scene in &project.scenes {
        if let Some(camera) = &scene.camera {
            max_zoom = max_zoom.max(camera.zoom_from).max(camera.zoom_to);
        }
    }
    max_zoom.clamp(1.0, 2.0)
}

fn validate(project: &Project) -> Result<(), Error> {
    if project.width % 2 == 1 || project.height % 2 == 1 {
        return Err(Error::invalid("width and height must be even for H.264"));
    }
    if project.width < 16 || project.height < 16 {
        return Err(Error::invalid("frame is too small"));
    }
    if project.fps == 0 || project.fps > 240 {
        return Err(Error::invalid("fps must be within 1..=240"));
    }
    if project.duration_frames == 0 {
        return Err(Error::invalid("duration_frames must be positive"));
    }
    if let Some(vignette) = project.post.vignette {
        if !vignette.is_finite() || !(0.0..=1.0).contains(&vignette) {
            return Err(Error::invalid(format!("post.vignette must be within 0..=1, got {vignette}")));
        }
    }
    let mut problems = Vec::new();
    for (index, scene) in project.scenes.iter().enumerate() {
        if scene.duration == 0 {
            return Err(Error::invalid("scene duration must be positive"));
        }
        if scene.start + scene.duration > project.duration_frames {
            return Err(Error::invalid(format!("scene at {} exceeds duration {}", scene.start, project.duration_frames)));
        }
        check_background(index, &scene.background, &mut problems);
        for (slot, node) in scene.items.iter().enumerate() {
            let where_ = format!("scene {index} item {slot}");
            check_node(project, &where_, node, &mut problems);
        }
    }
    for (index, component) in project.components.iter().enumerate() {
        let where_ = format!("component {index} `{}`", component.id);
        for (slot, node) in component.items.iter().enumerate() {
            check_node(project, &format!("{where_} item {slot}"), node, &mut problems);
        }
    }
    for problem in component_cycles(project) {
        problems.push(problem);
    }
    if problems.is_empty() {
        Ok(())
    } else {
        Err(Error::invalid(format!("scene contract: {}", problems.join("; "))))
    }
}

/// Validates one node: its identity, its motion contract and its item.
fn check_node(project: &Project, where_: &str, node: &Node, problems: &mut Vec<String>) {
    check_item_colors(where_, &node.item, problems);
    node.anim.validate(where_, problems);
    // Motion properties are part of the node contract, so a property the item
    // cannot express is a located error instead of a silently ignored track.
    let (supports_position, supports_progress) = match &node.item {
        Item::Bands { .. } | Item::Grain { .. } | Item::Layer { .. } => (false, false),
        Item::Path { .. } => (true, true),
        Item::Shape { shape, .. } => (true, *shape == ShapeKind::Line),
        _ => (true, false),
    };
    if !supports_position && (node.anim.x.is_some() || node.anim.y.is_some()) {
        problems.push(format!("{where_}: this item has no position, so `x`/`y` motion does not apply"));
    }
    if !supports_progress && (node.anim.progress.is_some() || node.anim.follow.is_some()) {
        problems.push(format!("{where_}: this item has no progress, so `progress`/`follow` motion does not apply"));
    }
    if let Item::Layer { items, .. } = &node.item {
        for (slot, child) in items.iter().enumerate() {
            check_node(project, &format!("{where_} layer item {slot}"), child, problems);
        }
    }
    if let Item::Instance { component, required, .. } = &node.item {
        if find_component(project, component).is_none() && *required {
            problems.push(format!("{where_}: instance references unknown component `{component}`"));
        }
    }
}

/// The component definition `id` names, if it exists.
fn find_component<'a>(project: &'a Project, id: &media_scene::NodeId) -> Option<&'a Component> {
    project.components.iter().find(|component| &component.id == id)
}

/// Reports component definitions that reference each other in a cycle.
///
/// A cycle cannot be painted, so it must be rejected while the project is still
/// data rather than discovered as a depth failure halfway through a render.
fn component_cycles(project: &Project) -> Vec<String> {
    fn references(node: &Node, out: &mut Vec<media_scene::NodeId>) {
        match &node.item {
            Item::Instance { component, .. } => out.push(component.clone()),
            Item::Layer { items, .. } => {
                for child in items {
                    references(child, out);
                }
            }
            _ => {}
        }
    }
    fn visit(
        project: &Project,
        id: &media_scene::NodeId,
        path: &mut Vec<media_scene::NodeId>,
        problems: &mut Vec<String>,
    ) {
        if path.contains(id) {
            let chain: Vec<String> = path.iter().map(ToString::to_string).chain(std::iter::once(id.to_string())).collect();
            problems.push(format!("component cycle: {}", chain.join(" -> ")));
            return;
        }
        let Some(component) = find_component(project, id) else {
            return;
        };
        path.push(id.clone());
        for node in &component.items {
            let mut used = Vec::new();
            references(node, &mut used);
            for next in used {
                visit(project, &next, path, problems);
            }
        }
        path.pop();
    }
    let mut problems = Vec::new();
    let mut seen = Vec::new();
    for component in &project.components {
        if seen.contains(&component.id) {
            problems.push(format!("component id `{}` is declared twice", component.id));
            continue;
        }
        seen.push(component.id.clone());
        visit(project, &component.id, &mut Vec::new(), &mut problems);
    }
    problems
}

/// Every declared color must parse. A silent fallback would repaint the film
/// in a color the author never wrote.
fn check_color(where_: &str, value: Option<&str>, problems: &mut Vec<String>) {
    if let Some(value) = value {
        if parse(value).is_none() {
            problems.push(format!("{where_}: `{value}` is not a #rgb/#rrggbb/#rrggbbaa color"));
        }
    }
}

fn check_background(index: usize, background: &Background, problems: &mut Vec<String>) {
    let where_ = format!("scene {index} background");
    match background {
        Background::Solid { color } => check_color(&where_, Some(color), problems),
        Background::VGradient { colors } => {
            if colors.is_empty() {
                problems.push(format!("{where_}: a vertical gradient needs at least one stop"));
            }
            for color in colors {
                check_color(&where_, Some(color), problems);
            }
        }
        Background::Radial { center, edge } => {
            check_color(&where_, Some(center), problems);
            check_color(&where_, Some(edge), problems);
        }
        Background::Starfield { base, tint, .. } => {
            if base.is_empty() {
                problems.push(format!("{where_}: a starfield needs at least one base color"));
            }
            for color in base {
                check_color(&where_, Some(color), problems);
            }
            check_color(&where_, tint.as_deref(), problems);
        }
    }
}

fn check_item_colors(where_: &str, item: &Item, problems: &mut Vec<String>) {
    match item {
        Item::Text { dur, color, sub_color, glow, .. } => {
            check_dur(where_, *dur, problems);
            check_color(where_, Some(color), problems);
            check_color(where_, sub_color.as_deref(), problems);
            check_color(where_, glow.as_deref(), problems);
        }
        Item::Code { dur, theme, .. } => {
            check_dur(where_, *dur, problems);
            for field in [&theme.plain, &theme.keyword, &theme.string, &theme.comment, &theme.number, &theme.function] {
                check_color(where_, Some(field), problems);
            }
            check_color(where_, theme.panel.as_deref(), problems);
        }
        Item::Counter { dur, color, .. } => {
            check_dur(where_, *dur, problems);
            check_color(where_, Some(color), problems);
        }
        Item::Card { dur, accent, panel, .. } => {
            check_dur(where_, *dur, problems);
            check_color(where_, accent.as_deref(), problems);
            check_color(where_, panel.as_deref(), problems);
        }
        Item::Image { dur, .. } | Item::Video { dur, .. } | Item::Instance { dur, .. } => check_dur(where_, *dur, problems),
        Item::Shape { dur, color, .. } => {
            check_dur(where_, *dur, problems);
            check_color(where_, Some(color), problems);
        }
        Item::Bands { dur, color, .. } => {
            check_dur(where_, *dur, problems);
            check_color(where_, Some(color), problems);
        }
        Item::Grain { dur, .. } => check_dur(where_, *dur, problems),
        Item::Path { dur, fill, color, .. } => {
            check_dur(where_, *dur, problems);
            check_color(where_, fill.as_deref(), problems);
            check_color(where_, Some(color), problems);
        }
        Item::Glow { dur, color, .. } => {
            check_dur(where_, *dur, problems);
            check_color(where_, Some(color), problems);
        }
        Item::Layer { dur, items, .. } => {
            check_dur(where_, *dur, problems);
            for (slot, child) in items.iter().enumerate() {
                check_item_colors(&format!("{where_} layer item {slot}"), &child.item, problems);
            }
        }
        Item::Footnote { dur, color, .. } => {
            check_dur(where_, *dur, problems);
            check_color(where_, Some(color), problems);
        }
        Item::Progress { dur, track, fill, .. } => {
            check_dur(where_, *dur, problems);
            check_color(where_, track.as_deref(), problems);
            check_color(where_, fill.as_deref(), problems);
        }
    }
}

fn check_dur(where_: &str, dur: u32, problems: &mut Vec<String>) {
    if dur == 0 {
        problems.push(format!("{where_}: dur must be at least one frame"));
    }
}

/// Preflights every imported resource before a frame is encoded.
///
/// Required resources must exist and decode; optional ones are recorded by name
/// so the finished stats state exactly which declared slots stayed silent.
/// Text is checked against the loaded faces, so a missing glyph fails here
/// instead of rendering as a blank box.
fn preflight(
    project: &Project,
    fonts: &mut Fonts,
    media: &mut MediaLibrary,
    limits: DecodeLimits,
    plan: &AudioPlan,
) -> Result<Vec<String>, Error> {
    let mut failures: Vec<String> = Vec::new();
    let mut optional_missing: Vec<String> = Vec::new();
    let mut decode_failed = false;
    for (index, scene) in project.scenes.iter().enumerate() {
        for (slot, item) in scene.items.iter().enumerate() {
            let where_ = format!("scene {index} item {slot}");
            preflight_item(project, &where_, item, fonts, media, limits, &mut failures, &mut optional_missing, &mut decode_failed);
        }
    }
    // The narration slot is a contract: it was lowered, so it must exist.
    if let Some(narration) = &plan.narration {
        match std::fs::read(&narration.wav).ok().and_then(|bytes| media_codec::parse_wav_pcm16(&bytes)) {
            Some(wav) if wav.sample_rate == plan.sample_rate => {}
            Some(wav) => failures.push(format!(
                "narration `{}` is {} Hz but the film is {} Hz",
                narration.wav, wav.sample_rate, plan.sample_rate
            )),
            None => {
                decode_failed = true;
                failures.push(format!("narration `{}` is missing or not a PCM16 WAV", narration.wav));
            }
        }
    }
    for clip in &plan.clips {
        match media.audio_pcm(&clip.src, plan.sample_rate, limits) {
            Ok(_) => {}
            Err(error) if clip.required => {
                decode_failed = true;
                failures.push(format!("required audio clip `{}`: {error}", clip.src));
            }
            Err(_) => optional_missing.push(clip.src.clone()),
        }
    }
    if failures.is_empty() {
        Ok(optional_missing)
    } else {
        let message = format!("resource preflight failed: {}", failures.join("; "));
        Err(if decode_failed { Error::media(message) } else { Error::invalid(message) })
    }
}

#[allow(clippy::too_many_arguments)]
fn preflight_item(
    project: &Project,
    where_: &str,
    node: &Node,
    fonts: &mut Fonts,
    media: &mut MediaLibrary,
    limits: DecodeLimits,
    failures: &mut Vec<String>,
    optional_missing: &mut Vec<String>,
    decode_failed: &mut bool,
) {
    match &node.item {
        Item::Image { src, required, .. } => match media.image(src, limits) {
            Ok(_) => {}
            Err(error) if *required => {
                *decode_failed = true;
                failures.push(format!("{where_} image `{src}`: {error}"));
            }
            Err(_) => optional_missing.push(src.clone()),
        },
        Item::Video { src, src_in, required, .. } => {
            let seconds = project.time_base().to_seconds(*src_in);
            match media.video_frame(src, seconds, limits) {
                Ok(_) => {}
                Err(error) if *required => {
                    *decode_failed = true;
                    failures.push(format!("{where_} video `{src}`: {error}"));
                }
                Err(_) => optional_missing.push(src.clone()),
            }
        }
        Item::Instance { component, required, .. } => match find_component(project, component) {
            Some(definition) => {
                // Every instance points at the same definition, so its content is
                // preflighted once per render instead of once per placement.
                if !definition.items.is_empty() {
                    for (slot, child) in definition.items.iter().enumerate() {
                        preflight_item(
                            project,
                            &format!("{where_} component `{component}` item {slot}"),
                            child,
                            fonts,
                            media,
                            limits,
                            failures,
                            optional_missing,
                            decode_failed,
                        );
                    }
                }
            }
            None if *required => {
                *decode_failed = true;
                failures.push(format!("{where_} instance references unknown component `{component}`"));
            }
            None => {}
        },
        Item::Layer { items, .. } => {
            for (slot, child) in items.iter().enumerate() {
                preflight_item(
                    project,
                    &format!("{where_} layer item {slot}"),
                    child,
                    fonts,
                    media,
                    limits,
                    failures,
                    optional_missing,
                    decode_failed,
                );
            }
        }
        _ => {}
    }
    for (mono, text) in item_texts(&node.item) {
        let missing: Vec<char> = text.chars().filter(|ch| !ch.is_whitespace() && !fonts.covers(*ch, mono)).collect();
        if !missing.is_empty() {
            let listed: String = missing.iter().take(12).collect();
            failures.push(format!("{where_}: no glyph for `{listed}` in the loaded fonts"));
        }
    }
}

/// Every string the item draws, tagged with whether it uses the code face.
fn item_texts(item: &Item) -> Vec<(bool, String)> {
    let mut out = Vec::new();
    match item {
        Item::Text { content, sub, .. } => {
            out.push((false, content.clone()));
            if let Some(sub) = sub {
                out.push((false, sub.clone()));
            }
        }
        Item::Code { lines, .. } => out.extend(lines.iter().map(|line| (true, line.clone()))),
        Item::Counter { prefix, suffix, .. } => {
            for part in [prefix, suffix].into_iter().flatten() {
                out.push((false, part.clone()));
            }
        }
        Item::Card { title, body, .. } => {
            out.push((false, title.clone()));
            if let Some(body) = body {
                out.push((false, body.clone()));
            }
        }
        Item::Footnote { text, .. } => out.push((false, text.clone())),
        Item::Layer { items, .. } => {
            for child in items {
                out.extend(item_texts(&child.item));
            }
        }
        _ => {}
    }
    out
}

/// Reusable off-screen buffers for one render.
///
/// The world buffer (camera) and each component instance need their own frame at
/// the same time, so buffers are taken by value and returned; nothing borrows
/// the pool while painting into a buffer.
struct FramePool {
    width: u32,
    height: u32,
    free: Vec<Frame>,
}

impl FramePool {
    fn new(width: u32, height: u32) -> Self {
        Self { width, height, free: Vec::new() }
    }

    fn take(&mut self) -> Frame {
        self.free.pop().unwrap_or_else(|| Frame::new(self.width, self.height))
    }

    fn put(&mut self, frame: Frame) {
        self.free.push(frame);
    }
}

/// How many component instances may nest before the graph is treated as a cycle.
const MAX_INSTANCE_DEPTH: usize = 8;

/// Evaluates one node's motion and paints its item.
#[allow(clippy::too_many_arguments)]
fn paint_node(
    project: &Project,
    fonts: &mut Fonts,
    media: &mut MediaLibrary,
    frame: &mut Frame,
    node: &Node,
    local: u32,
    frame_index: u32,
    seconds: f32,
    zoom: f32,
    alpha_mul: f32,
    limits: DecodeLimits,
    pool: &mut FramePool,
    depth: usize,
) -> Result<(), Error> {
    let motion = node.anim.sample(local as f32);
    paint_item(project, fonts, media, frame, &node.item, local, frame_index, seconds, zoom, alpha_mul, limits, &motion, pool, depth)
}

#[allow(clippy::too_many_arguments)]
fn paint_frame(
    project: &Project,
    fonts: &mut Fonts,
    media: &mut MediaLibrary,
    vignette: Option<&Vignette>,
    pool: &mut FramePool,
    frame_index: u32,
    out: &mut Frame,
    limits: DecodeLimits,
) -> Result<(), Error> {
    out.fill(Rgba([0, 0, 0, 255]));
    // A global curve gives one continuous move for the whole film; a scene
    // camera is a per-block move used when no global curve is declared.
    let global = !project.camera.is_empty();
    let pose = camera_at(&project.camera, frame_index);
    let (global_zoom, global_focus, global_rotate) = (pose.zoom, pose.focus, pose.rotate);
    let scene = project.scenes.iter().find(|s| frame_index >= s.start && frame_index < s.start + s.duration);
    if let Some(scene) = scene {
        let seconds = project.time_base().to_seconds(frame_index) as f32;
        let local = frame_index - scene.start;
        // With a camera the scene renders off-screen, then that buffer is
        // pushed, pulled or panned into the output frame like a real lens.
        let camera = scene.camera.as_ref();
        let offscreen = global || camera.is_some();
        let mut scratch = if offscreen { Some(pool.take()) } else { None };
        let target: &mut Frame = match scratch.as_mut() {
            Some(buffer) => {
                buffer.fill(Rgba([0, 0, 0, 255]));
                buffer
            }
            None => out,
        };
        // The off-screen buffer may be larger than the output so a pushed-in
        // camera still has real pixels to show instead of an upscale.
        paint_background(target, &scene.background, seconds);
        for node in &scene.items {
            paint_node(project, fonts, media, target, node, local, frame_index, seconds, global_zoom, 1.0, limits, pool, 0)?;
        }
        if global {
            // `zoom` is normalized to the whole world buffer, so the buffer's
            // own scale cancels out: a 1.62 push samples exactly as many source
            // pixels as the output needs.
            let buffer = scratch.as_ref().expect("global camera renders off-screen");
            out.resample_from(buffer, global_zoom, global_focus, global_rotate);
        } else if let Some(camera) = camera {
            let t = if scene.duration > 1 { local as f32 / (scene.duration - 1) as f32 } else { 1.0 };
            let eased = apply(camera.ease, t);
            let zoom = camera.zoom_from + (camera.zoom_to - camera.zoom_from) * eased;
            let focus = [
                camera.focus_from[0] + (camera.focus_to[0] - camera.focus_from[0]) * eased,
                camera.focus_from[1] + (camera.focus_to[1] - camera.focus_from[1]) * eased,
            ];
            let buffer = scratch.as_ref().expect("scene camera renders off-screen");
            out.resample_from(buffer, zoom, focus, camera.rotate_to * eased);
        }
        if let Some(buffer) = scratch.take() {
            pool.put(buffer);
        }
    }
    // Corner darkening is an authored look, not a house style: no declaration,
    // no darkening.
    if let Some(vignette) = vignette {
        vignette.apply(out);
    }
    Ok(())
}

fn paint_background(frame: &mut Frame, bg: &Background, seconds: f32) {
    match bg {
        Background::Solid { color } => frame.fill(parse_or(Some(color), Rgba([8, 10, 20, 255]))),
        Background::VGradient { colors } => {
            let stops: Vec<Rgba> = colors.iter().filter_map(|c| parse(c)).collect();
            if stops.is_empty() {
                frame.fill(Rgba([8, 10, 20, 255]));
            } else {
                frame.v_gradient(&stops);
            }
        }
        Background::Radial { center, edge } => {
            frame.radial(parse_or(Some(center), Rgba([12, 18, 38, 255])), parse_or(Some(edge), Rgba([4, 6, 12, 255])));
        }
        Background::Starfield { base, stars, tint, drift, twinkle } => {
            let stops: Vec<Rgba> = base.iter().filter_map(|c| parse(c)).collect();
            if stops.is_empty() {
                frame.fill(Rgba([5, 7, 15, 255]));
            } else {
                frame.v_gradient(&stops);
            }
            let tint = parse_or(tint.as_deref(), Rgba([180, 205, 255, 255]));
            frame.starfield(*stars, tint, *drift, *twinkle, seconds);
        }
    }
}

fn px(x: f32, w: u32) -> f32 {
    x * w as f32
}

#[allow(clippy::too_many_arguments)]
fn paint_item(
    project: &Project,
    fonts: &mut Fonts,
    media: &mut MediaLibrary,
    frame: &mut Frame,
    item: &Item,
    local: u32,
    frame_index: u32,
    _seconds: f32,
    zoom: f32,
    alpha_mul: f32,
    limits: DecodeLimits,
    motion: &Motion,
    pool: &mut FramePool,
    depth: usize,
) -> Result<(), Error> {
    // Layout lives in output-relative units. When the world is rasterized
    // supersampled for a camera push, every pixel measure scales with it.
    let (fw, fh) = (frame.width as f32, frame.height as f32);
    let s = fw / project.width as f32;
    // Motion is evaluated once per node and then applied uniformly: position,
    // an opacity multiplier and progress. A property the item cannot express is
    // rejected during validation, never silently dropped here.
    let alpha_mul = alpha_mul * motion.opacity;
    // A follow path owns the position; an explicit track is the fallback.
    let mx = motion.follow.map(|point| point[0]).or(motion.x);
    let my = motion.follow.map(|point| point[1]).or(motion.y);
    let mprogress = motion.progress;
    // Position overrides apply to every item kind through one pair of helpers,
    // so no arm can forget them or invent its own precedence.
    let x_of = |x: f32| mx.unwrap_or(x);
    let y_of = |y: f32| my.unwrap_or(y);
    let progress_of = |progress: f32| mprogress.unwrap_or(progress);
    match item {
        Item::Text {
            at,
            dur,
            content,
            sub,
            size,
            bold,
            color,
            sub_color,
            sub_size,
            x,
            y,
            align,
            fade_in,
            fade_out,
            drift_y,
            max_width,
            line_height,
            typing,
            glow,
            stagger,
            rise,
            clip,
        } => {
            if local < *at || local >= *at + *dur {
                return Ok(());
            }
            let life = local - *at;
            let alpha = fade(life, *dur, *fade_in, *fade_out) * alpha_mul;
            if alpha <= 0.001 {
                return Ok(());
            }
            let drift = *drift_y * s * (1.0 - alpha);
            let size = *size * s;
            let mut visible = content.as_str();
            if let Some(speed) = typing {
                let count = ((life as f32 * speed).floor() as usize).min(content.chars().count());
                visible = &content[..content.char_indices().nth(count).map(|(i, _)| i).unwrap_or(content.len())];
            }
            let color = parse_or(Some(color), Rgba([230, 238, 250, 255])).with_alpha(alpha);
            let limit = max_width.map(|m| m * fw);
            let lh = line_height.unwrap_or(1.25);
            let top = px(y_of(*y), fh as u32) - drift;
            // A reveal rectangle lets text grow out of a shape.
            if let Some(rect) = clip {
                frame.set_clip([rect[0] * fw, rect[1] * fh, (rect[0] + rect[2]) * fw, (rect[1] + rect[3]) * fh]);
            }
            let drawn = if *rise > 0.0 {
                // Character-by-character entrance reads as motion, not as a
                // block of text fading in.
                fonts.draw_staggered(
                    frame,
                    visible,
                    px(x_of(*x), fw as u32),
                    top,
                    size,
                    *bold,
                    color,
                    *align,
                    *stagger,
                    *rise * s,
                    life,
                )
            } else {
                fonts.draw_block(frame, visible, px(x_of(*x), fw as u32), top, size, *bold, color, *align, limit, lh)
            };
            if let Some(glow_color) = glow {
                let gc = parse(glow_color);
                if let Some(gc) = gc {
                    frame.glow(
                        px(x_of(*x), fw as u32),
                        px(y_of(*y), fh as u32) - drift + drawn / 2.0,
                        size * 2.2,
                        gc.with_alpha(alpha * 0.10),
                    );
                }
            }
            if let Some(sub_text) = sub {
                let sub_color = parse_or(sub_color.as_deref(), Rgba([140, 160, 195, 255])).with_alpha(alpha * 0.9);
                let sub_size = sub_size.map(|v| v * s).unwrap_or(size * 0.42);
                fonts.draw_block(
                    frame,
                    sub_text,
                    px(x_of(*x), fw as u32),
                    px(y_of(*y), fh as u32) - drift + drawn + sub_size * 0.5,
                    sub_size,
                    false,
                    sub_color,
                    *align,
                    max_width.map(|m| m * fw),
                    1.2,
                );
            }
            if clip.is_some() {
                frame.clear_clip();
            }
        }
        Item::Code { at, dur, lines, x, y, size, theme, fade_in, fade_out, typing, line_height } => {
            if local < *at || local >= *at + *dur {
                return Ok(());
            }
            let life = local - *at;
            let alpha = fade(life, *dur, *fade_in, *fade_out) * alpha_mul;
            if alpha <= 0.001 {
                return Ok(());
            }
            let lh = line_height.unwrap_or(1.5);
            let size = *size * s;
            let panel_pad = size * 0.9;
            let total_chars: usize = lines.iter().map(|l| l.chars().count()).sum();
            let visible_total = match typing {
                Some(speed) => (life as f32 * speed).floor() as usize,
                None => total_chars,
            };
            // The panel grows with the typewriter so it never looks pre-filled.
            let mut revealed = 0usize;
            let mut budget = visible_total;
            for line in lines {
                if budget == 0 {
                    break;
                }
                budget = budget.saturating_sub(line.chars().count());
                revealed += 1;
            }
            let revealed = revealed.max(1);
            let block_h = revealed as f32 * size * lh + panel_pad * 2.0;
            let widest = lines.iter().map(|l| fonts.measure(l, size, false, true)).fold(0.0f32, f32::max);
            let block_w = widest + panel_pad * 2.0;
            let ox = px(x_of(*x), fw as u32);
            let oy = px(y_of(*y), fh as u32);
            if let Some(pc) = theme.panel.as_deref().and_then(parse) {
                frame.round_rect(ox, oy, block_w, block_h, size * 0.6, pc.with_alpha(alpha));
            }
            let total_chars: usize = lines.iter().map(|l| l.chars().count()).sum();
            let visible_total = match typing {
                Some(speed) => (life as f32 * speed).floor() as usize,
                None => total_chars,
            };
            let mut used = 0usize;
            for (i, line) in lines.iter().enumerate() {
                if used >= visible_total && typing.is_some() {
                    break;
                }
                let baseline = oy + panel_pad + size * 0.82 + i as f32 * size * lh;
                let remain = visible_total.saturating_sub(used);
                let visible = if remain >= line.chars().count() { None } else { Some(remain) };
                let mut themed = theme.clone();
                for field in [
                    &mut themed.plain,
                    &mut themed.keyword,
                    &mut themed.string,
                    &mut themed.comment,
                    &mut themed.number,
                    &mut themed.function,
                ] {
                    if let Some(mut c) = parse(field) {
                        c.0[3] = (c.0[3] as f32 * alpha) as u8;
                        *field = format!("#{:02x}{:02x}{:02x}{:02x}", c.0[0], c.0[1], c.0[2], c.0[3]);
                    }
                }
                fonts.draw_code_line(frame, line, ox + panel_pad, baseline, size, &themed, visible);
                used += line.chars().count();
            }
        }
        Item::Counter { at, dur, from, to, x, y, size, color, prefix, suffix, decimals, ease, fade_in, fade_out } => {
            if local < *at || local >= *at + *dur {
                return Ok(());
            }
            let life = local - *at;
            let alpha = fade(life, *dur, *fade_in, *fade_out) * alpha_mul;
            if alpha <= 0.001 {
                return Ok(());
            }
            let span = (*dur - (*dur).min(*fade_in + *fade_out)).max(1);
            let t = apply(*ease, life as f32 / span as f32);
            let value = from + (to - from) * t as f64;
            let mut text = media_text::format_number(value, *decimals);
            if let Some(p) = prefix {
                text = format!("{p}{text}");
            }
            if let Some(s) = suffix {
                text = format!("{text}{s}");
            }
            let color = parse_or(Some(color), Rgba([240, 246, 255, 255])).with_alpha(alpha);
            fonts.draw_block(
                frame,
                &text,
                px(x_of(*x), fw as u32),
                px(y_of(*y), fh as u32),
                *size * s,
                true,
                color,
                Align::Center,
                None,
                1.2,
            );
        }
        Item::Card { at, dur, x, y, w, h, title, body, accent, panel, fade_in, fade_out, drift_y } => {
            if local < *at || local >= *at + *dur {
                return Ok(());
            }
            let life = local - *at;
            let alpha = fade(life, *dur, *fade_in, *fade_out) * alpha_mul;
            if alpha <= 0.001 {
                return Ok(());
            }
            let drift = drift_y * s * (1.0 - alpha);
            let cx = px(x_of(*x), fw as u32) - px(*w, fw as u32) / 2.0;
            let cy = px(y_of(*y), fh as u32) - drift - px(*h, fh as u32) / 2.0;
            let cw = px(*w, fw as u32);
            let ch = px(*h, fh as u32);
            let panel_color = parse_or(panel.as_deref(), Rgba([16, 24, 44, 255])).with_alpha(alpha * 0.85);
            frame.round_rect(cx, cy, cw, ch, 18.0 * s, panel_color);
            let accent_color = parse_or(accent.as_deref(), Rgba([110, 160, 255, 255])).with_alpha(alpha);
            frame.round_rect(cx, cy, 6.0 * s, ch, 3.0 * s, accent_color);
            let title_size = ch * 0.22;
            fonts.draw_block(
                frame,
                title,
                cx + 28.0 * s,
                cy + ch * 0.16,
                title_size,
                true,
                Rgba([235, 242, 252, (alpha * 255.0) as u8]),
                Align::Left,
                Some(cw - 56.0 * s),
                1.15,
            );
            if let Some(body) = body {
                let body_size = ch * 0.14;
                fonts.draw_block(
                    frame,
                    body,
                    cx + 28.0 * s,
                    cy + ch * 0.44,
                    body_size,
                    false,
                    Rgba([150, 170, 200, (alpha * 255.0) as u8]),
                    Align::Left,
                    Some(cw - 56.0 * s),
                    1.3,
                );
            }
        }
        Item::Image { at, dur, src, x, y, w, h, fit, opacity, radius, fade_in, fade_out, required } => {
            if local < *at || local >= *at + *dur {
                return Ok(());
            }
            let life = local - *at;
            let alpha = fade(life, *dur, *fade_in, *fade_out) * opacity * alpha_mul;
            if alpha <= 0.001 {
                return Ok(());
            }
            // A required import that fails here aborts the render: preflight
            // already proved it decodes, so this is a real mid-run failure.
            let bitmap = match media.image(src, limits) {
                Ok(bitmap) => bitmap,
                Err(error) if *required => {
                    return Err(Error::media(format!("required image `{src}` cannot be decoded: {error}")));
                }
                Err(_) => return Ok(()),
            };
            let (bw, bh) = (px(*w, fw as u32), px(*h, fh as u32));
            frame.draw_bitmap(
                &bitmap,
                px(x_of(*x), fw as u32) - bw / 2.0,
                px(y_of(*y), fh as u32) - bh / 2.0,
                bw,
                bh,
                *fit,
                alpha,
                *radius * s,
            );
        }
        Item::Video { at, dur, src, src_in, x, y, w, h, fit, opacity, radius, fade_in, fade_out, required } => {
            if local < *at || local >= *at + *dur {
                return Ok(());
            }
            let life = local - *at;
            let alpha = fade(life, *dur, *fade_in, *fade_out) * opacity * alpha_mul;
            if alpha <= 0.001 {
                return Ok(());
            }
            let source_seconds = project.time_base().to_seconds(*src_in + life);
            // Same contract as images: required clips fail loudly, optional
            // ones are documented silence.
            let bitmap = match media.video_frame(src, source_seconds, limits) {
                Ok(bitmap) => bitmap,
                Err(error) if *required => {
                    return Err(Error::media(format!(
                        "required video `{src}` cannot be decoded at {source_seconds:.3}s: {error}"
                    )));
                }
                Err(_) => return Ok(()),
            };
            let (bw, bh) = (px(*w, fw as u32), px(*h, fh as u32));
            frame.draw_bitmap(
                &bitmap,
                px(x_of(*x), fw as u32) - bw / 2.0,
                px(y_of(*y), fh as u32) - bh / 2.0,
                bw,
                bh,
                *fit,
                alpha,
                *radius * s,
            );
        }
        Item::Shape { at, dur, shape, x, y, w, h, color, radius, stroke, progress, blend, fade_in, fade_out } => {
            if local < *at || local >= *at + *dur {
                return Ok(());
            }
            let life = local - *at;
            let alpha = fade(life, *dur, *fade_in, *fade_out) * alpha_mul;
            if alpha <= 0.001 {
                return Ok(());
            }
            let c = parse_or(Some(color), Rgba([110, 160, 255, 255])).with_alpha(alpha);
            let cx = px(x_of(*x), fw as u32);
            let cy = px(y_of(*y), fh as u32);
            match shape {
                ShapeKind::Rect => frame.round_rect(cx, cy, px(*w, fw as u32), px(*h, fh as u32), *radius * s, c),
                ShapeKind::Circle => frame.circle(cx, cy, px(*w, fw as u32) / 2.0, c, None),
                ShapeKind::Ring => frame.circle(cx, cy, px(*w, fw as u32) / 2.0, c, stroke.map(|v| v * s).or(Some(3.0 * s))),
                ShapeKind::Line => {
                    // `progress` lets a line draw itself, which is how motion
                    // reads as intent rather than as a slideshow.
                    let p = progress_of(*progress).clamp(0.0, 1.0);
                    let ex = cx + px(*w, fw as u32) * p;
                    let ey = cy + px(*h, fh as u32) * p;
                    let thickness = stroke.unwrap_or(2.0) * s;
                    match blend {
                        Blend::Normal => frame.line(cx, cy, ex, ey, thickness, c),
                        Blend::Add => {
                            let steps = ((ex - cx).abs().max((ey - cy).abs()) as u32).max(1);
                            for step in 0..=steps {
                                let t = step as f32 / steps as f32;
                                let point = (cx + (ex - cx) * t, cy + (ey - cy) * t);
                                frame.add_pixel(point.0 as u32, point.1 as u32, c);
                                frame.add_pixel(point.0 as u32 + 1, point.1 as u32, c);
                            }
                        }
                    }
                    if p > 0.01 && p < 0.999 {
                        frame.glow(ex, ey, thickness * 5.0, c);
                    }
                }
            }
        }
        Item::Bands { at, dur, count, angle, speed, color, alpha, thickness, fade_in, fade_out } => {
            if local < *at || local >= *at + *dur {
                return Ok(());
            }
            let life = local - *at;
            let opacity = fade(life, *dur, *fade_in, *fade_out) * alpha * alpha_mul;
            if opacity <= 0.001 {
                return Ok(());
            }
            let c = parse_or(Some(color), Rgba([120, 170, 255, 255])).with_alpha(opacity);
            let height = fh;
            let seconds = project.time_base().to_seconds(life) as f32;
            let count = (*count).max(1);
            for i in 0..count {
                let base = height * (i as f32 + 1.0) / (count as f32 + 1.0);
                let drift = seconds * speed * if i % 2 == 0 { 1.0 } else { -0.6 };
                frame.band(base + drift, *angle, *thickness * s, c);
            }
        }
        Item::Grain { at, dur, strength, density, fade_in, fade_out } => {
            if local < *at || local >= *at + *dur {
                return Ok(());
            }
            let life = local - *at;
            let opacity = fade(life, *dur, *fade_in, *fade_out) * alpha_mul;
            if opacity <= 0.001 {
                return Ok(());
            }
            frame.grain((*density as f32 * s * s) as u32, strength * opacity, frame_index.wrapping_add(1));
        }
        Item::Path {
            at,
            dur,
            points,
            smooth,
            closed,
            fill,
            color,
            stroke,
            progress,
            rotate,
            scale,
            blend,
            fade_in,
            fade_out,
        } => {
            if local < *at || local >= *at + *dur {
                return Ok(());
            }
            let life = local - *at;
            let alpha = fade(life, *dur, *fade_in, *fade_out) * alpha_mul;
            if alpha <= 0.001 {
                return Ok(());
            }
            let to_pixels = |list: &[[f32; 2]]| -> Vec<[f32; 2]> { list.iter().map(|p| [p[0] * fw, p[1] * fh]).collect() };
            let mut shape = if *smooth { smooth_polyline(points, *closed, 12) } else { points.clone() };
            if *closed && !*smooth && !shape.is_empty() {
                shape.push(shape[0]);
            }
            // Per-item transform: scale, then rotate, around the centroid.
            if (*scale - 1.0).abs() > 0.001 || rotate.abs() > 0.001 {
                let mut px_points = to_pixels(&shape);
                let count = px_points.len().max(1) as f32;
                let cx = px_points.iter().map(|p| p[0]).sum::<f32>() / count;
                let cy = px_points.iter().map(|p| p[1]).sum::<f32>() / count;
                let (sin, cos) = rotate.to_radians().sin_cos();
                for point in &mut px_points {
                    let dx = (point[0] - cx) * scale;
                    let dy = (point[1] - cy) * scale;
                    point[0] = cx + dx * cos - dy * sin;
                    point[1] = cy + dx * sin + dy * cos;
                }
                shape = px_points.iter().map(|p| [p[0] / fw, p[1] / fh]).collect();
            }
            let pixels = to_pixels(&shape);
            let additive = *blend == Blend::Add;
            if let Some(fill_color) = fill {
                let c = parse_or(Some(fill_color), Rgba([20, 40, 90, 255])).with_alpha(alpha);
                frame.fill_polygon(&pixels, c, additive);
            }
            if let Some(width) = stroke {
                let c = parse_or(Some(color), Rgba([110, 160, 255, 255])).with_alpha(alpha);
                frame.stroke_polyline(&pixels, width * s, c, progress_of(*progress).clamp(0.0, 1.0), additive);
                // A moving tip reads as a pen rather than a static outline.
                let drawn = progress_of(*progress).clamp(0.0, 1.0);
                if drawn > 0.01 && drawn < 0.999 {
                    if let Some(tip) = pixels.get((pixels.len() as f32 * drawn) as usize) {
                        frame.glow(tip[0], tip[1], width * 4.0 * s, c);
                    }
                }
            }
        }
        Item::Glow { at, dur, x, y, radius, color, intensity, fade_in, fade_out } => {
            if local < *at || local >= *at + *dur {
                return Ok(());
            }
            let life = local - *at;
            let alpha = fade(life, *dur, *fade_in, *fade_out) * intensity * alpha_mul;
            if alpha <= 0.001 {
                return Ok(());
            }
            let c = parse_or(Some(color), Rgba([242, 201, 138, 255])).with_alpha(alpha);
            frame.glow(px(x_of(*x), fw as u32), px(y_of(*y), fh as u32), px(*radius, fw as u32) * s, c);
        }
        Item::Layer { at, dur, zoom_from, zoom_to, edge, items } => {
            if local < *at || local >= *at + *dur {
                return Ok(());
            }
            // Layers hand off by zoom instead of by cut: while the camera keeps
            // moving, the layer that no longer owns this scale fades out and the
            // next one fades in. That is what makes a continuous take possible.
            let weight = layer_weight(zoom, *zoom_from, *zoom_to, *edge);
            if weight <= 0.001 {
                return Ok(());
            }
            let local_life = local - *at;
            for child in items {
                paint_node(
                    project,
                    fonts,
                    media,
                    frame,
                    child,
                    local_life,
                    frame_index,
                    _seconds,
                    zoom,
                    alpha_mul * weight,
                    limits,
                    pool,
                    depth,
                )?;
            }
        }
        Item::Instance { at, dur, component, x, y, w, h, fit, opacity, fade_in, fade_out, required } => {
            if local < *at || local >= *at + *dur {
                return Ok(());
            }
            let life = local - *at;
            let alpha = fade(life, *dur, *fade_in, *fade_out) * opacity * alpha_mul;
            if alpha <= 0.001 {
                return Ok(());
            }
            if depth >= MAX_INSTANCE_DEPTH {
                return Err(Error::invalid(format!(
                    "component `{component}` nests deeper than {MAX_INSTANCE_DEPTH} levels; check for a cycle"
                )));
            }
            let Some(definition) = find_component(project, component) else {
                if *required {
                    return Err(Error::invalid(format!("instance references unknown component `{component}`")));
                }
                return Ok(());
            };
            // The component is painted on its own transparent buffer and then
            // placed into the box like an image, so an instance stays a reuse
            // boundary instead of a copy of the definition's pixels.
            let mut buffer = pool.take();
            buffer.fill(Rgba([0, 0, 0, 0]));
            for child in &definition.items {
                paint_node(project, fonts, media, &mut buffer, child, life, frame_index, _seconds, zoom, 1.0, limits, pool, depth + 1)?;
            }
            let bitmap = Bitmap::new(buffer.width, buffer.height, buffer.pixels.clone());
            pool.put(buffer);
            let (bw, bh) = (px(*w, fw as u32), px(*h, fh as u32));
            frame.draw_bitmap(
                &bitmap,
                px(x_of(*x), fw as u32) - bw / 2.0,
                px(y_of(*y), fh as u32) - bh / 2.0,
                bw,
                bh,
                *fit,
                alpha,
                0.0,
            );
        }
        Item::Footnote { at, dur, text, size, x, y, align, color, fade_in, fade_out } => {
            if local < *at || local >= *at + *dur {
                return Ok(());
            }
            let life = local - *at;
            let alpha = fade(life, *dur, *fade_in, *fade_out) * 0.9;
            if alpha <= 0.001 {
                return Ok(());
            }
            let c = parse_or(Some(color), Rgba([93, 111, 140, 255])).with_alpha(alpha);
            fonts.draw_block(
                frame,
                text,
                px(x_of(*x), fw as u32),
                px(y_of(*y), fh as u32),
                *size * s,
                false,
                c,
                *align,
                Some(0.6 * fw),
                1.2,
            );
        }
        Item::Progress { at, dur, track, fill, y } => {
            if local < *at || local >= *at + *dur {
                return Ok(());
            }
            let life = local - *at;
            let alpha = fade(life, *dur, 6, 6) * alpha_mul;
            if alpha <= 0.001 {
                return Ok(());
            }
            let track = parse_or(track.as_deref(), Rgba([40, 52, 78, 255])).with_alpha(alpha * 0.7);
            let fill = parse_or(fill.as_deref(), Rgba([110, 160, 255, 255])).with_alpha(alpha);
            let bar_y = px(y.unwrap_or(0.965), fh as u32);
            let margin = 0.08 * fw;
            let full = fw - margin * 2.0;
            frame.round_rect(margin, bar_y, full, 4.0 * s, 2.0 * s, track);
            let t = frame_index as f32 / project.duration_frames as f32;
            frame.round_rect(margin, bar_y, full * t, 4.0 * s, 2.0 * s, fill);
        }
    }
    Ok(())
}

/// Opacity of a layer at a given camera zoom: 1 inside the window, easing to 0
/// across `edge` outside it. Windows that overlap cross-fade.
fn layer_weight(zoom: f32, from: f32, to: f32, edge: f32) -> f32 {
    let edge = edge.max(0.0001);
    let fade_in = ((zoom - (from - edge)) / edge).clamp(0.0, 1.0);
    let fade_out = (((to + edge) - zoom) / edge).clamp(0.0, 1.0);
    // Smoothstep on both ends avoids a visible start or stop.
    let ease = |t: f32| t * t * (3.0 - 2.0 * t);
    ease(fade_in) * ease(fade_out)
}

/// Deterministic jitter helper kept for future item kinds.
#[allow(dead_code)]
fn jitter(seed: u32) -> f32 {
    (hash(seed) as f32 / u32::MAX as f32) * 2.0 - 1.0
}

#[cfg(test)]
mod tests {
    /// Encode, import, and round-trip paths sit on Windows Media Foundation.
    fn windows_media_foundation() -> bool {
        if cfg!(windows) {
            true
        } else {
            eprintln!("SKIP: OpenCut encode/import requires Windows Media Foundation");
            false
        }
    }

    /// Renders a tiny film end to end: rasterize, synthesize audio, encode MP4.
    #[test]
    fn renders_minimal_film() {
        if !windows_media_foundation() {
            return;
        }
        let project = serde_json::json!({
            "width": 320,
            "height": 240,
            "fps": 15,
            "duration_frames": 30,
            "scenes": [{
                "start": 0,
                "duration": 30,
                "background": { "kind": "v-gradient", "colors": ["#05070f", "#0b1226"] },
                "items": [
                    { "kind": "text", "at": 0, "dur": 30, "content": "OpenCut", "size": 48.0,
                      "bold": true, "color": "#e6eefa", "x": 0.5, "y": 0.4, "align": "center",
                      "fade_in": 5, "fade_out": 5 },
                    { "kind": "counter", "at": 0, "dur": 30, "from": 0.0, "to": 1234.0, "x": 0.5,
                      "y": 0.7, "size": 36.0, "color": "#7aa2ff", "decimals": 0, "ease": "out-cubic",
                      "fade_in": 5, "fade_out": 5 },
                    { "kind": "progress", "at": 0, "dur": 30 }
                ]
            }],
            "audio": {
                "bgm": { "volume": 0.2, "mood": "pad" },
                "sfx": [{ "at": 5, "kind": "whoosh", "volume": 0.4 }]
            }
        });
        let out = std::env::temp_dir().join("opencut-smoke.mp4");
        let stats = super::render_project(&project.to_string(), &out, &mut |_, _| {}).expect("render");
        assert_eq!(stats.frames, 30);
        assert!(stats.audio);
        let meta = std::fs::metadata(&out).expect("output exists");
        assert!(meta.len() > 1_000, "mp4 too small: {}", meta.len());
    }

    /// A film-wide camera curve with zoom-gated layers and a soft light: the
    /// one-take path must compile, render and move the frame without cutting.
    #[test]
    fn renders_one_take_camera() {
        if !windows_media_foundation() {
            return;
        }
        let project = serde_json::json!({
            "width": 320, "height": 240, "fps": 15, "duration_frames": 30,
            "scenes": [{
                "start": 0, "duration": 30,
                "background": { "kind": "solid", "color": "#05070f" },
                "items": [{
                    "kind": "layer", "at": 0, "dur": 30, "zoom_from": 0.9, "zoom_to": 2.0,
                    "items": [
                        { "kind": "glow", "at": 0, "dur": 30, "x": 0.5, "y": 0.5,
                          "radius": 0.2, "color": "#ffcc88", "intensity": 1.0 },
                        { "kind": "path", "at": 0, "dur": 30,
                          "points": [[0.0, 0.8], [0.5, 0.6], [1.0, 0.8]],
                          "smooth": true, "stroke": 3.0, "color": "#7aa2ff" }
                    ]
                }]
            }],
            "camera": [
                { "at": 0, "zoom": 1.6, "focus": [0.5, 0.5] },
                { "at": 29, "zoom": 1.0, "focus": [0.5, 0.5], "ease": "in-out-quart" }
            ],
            "audio": null
        });
        let out = std::env::temp_dir().join("opencut-one-take.mp4");
        let stats = super::render_project(&project.to_string(), &out, &mut |_, _| {}).expect("render");
        assert_eq!(stats.frames, 30);
        assert!(std::fs::metadata(&out).unwrap().len() > 1_000);
    }

    #[test]
    fn renders_without_audio_track() {
        if !windows_media_foundation() {
            return;
        }
        let project = serde_json::json!({
            "width": 320, "height": 240, "fps": 15, "duration_frames": 15,
            "scenes": [{ "start": 0, "duration": 15,
                "background": { "kind": "solid", "color": "#101828" },
                "items": [] }],
            "audio": null
        });
        let out = std::env::temp_dir().join("opencut-video-only.mp4");
        let stats = super::render_project(&project.to_string(), &out, &mut |_, _| {}).expect("render");
        assert!(!stats.audio);
        assert!(std::fs::metadata(&out).unwrap().len() > 500);
    }

    /// The OpenCut sound-effect path end to end inside the audio domain: the
    /// scene wire JSON lowers through `lower_audio`, the bed is mixed from
    /// `media.synthesis` renders, and the placed effects are audible, finite
    /// and byte-identical across builds (the seed contract).
    #[test]
    fn sfx_bed_is_audible_deterministic_and_lowers_from_wire_json() {
        let json = serde_json::json!({
            "width": 64, "height": 64, "fps": 30, "duration_frames": 90,
            "scenes": [],
            "audio": { "sfx": [
                { "at": 10, "kind": "whoosh", "volume": 0.8 },
                { "at": 60, "kind": "impact", "volume": 0.6 }
            ] }
        })
        .to_string();
        let project: media_scene::Project = serde_json::from_str(&json).expect("openscut wire json");
        let plan = super::lower_audio(&project);
        assert_eq!(plan.sfx.len(), 2);
        let build = || {
            let mut media = media_codec::MediaLibrary::new();
            media_audio::build_audio(&plan, &mut media, media_codec::DecodeLimits::unbounded()).expect("bed")
        };
        let a = build();
        let b = build();
        assert_eq!(a, b, "the same plan must render the same bed");
        let sr = plan.sample_rate as usize;
        for (at, name) in [(10usize, "whoosh"), (60, "impact")] {
            let start = at * sr / 30;
            let end = (start + sr / 4).min(a.len() / 2);
            let peak = a[start * 2..end * 2].iter().fold(0i16, |m, s| m.max(s.abs()));
            assert!(peak > 200, "{name} region silent: peak {peak}");
        }
        let bad = serde_json::json!({
            "width": 64, "height": 64, "fps": 30, "duration_frames": 10,
            "scenes": [], "audio": { "sfx": [{ "at": 0, "kind": "airhorn" }] }
        })
        .to_string();
        let project = serde_json::from_str::<media_scene::Project>(&bad).expect("wire json parses");
        let plan = super::lower_audio(&project);
        let mut media = media_codec::MediaLibrary::new();
        let error = media_audio::build_audio(&plan, &mut media, media_codec::DecodeLimits::unbounded())
            .expect_err("unknown kind must fail explicitly");
        assert_eq!(error.code(), "invalid_input");
    }

    /// The OpenCut voice path end to end: a song written as scene wire JSON
    /// lowers to a voice placement and is synthesized into the film bed at the
    /// declared frame, without touching any external audio asset.
    #[test]
    fn voice_bed_is_synthesized_from_wire_json() {
        let json = serde_json::json!({
            "width": 64, "height": 64, "fps": 30, "duration_frames": 90,
            "scenes": [],
            "audio": {
                "sfx": [{ "at": 0, "preset": "impact-snap", "volume": 0.5 }],
                "voice": [{
                    "at": 15,
                    "volume": 0.9,
                    "song": {
                        "tempo_bpm": 120,
                        "vibrato": { "rate_hz": 5.0, "depth": 0.02, "delay_s": 0.1 },
                        "breath": 0.1,
                        "seed": 3,
                        "syllables": [
                            { "onset": "r", "vowel": "a", "pitch": 392.0, "beats": 1.0 },
                            { "vowel": "o", "pitch": 523.25, "beats": 1.0 }
                        ]
                    }
                }]
            }
        })
        .to_string();
        let project: media_scene::Project = serde_json::from_str(&json).expect("wire json");
        let plan = super::lower_audio(&project);
        assert_eq!(plan.voice.len(), 1, "the song must lower to a voice placement");
        assert_eq!(plan.voice[0].at_frames, 15);
        assert!((plan.voice[0].song.tempo_bpm - 120.0).abs() < 1e-6, "tempoBpm must survive the wire");
        let mut media = media_codec::MediaLibrary::new();
        let bed = media_audio::build_audio(&plan, &mut media, media_codec::DecodeLimits::unbounded()).expect("bed");
        let start = 15 * plan.sample_rate as usize / 30;
        let end = (start + plan.sample_rate as usize / 2).min(bed.len() / 2);
        let peak = bed[start * 2..end * 2].iter().fold(0i16, |m, s| m.max(s.abs()));
        assert!(peak > 200, "voice region silent: peak {peak}");
    }

    /// Drums and instruments are authored in the film audio bed: a groove and
    /// a bass note lower to the mixer and are audible at the declared frames.
    #[test]
    fn instrument_notes_and_grooves_lower_from_wire_json() {
        let json = serde_json::json!({
            "width": 64, "height": 64, "fps": 30, "duration_frames": 90,
            "scenes": [],
            "audio": {
                "patterns": [{ "at": 0, "preset": "four-on-floor", "bars": 1, "volume": 0.9 }],
                "notes": [
                    { "at": 15, "instrument": "bass", "note": 36, "duration": 0.5, "velocity": 0.8 },
                    { "at": 30, "instrument": "kick", "velocity": 0.9 }
                ]
            }
        })
        .to_string();
        let project: media_scene::Project = serde_json::from_str(&json).expect("wire json");
        let plan = super::lower_audio(&project);
        assert_eq!(plan.patterns.len(), 1);
        assert_eq!(plan.notes.len(), 2);
        assert_eq!(plan.patterns[0].at_frames, 0);
        let mut media = media_codec::MediaLibrary::new();
        let bed = media_audio::build_audio(&plan, &mut media, media_codec::DecodeLimits::unbounded()).expect("bed");
        let peak = bed.iter().fold(0i16, |a, s| a.max(s.abs()));
        assert!(peak > 500, "drums and bass must be audible: {peak}");
        // 未知乐器必须显式报错，不能静默丢弃。
        let bad = serde_json::json!({
            "width": 64, "height": 64, "fps": 30, "duration_frames": 10,
            "scenes": [], "audio": { "notes": [{ "at": 0, "instrument": "theremin" }] }
        })
        .to_string();
        let project = serde_json::from_str::<media_scene::Project>(&bad).expect("wire json");
        let plan = super::lower_audio(&project);
        let mut media = media_codec::MediaLibrary::new();
        let error = media_audio::build_audio(&plan, &mut media, media_codec::DecodeLimits::unbounded())
            .expect_err("unknown instrument must fail");
        assert_eq!(error.code(), "invalid_input");
    }

    /// Writes a PNG with our own encoder, imports it back through WIC and
    /// checks that the pixels survive the round trip.
    #[test]
    fn imports_image_roundtrip() {
        if !windows_media_foundation() {
            return;
        }
        let dir = std::env::temp_dir().join("opencut-media-tests");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("fixture.png");
        let (width, height) = (64u32, 48u32);
        let mut pixels = Vec::with_capacity((width * height * 4) as usize);
        for y in 0..height {
            for x in 0..width {
                pixels.extend_from_slice(&[(x * 4) as u8, (y * 5) as u8, 0x40, 0xFF]);
            }
        }
        media_codec::write_png(&path, width, height, &pixels).expect("write png");

        let mut media = media_codec::MediaLibrary::new();
        let bitmap = media.image(path.to_str().unwrap(), media_codec::DecodeLimits::unbounded()).expect("decode png");
        assert_eq!((bitmap.width, bitmap.height), (width, height));
        // Row 1, column 3 should come back as encoded (allow 1 bit of slack).
        let i = (width + 3) as usize * 4;
        assert!((i32::from(bitmap.rgba[i]) - 12).abs() <= 2, "red {}", bitmap.rgba[i]);
        assert!((i32::from(bitmap.rgba[i + 1]) - 5).abs() <= 2, "green {}", bitmap.rgba[i + 1]);
        assert!((i32::from(bitmap.rgba[i + 2]) - 0x40).abs() <= 2, "blue {}", bitmap.rgba[i + 2]);
    }

    /// Renders a clip with this pipeline, imports it as media, and mixes an
    /// audio file generated by our own WAV writer.
    #[test]
    fn imports_video_and_audio() {
        if !windows_media_foundation() {
            return;
        }
        let dir = std::env::temp_dir().join("opencut-media-tests");
        std::fs::create_dir_all(&dir).unwrap();

        // 1. Produce a source clip with OpenCut itself.
        let source = dir.join("source.mp4");
        let clip = serde_json::json!({
            "width": 160, "height": 120, "fps": 15, "duration_frames": 15,
            "scenes": [{ "start": 0, "duration": 15,
                "background": { "kind": "solid", "color": "#c02030" }, "items": [] }],
            "audio": null
        });
        super::render_project(&clip.to_string(), &source, &mut |_, _| {}).expect("source clip");

        // 1b. Decode the clip back and check the pixels, so a silent import
        // failure cannot pass as a green test.
        let mut library = media_codec::MediaLibrary::new();
        let frame = library
            .video_frame(source.to_str().unwrap(), 0.1, media_codec::DecodeLimits::unbounded())
            .expect("decode video frame");
        assert_eq!((frame.width, frame.height), (160, 120));
        // 取中心像素：暗角（post::Vignette）只压边缘角，中心保留源色，
        // 断言验证的是视频回环而不是暗角强度。
        let center = (80 * 160 + 60) * 4;
        assert!(
            frame.rgba[center] > 120 && frame.rgba[center + 1] < 100,
            "expected the red source, got {:?}",
            &frame.rgba[center..center + 4]
        );

        // 2. Write a short PCM16 WAV with our own writer.
        let wav = dir.join("tone.wav");
        let mut samples = Vec::new();
        for i in 0..media_types::SAMPLE_RATE as usize / 2 {
            let t = i as f64 / media_types::SAMPLE_RATE as f64;
            let value = ((t * 440.0 * std::f64::consts::TAU).sin() * 0.4 * i16::MAX as f64) as i16;
            samples.push(value);
            samples.push(value);
        }
        media_codec::write_wav(wav.to_str().unwrap(), &samples, media_types::SAMPLE_RATE).expect("wav");

        // 3. Import both in a film.
        let film = serde_json::json!({
            "width": 320, "height": 240, "fps": 15, "duration_frames": 30,
            "scenes": [{ "start": 0, "duration": 30,
                "background": { "kind": "solid", "color": "#05070f" },
                "items": [
                    { "kind": "video", "at": 0, "dur": 30, "src": source.to_str().unwrap(),
                      "x": 0.5, "y": 0.5, "w": 0.6, "h": 0.6, "fit": "contain" },
                    { "kind": "image", "at": 0, "dur": 30, "src": dir.join("fixture.png").to_str().unwrap(),
                      "x": 0.82, "y": 0.18, "w": 0.14, "h": 0.14, "fit": "cover", "radius": 12.0 }
                ]
            }],
            "audio": { "clips": [{ "src": wav.to_str().unwrap(), "at": 0, "volume": 0.8 }] }
        });
        let out = dir.join("imported.mp4");
        let stats = super::render_project(&film.to_string(), &out, &mut |_, _| {}).expect("render imports");
        assert_eq!(stats.frames, 30);
        assert!(stats.audio);
        assert!(std::fs::metadata(&out).unwrap().len() > 2_000);
    }

    /// A declared required image that cannot decode stops the render before any
    /// frame is encoded: a missing picture must never pass as a finished film.
    #[test]
    fn missing_required_image_fails_without_publishing() {
        let dir = std::env::temp_dir().join("opencut-media-tests");
        let absent = dir.join("definitely-absent-import.png");
        let out = dir.join("required-image.mp4");
        let _ = std::fs::remove_file(&out);
        let film = serde_json::json!({
            "width": 64, "height": 64, "fps": 15, "duration_frames": 15,
            "scenes": [{ "start": 0, "duration": 15,
                "background": { "kind": "solid", "color": "#05070f" },
                "items": [{ "kind": "image", "at": 0, "dur": 15, "src": absent.to_str().unwrap(),
                            "x": 0.5, "y": 0.5, "w": 0.5, "h": 0.5 }] }],
            "audio": null
        });
        let error = super::render_project(&film.to_string(), &out, &mut |_, _| {}).expect_err("missing image must fail");
        assert_eq!(error.code(), "media", "{error}");
        assert!(error.message.contains("resource preflight failed"), "{error}");
        assert!(!out.exists(), "a failed render must not publish an artifact");
    }

    /// The same absent picture declared optional is documented silence: the
    /// render succeeds and the stats name the slot that stayed empty.
    #[test]
    fn optional_image_stays_silent_and_is_reported() {
        if !windows_media_foundation() {
            return;
        }
        let dir = std::env::temp_dir().join("opencut-media-tests");
        std::fs::create_dir_all(&dir).unwrap();
        let absent = dir.join("optional-absent.png");
        let out = dir.join("optional-image.mp4");
        let film = serde_json::json!({
            "width": 64, "height": 64, "fps": 15, "duration_frames": 10,
            "scenes": [{ "start": 0, "duration": 10,
                "background": { "kind": "solid", "color": "#05070f" },
                "items": [{ "kind": "image", "at": 0, "dur": 10, "src": absent.to_str().unwrap(),
                            "x": 0.5, "y": 0.5, "w": 0.5, "h": 0.5, "required": false }] }],
            "audio": null
        });
        let stats = super::render_project(&film.to_string(), &out, &mut |_, _| {}).expect("optional slot renders");
        assert_eq!(stats.optional_missing.len(), 1);
        assert!(stats.optional_missing[0].ends_with("optional-absent.png"));
        assert!(out.exists());
    }

    /// Narration is a slot, not a guess: a path that does not decode aborts the
    /// mix instead of producing a film that quietly has no voice.
    #[test]
    fn undecodable_narration_fails() {
        let dir = std::env::temp_dir().join("opencut-media-tests");
        std::fs::create_dir_all(&dir).unwrap();
        let broken = dir.join("broken-narration.wav");
        std::fs::write(&broken, b"not a wave file").unwrap();
        let out = dir.join("narration.mp4");
        let film = serde_json::json!({
            "width": 64, "height": 64, "fps": 15, "duration_frames": 10,
            "scenes": [],
            "audio": { "narration": { "wav": broken.to_str().unwrap() } }
        });
        let error = super::render_project(&film.to_string(), &out, &mut |_, _| {}).expect_err("broken narration must fail");
        assert!(error.message.contains("narration"), "{error}");
        assert!(!out.exists());
    }

    /// Colors are a contract: an unparseable value is reported, never replaced
    /// by a default that repaints the film.
    #[test]
    fn invalid_color_is_reported_with_its_location() {
        let film = serde_json::json!({
            "width": 64, "height": 64, "fps": 15, "duration_frames": 10,
            "scenes": [{ "start": 0, "duration": 10,
                "background": { "kind": "solid", "color": "#05070f" },
                "items": [{ "kind": "text", "at": 0, "dur": 10, "content": "hi", "size": 20.0,
                            "color": "tomato", "x": 0.5, "y": 0.5 }] }],
            "audio": null
        });
        let out = std::env::temp_dir().join("opencut-media-tests").join("bad-color.mp4");
        let error = super::render_project(&film.to_string(), &out, &mut |_, _| {}).expect_err("bad color must fail");
        assert_eq!(error.code(), "invalid_scene", "{error}");
        assert!(error.message.contains("scene 0 item 0"), "{error}");
        assert!(error.message.contains("tomato"), "{error}");
    }

    /// Zero fades mean "no fade": the first frame already shows the text at full
    /// opacity instead of the old forced one-frame ramp.
    #[test]
    fn zero_fade_in_is_not_forced_to_one_frame() {
        if !windows_media_foundation() {
            return;
        }
        let render = |fade_in: u32| {
            let film = serde_json::json!({
                "width": 64, "height": 64, "fps": 15, "duration_frames": 6,
                "scenes": [{ "start": 0, "duration": 6,
                    "background": { "kind": "solid", "color": "#000000" },
                    "items": [{ "kind": "text", "at": 0, "dur": 6, "content": "OO", "size": 40.0,
                                "bold": true, "color": "#ffffff", "x": 0.5, "y": 0.4,
                                "fade_in": fade_in, "fade_out": 0 }] }],
                "audio": null
            });
            let out = std::env::temp_dir().join(format!("opencut-fade-{fade_in}.mp4"));
            super::render_project(&film.to_string(), &out, &mut |_, _| {}).expect("render");
            let mut library = media_codec::MediaLibrary::new();
            let frame = library.video_frame(out.to_str().unwrap(), 0.0, media_codec::DecodeLimits::unbounded()).expect("frame");
            frame.rgba.chunks(4).map(|p| u32::from(p[0]) + u32::from(p[1]) + u32::from(p[2])).max().unwrap_or(0)
        };
        let sharp = render(0);
        let ramped = render(15);
        assert!(sharp > 300, "zero fade must paint the text on frame 0, got {sharp}");
        assert!(ramped < sharp / 3, "a long fade must still start dark: {ramped} vs {sharp}");
    }

    /// A glyph missing from every loaded face is located and rejected before the
    /// encoder opens, instead of rendering as a blank box.
    #[test]
    fn missing_glyph_fails_preflight() {
        let film = serde_json::json!({
            "width": 64, "height": 64, "fps": 15, "duration_frames": 10,
            "scenes": [{ "start": 0, "duration": 10,
                "background": { "kind": "solid", "color": "#000000" },
                "items": [{ "kind": "text", "at": 0, "dur": 10, "content": "\u{10FFFD}", "size": 20.0,
                            "color": "#ffffff", "x": 0.5, "y": 0.5 }] }],
            "audio": null
        });
        let out = std::env::temp_dir().join("opencut-media-tests").join("missing-glyph.mp4");
        let error = super::render_project(&film.to_string(), &out, &mut |_, _| {}).expect_err("missing glyph must fail");
        assert!(error.message.contains("no glyph"), "{error}");
        assert!(!out.exists());
    }

    /// Cancellation stops between frames and leaves the previous good output
    /// untouched; the partial file never appears under the final name.
    #[test]
    fn cancel_keeps_the_previous_output() {
        if !windows_media_foundation() {
            return;
        }
        let dir = std::env::temp_dir().join("opencut-media-tests");
        std::fs::create_dir_all(&dir).unwrap();
        let out = dir.join("cancel.mp4");
        std::fs::write(&out, b"previous good output").unwrap();
        let film = serde_json::json!({
            "width": 64, "height": 64, "fps": 15, "duration_frames": 60,
            "scenes": [{ "start": 0, "duration": 60,
                "background": { "kind": "solid", "color": "#101828" }, "items": [] }],
            "audio": null
        });
        let seen = std::cell::Cell::new(0u32);
        let stop = || seen.get() >= 15;
        let options = super::RenderOptions { range: None, cancel: Some(&stop) };
        let error = super::render_project_with(&film.to_string(), &out, &options, &mut |done, _| seen.set(done))
            .expect_err("cancel must surface as an error");
        assert_eq!(error.code(), "cancelled", "{error}");
        assert_eq!(std::fs::read(&out).unwrap(), b"previous good output");
        let residue: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                let name = entry.file_name().to_string_lossy().into_owned();
                name.starts_with("cancel.") && name.ends_with(".partial.mp4")
            })
            .collect();
        assert!(residue.is_empty(), "a cancelled run must remove its partial file");
    }

    /// An interval export writes only that range, renumbered from zero.
    #[test]
    fn range_export_writes_only_the_interval() {
        if !windows_media_foundation() {
            return;
        }
        let dir = std::env::temp_dir().join("opencut-media-tests");
        std::fs::create_dir_all(&dir).unwrap();
        let out = dir.join("range.mp4");
        let film = serde_json::json!({
            "width": 64, "height": 64, "fps": 15, "duration_frames": 60,
            "scenes": [{ "start": 0, "duration": 60,
                "background": { "kind": "solid", "color": "#203040" }, "items": [] }],
            "audio": null
        });
        let options = super::RenderOptions { range: Some(super::FrameRange { start: 15, end: 45 }), cancel: None };
        let stats = super::render_project_with(&film.to_string(), &out, &options, &mut |_, _| {}).expect("range");
        assert_eq!(stats.frames, 30);
        assert_eq!(stats.range, [15, 45]);
        assert!(out.exists());
    }

    /// Vignette is opt-in: the same film keeps bright corners unless `post`
    /// declares a strength.
    #[test]
    fn vignette_is_opt_in() {
        if !windows_media_foundation() {
            return;
        }
        let dir = std::env::temp_dir().join("opencut-media-tests");
        std::fs::create_dir_all(&dir).unwrap();
        let corner = |post: serde_json::Value, name: &str| {
            let film = serde_json::json!({
                "width": 64, "height": 64, "fps": 15, "duration_frames": 4,
                "scenes": [{ "start": 0, "duration": 4,
                    "background": { "kind": "solid", "color": "#ffffff" }, "items": [] }],
                "audio": null,
                "post": post
            });
            let out = dir.join(name);
            super::render_project(&film.to_string(), &out, &mut |_, _| {}).expect("render");
            let mut library = media_codec::MediaLibrary::new();
            let frame = library.video_frame(out.to_str().unwrap(), 0.0, media_codec::DecodeLimits::unbounded()).expect("frame");
            u32::from(frame.rgba[0])
        };
        let plain = corner(serde_json::json!({}), "vignette-off.mp4");
        let dark = corner(serde_json::json!({ "vignette": 0.9 }), "vignette-on.mp4");
        assert!(plain > 200, "an undeclared vignette must leave the corner bright: {plain}");
        assert!(dark < plain, "a declared vignette must darken the corner: {dark} vs {plain}");
    }

    /// Reads back a preview PNG through the same decoder used for imports.
    fn preview_pixels(json: &str, frame: u32, name: &str) -> (u32, u32, Vec<u8>) {
        let dir = std::env::temp_dir().join("opencut-media-tests").join("preview");
        std::fs::create_dir_all(&dir).unwrap();
        let out = dir.join(name);
        let stats = super::preview_frame(json, frame, &out).expect("preview frame");
        let mut library = media_codec::MediaLibrary::new();
        let bitmap = library.image(out.to_str().unwrap(), media_codec::DecodeLimits::unbounded()).expect("decode png");
        assert_eq!((bitmap.width, bitmap.height), (stats.width, stats.height));
        (bitmap.width, bitmap.height, bitmap.rgba.clone())
    }

    /// Brightest column of a frame: the horizontal position of a white shape.
    fn brightest_column(width: u32, height: u32, rgba: &[u8]) -> u32 {
        let mut best = (0u32, 0u32);
        for x in 0..width {
            let mut sum = 0u32;
            for y in 0..height {
                let i = ((y * width + x) as usize) * 4;
                sum += u32::from(rgba[i]) + u32::from(rgba[i + 1]) + u32::from(rgba[i + 2]);
            }
            if sum > best.1 {
                best = (x, sum);
            }
        }
        best.0
    }

    /// A position track moves a node across frames through the preview path.
    #[test]
    fn keyframed_position_moves_a_node() {
        if !windows_media_foundation() {
            return;
        }
        let film = serde_json::json!({
            "width": 64, "height": 64, "fps": 15, "duration_frames": 30,
            "scenes": [{ "start": 0, "duration": 30,
                "background": { "kind": "solid", "color": "#000000" },
                "items": [{
                    "kind": "shape", "shape": "rect", "at": 0, "dur": 30,
                    "x": 0.1, "y": 0.5, "w": 0.06, "h": 0.06, "color": "#ffffff",
                    "fade_in": 0, "fade_out": 0,
                    "id": "mover",
                    "anim": { "x": { "keys": [
                        { "at": 0, "value": 0.1 },
                        { "at": 29, "value": 0.9, "ease": "linear" }
                    ] } }
                }]
            }],
            "audio": null
        });
        let json = film.to_string();
        let (w, h, early) = preview_pixels(&json, 2, "mover-early.png");
        let (_, _, late) = preview_pixels(&json, 27, "mover-late.png");
        let start = brightest_column(w, h, &early);
        let end = brightest_column(w, h, &late);
        assert!(start < 20, "the shape starts left, got column {start}");
        assert!(end > 44, "the shape ends right, got column {end}");
    }

    /// Following a path places the node along the path's length, not at its
    /// declared `x`.
    #[test]
    fn follow_path_drives_position() {
        if !windows_media_foundation() {
            return;
        }
        let film = serde_json::json!({
            "width": 64, "height": 64, "fps": 15, "duration_frames": 30,
            "scenes": [{ "start": 0, "duration": 30,
                "background": { "kind": "solid", "color": "#000000" },
                "items": [{
                    "kind": "shape", "shape": "line", "at": 0, "dur": 30,
                    "x": 0.5, "y": 0.5, "w": 0.08, "h": 0.0, "color": "#ffffff", "stroke": 6.0,
                    "fade_in": 0, "fade_out": 0,
                    "anim": {
                        "progress": { "keys": [
                            { "at": 0, "value": 0.0 },
                            { "at": 29, "value": 1.0, "ease": "linear" }
                        ] },
                        "follow": { "points": [[0.1, 0.5], [0.9, 0.5]] }
                    }
                }]
            }],
            "audio": null
        });
        let json = film.to_string();
        let (w, h, early) = preview_pixels(&json, 1, "follow-early.png");
        let (_, _, late) = preview_pixels(&json, 28, "follow-late.png");
        assert!(brightest_column(w, h, &early) < 20, "follow starts at the path head");
        assert!(brightest_column(w, h, &late) > 44, "follow ends at the path tail");
    }

    /// One component definition, two instances: the definition is painted at
    /// each placement instead of being copied into the film data.
    #[test]
    fn component_instances_reuse_one_definition() {
        if !windows_media_foundation() {
            return;
        }
        let film = serde_json::json!({
            "width": 64, "height": 64, "fps": 15, "duration_frames": 10,
            "components": [{
                "id": "dot",
                "items": [{ "kind": "shape", "shape": "rect", "at": 0, "dur": 10,
                            "x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0, "color": "#ffffff",
                            "fade_in": 0, "fade_out": 0 }]
            }],
            "scenes": [{ "start": 0, "duration": 10,
                "background": { "kind": "solid", "color": "#000000" },
                "items": [
                    { "kind": "instance", "at": 0, "dur": 10, "component": "dot",
                      "x": 0.25, "y": 0.5, "w": 0.2, "h": 0.2, "fade_in": 0, "fade_out": 0 },
                    { "kind": "instance", "at": 0, "dur": 10, "component": "dot",
                      "x": 0.75, "y": 0.5, "w": 0.2, "h": 0.2, "fade_in": 0, "fade_out": 0 }
                ]
            }],
            "audio": null
        });
        let (w, h, rgba) = preview_pixels(&film.to_string(), 0, "instances.png");
        let bright = |x: f32| {
            let column = (x * w as f32) as u32;
            let middle = h / 2;
            let i = ((middle * w + column) as usize) * 4;
            u32::from(rgba[i]) + u32::from(rgba[i + 1]) + u32::from(rgba[i + 2])
        };
        assert!(bright(0.25) > 200, "left instance must paint its component");
        assert!(bright(0.75) > 200, "right instance must paint the same definition");
        assert!(bright(0.5) < 40, "the gap between placements stays empty");
    }

    /// A component that reaches itself cannot be painted; it is rejected while
    /// the project is still data.
    #[test]
    fn component_cycles_are_rejected() {
        let film = serde_json::json!({
            "width": 64, "height": 64, "fps": 15, "duration_frames": 10,
            "components": [{
                "id": "loop",
                "items": [{ "kind": "instance", "at": 0, "dur": 10, "component": "loop",
                            "x": 0.5, "y": 0.5, "w": 1.0, "h": 1.0 }]
            }],
            "scenes": [{ "start": 0, "duration": 10, "background": { "kind": "solid", "color": "#000000" }, "items": [] }],
            "audio": null
        });
        let out = std::env::temp_dir().join("opencut-media-tests").join("cycle.png");
        let error = super::preview_frame(&film.to_string(), 0, &out).expect_err("a cycle must be rejected");
        assert!(error.message.contains("component cycle"), "{error}");
    }

    /// Motion the item cannot express is a located error, not a dropped track.
    #[test]
    fn motion_the_item_cannot_express_is_rejected() {
        let film = serde_json::json!({
            "width": 64, "height": 64, "fps": 15, "duration_frames": 10,
            "scenes": [{ "start": 0, "duration": 10,
                "background": { "kind": "solid", "color": "#000000" },
                "items": [{ "kind": "bands", "at": 0, "dur": 10, "color": "#8fb4ff",
                            "fade_in": 0, "fade_out": 0,
                            "anim": { "x": { "keys": [{ "at": 0, "value": 0.2 }] } } }]
            }],
            "audio": null
        });
        let out = std::env::temp_dir().join("opencut-media-tests").join("bad-motion.png");
        let error = super::preview_frame(&film.to_string(), 0, &out).expect_err("unsupported motion must fail");
        assert!(error.message.contains("does not apply"), "{error}");
    }

    /// A previewed frame and the exported frame come from the same painter: a
    /// flat frame must agree within codec tolerance.
    #[test]
    fn preview_matches_the_exported_frame() {
        if !windows_media_foundation() {
            return;
        }
        let film = serde_json::json!({
            "width": 64, "height": 64, "fps": 15, "duration_frames": 6,
            "scenes": [{ "start": 0, "duration": 6,
                "background": { "kind": "solid", "color": "#3060a0" }, "items": [] }],
            "audio": null
        });
        let json = film.to_string();
        let (_, _, preview) = preview_pixels(&json, 2, "identity.png");
        let out = std::env::temp_dir().join("opencut-media-tests").join("identity.mp4");
        super::render_project(&json, &out, &mut |_, _| {}).expect("render");
        let mut library = media_codec::MediaLibrary::new();
        let exported = library.video_frame(out.to_str().unwrap(), 2.0 / 15.0, media_codec::DecodeLimits::unbounded()).expect("frame");
        let middle = ((32 * 64 + 32) as usize) * 4;
        for channel in 0..3 {
            let a = i32::from(preview[middle + channel]);
            let b = i32::from(exported.rgba[middle + channel]);
            assert!((a - b).abs() <= 24, "channel {channel}: preview {a} vs export {b}");
        }
    }
}
