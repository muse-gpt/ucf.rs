//! Scene IR: the whole film is one serializable data structure.
//!
//! The TypeScript authoring surface compiles to this shape and the renderer
//! consumes nothing else. Frame counts are the unit of time inside the IR;
//! seconds belong to the authoring layer.

use serde::{Deserialize, Serialize};

// Camera keyframes, easing and sampling live in the geometry library: a camera
// is a geometry pose plus a curve, and both are shared with downstream hosts.
pub use geometry::{CameraKey, CameraPose, Ease, camera_at};

/// A complete renderable project.
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct Project {
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub duration_frames: u32,
    pub scenes: Vec<Scene>,
    #[serde(default)]
    pub audio: Option<AudioSpec>,
    /// One camera curve for the whole film. When present it drives every frame
    /// continuously, which is what makes a one-take move possible.
    #[serde(default)]
    pub camera: Vec<CameraKey>,
}

/// One temporal block with a background and its items.
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct Scene {
    pub start: u32,
    pub duration: u32,
    #[serde(default)]
    pub background: Background,
    #[serde(default)]
    pub items: Vec<Item>,
    /// Camera move for this scene: content renders off-screen, then this
    /// transform pushes/pulls and pans it into the output frame.
    #[serde(default)]
    pub camera: Option<Camera>,
}

/// A single camera move, interpolated across the whole scene.
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct Camera {
    #[serde(default = "default_one")]
    pub zoom_from: f32,
    #[serde(default = "default_one")]
    pub zoom_to: f32,
    /// Focus point at the start and end, in normalized frame coordinates.
    #[serde(default = "default_center")]
    pub focus_from: [f32; 2],
    #[serde(default = "default_center")]
    pub focus_to: [f32; 2],
    #[serde(default)]
    pub ease: Ease,
    /// Extra rotation in degrees at the end of the move.
    #[serde(default)]
    pub rotate_to: f32,
}

fn default_center() -> [f32; 2] {
    [0.5, 0.5]
}

/// Painted behind the items for the life of the scene.
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(tag = "kind", rename_all = "kebab-case", rename_all_fields = "camelCase")]
pub enum Background {
    Solid {
        color: String,
    },
    VGradient {
        colors: Vec<String>,
    },
    Radial {
        center: String,
        edge: String,
    },
    Starfield {
        base: Vec<String>,
        #[serde(default = "default_stars")]
        stars: u32,
        #[serde(default)]
        tint: Option<String>,
        /// Horizontal drift in pixels per second.
        #[serde(default)]
        drift: f32,
        /// Twinkle depth 0..=1.
        #[serde(default = "default_twinkle")]
        twinkle: f32,
    },
}

impl Default for Background {
    fn default() -> Self {
        Self::Starfield {
            base: vec!["#05070f".to_string(), "#0b1226".to_string()],
            stars: 140,
            tint: Some("#123a6b".to_string()),
            drift: 3.0,
            twinkle: 0.6,
        }
    }
}

fn default_stars() -> u32 {
    140
}

fn default_twinkle() -> f32 {
    0.6
}

fn default_fade() -> u32 {
    10
}

fn default_stagger() -> u32 {
    1
}

fn default_one_f32() -> f32 {
    1.0
}

/// Horizontal anchor of a text-anchored item.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum Align {
    Left,
    #[default]
    Center,
    Right,
}

/// How a shape is composited onto the frame.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum Blend {
    #[default]
    Normal,
    Add,
}

fn default_band_count() -> u32 {
    3
}
fn default_band_angle() -> f32 {
    -18.0
}
fn default_band_speed() -> f32 {
    26.0
}
fn default_band_alpha() -> f32 {
    0.08
}
fn default_band_thickness() -> f32 {
    90.0
}
fn default_grain() -> f32 {
    0.05
}
fn default_grain_density() -> u32 {
    9_000
}

fn default_layer_edge() -> f32 {
    0.25
}

/// Primitive shape kinds.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum ShapeKind {
    Rect,
    Circle,
    Ring,
    Line,
}

/// How a decoded frame is mapped into its layout box.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum Fit {
    /// Scale to fit inside the box, preserving aspect ratio (letterbox).
    #[default]
    Contain,
    /// Fill the box, preserving aspect ratio, cropping the overflow.
    Cover,
    /// Fill the box exactly, ignoring aspect ratio.
    Stretch,
}

/// Syntax tone of a code line fragment.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum Tone {
    Plain,
    Keyword,
    String,
    Comment,
    Number,
    Fn,
}

/// Color roles for the built-in code highlighter.
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct CodeTheme {
    #[serde(default)]
    pub panel: Option<String>,
    #[serde(default = "default_plain")]
    pub plain: String,
    #[serde(default = "default_keyword")]
    pub keyword: String,
    #[serde(default = "default_string")]
    pub string: String,
    #[serde(default = "default_comment")]
    pub comment: String,
    #[serde(default = "default_number")]
    pub number: String,
    #[serde(default = "default_fn")]
    pub function: String,
}

impl Default for CodeTheme {
    fn default() -> Self {
        Self {
            panel: Some("#0d1526e0".to_string()),
            plain: default_plain(),
            keyword: default_keyword(),
            string: default_string(),
            comment: default_comment(),
            number: default_number(),
            function: default_fn(),
        }
    }
}

fn default_plain() -> String {
    "#d6e2f5".to_string()
}
fn default_keyword() -> String {
    "#7aa2ff".to_string()
}
fn default_string() -> String {
    "#6fd6a4".to_string()
}
fn default_comment() -> String {
    "#5d6f8c".to_string()
}
fn default_number() -> String {
    "#e8b86d".to_string()
}
fn default_fn() -> String {
    "#c792ea".to_string()
}

/// One animated element inside a scene. `at` and `dur` are scene-relative frames.
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(tag = "kind", rename_all = "kebab-case", rename_all_fields = "camelCase")]
pub enum Item {
    Text {
        at: u32,
        dur: u32,
        content: String,
        #[serde(default)]
        sub: Option<String>,
        size: f32,
        #[serde(default)]
        bold: bool,
        color: String,
        #[serde(default)]
        sub_color: Option<String>,
        #[serde(default)]
        sub_size: Option<f32>,
        x: f32,
        y: f32,
        #[serde(default)]
        align: Align,
        #[serde(default = "default_fade")]
        fade_in: u32,
        #[serde(default = "default_fade")]
        fade_out: u32,
        #[serde(default)]
        drift_y: f32,
        #[serde(default)]
        max_width: Option<f32>,
        #[serde(default)]
        line_height: Option<f32>,
        #[serde(default)]
        typing: Option<f32>,
        #[serde(default)]
        glow: Option<String>,
        /// Character entrance: each glyph rises and fades in on a stagger.
        #[serde(default = "default_stagger")]
        stagger: u32,
        /// Pixels each glyph rises from, when `stagger` > 0.
        #[serde(default)]
        rise: f32,
        /// Reveal rectangle in normalized coordinates: glyphs only paint inside
        /// it. Used for text that grows out of a shape.
        #[serde(default)]
        clip: Option<[f32; 4]>,
    },
    Code {
        at: u32,
        dur: u32,
        lines: Vec<String>,
        x: f32,
        y: f32,
        size: f32,
        #[serde(default)]
        theme: CodeTheme,
        #[serde(default = "default_fade")]
        fade_in: u32,
        #[serde(default = "default_fade")]
        fade_out: u32,
        #[serde(default)]
        typing: Option<f32>,
        #[serde(default)]
        line_height: Option<f32>,
    },
    Counter {
        at: u32,
        dur: u32,
        from: f64,
        to: f64,
        x: f32,
        y: f32,
        size: f32,
        color: String,
        #[serde(default)]
        prefix: Option<String>,
        #[serde(default)]
        suffix: Option<String>,
        #[serde(default)]
        decimals: u32,
        #[serde(default)]
        ease: Ease,
        #[serde(default = "default_fade")]
        fade_in: u32,
        #[serde(default = "default_fade")]
        fade_out: u32,
    },
    Card {
        at: u32,
        dur: u32,
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        title: String,
        #[serde(default)]
        body: Option<String>,
        #[serde(default)]
        accent: Option<String>,
        #[serde(default)]
        panel: Option<String>,
        #[serde(default = "default_fade")]
        fade_in: u32,
        #[serde(default = "default_fade")]
        fade_out: u32,
        #[serde(default)]
        drift_y: f32,
    },
    Shape {
        at: u32,
        dur: u32,
        shape: ShapeKind,
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        color: String,
        #[serde(default)]
        radius: f32,
        #[serde(default)]
        stroke: Option<f32>,
        /// Fraction of the shape drawn, used for lines that draw themselves.
        #[serde(default = "default_one_f32")]
        progress: f32,
        /// `normal` or `add` (light-like overlays).
        #[serde(default)]
        blend: Blend,
        #[serde(default = "default_fade")]
        fade_in: u32,
        #[serde(default = "default_fade")]
        fade_out: u32,
    },
    /// Soft diagonal light bands that drift across the frame.
    Bands {
        at: u32,
        dur: u32,
        #[serde(default = "default_band_count")]
        count: u32,
        #[serde(default = "default_band_angle")]
        angle: f32,
        /// Vertical drift in pixels per second.
        #[serde(default = "default_band_speed")]
        speed: f32,
        color: String,
        #[serde(default = "default_band_alpha")]
        alpha: f32,
        #[serde(default = "default_band_thickness")]
        thickness: f32,
        #[serde(default = "default_fade")]
        fade_in: u32,
        #[serde(default = "default_fade")]
        fade_out: u32,
    },
    /// Deterministic per-frame film grain.
    Grain {
        at: u32,
        dur: u32,
        #[serde(default = "default_grain")]
        strength: f32,
        #[serde(default = "default_grain_density")]
        density: u32,
        #[serde(default = "default_fade")]
        fade_in: u32,
        #[serde(default = "default_fade")]
        fade_out: u32,
    },
    /// A filled and/or stroked path. Points are normalized frame coordinates.
    Path {
        at: u32,
        dur: u32,
        points: Vec<[f32; 2]>,
        /// Treat the points as a smooth Catmull-Rom curve instead of a polyline.
        #[serde(default)]
        smooth: bool,
        #[serde(default)]
        closed: bool,
        #[serde(default)]
        fill: Option<String>,
        /// Stroke color.
        color: String,
        #[serde(default)]
        stroke: Option<f32>,
        /// Fraction of the outline drawn, for a path that draws itself.
        #[serde(default = "default_one_f32")]
        progress: f32,
        /// Rotation in degrees around the path centroid.
        #[serde(default)]
        rotate: f32,
        /// Scale around the path centroid.
        #[serde(default = "default_one_f32")]
        scale: f32,
        #[serde(default)]
        blend: Blend,
        #[serde(default = "default_fade")]
        fade_in: u32,
        #[serde(default = "default_fade")]
        fade_out: u32,
    },
    /// A nested layer whose opacity follows the camera zoom. Layers crossing at
    /// the same zoom fade into each other, so a one-take push never cuts.
    Layer {
        at: u32,
        dur: u32,
        /// Zoom window in which this layer is fully visible.
        zoom_from: f32,
        zoom_to: f32,
        /// Width of the fade at both ends of the window, in zoom ratio.
        #[serde(default = "default_layer_edge")]
        edge: f32,
        #[serde(default)]
        items: Vec<Item>,
    },
    /// A soft additive light source. Additive falloff reads as light in air,
    /// which a flat disc never does.
    Glow {
        at: u32,
        dur: u32,
        x: f32,
        y: f32,
        /// Radius as a fraction of the frame width.
        radius: f32,
        #[serde(default = "default_glow_color")]
        color: String,
        /// Peak opacity at the center, 0..1.
        #[serde(default = "default_one")]
        intensity: f32,
        #[serde(default = "default_fade")]
        fade_in: u32,
        #[serde(default = "default_fade")]
        fade_out: u32,
    },
    Footnote {
        at: u32,
        dur: u32,
        text: String,
        #[serde(default = "default_note_size")]
        size: f32,
        x: f32,
        y: f32,
        #[serde(default)]
        align: Align,
        #[serde(default = "default_note_color")]
        color: String,
        #[serde(default = "default_fade")]
        fade_in: u32,
        #[serde(default = "default_fade")]
        fade_out: u32,
    },
    /// A still image decoded from disk and drawn into a layout box.
    Image {
        at: u32,
        dur: u32,
        src: String,
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        #[serde(default)]
        fit: Fit,
        #[serde(default = "default_one")]
        opacity: f32,
        #[serde(default)]
        radius: f32,
        #[serde(default = "default_fade")]
        fade_in: u32,
        #[serde(default = "default_fade")]
        fade_out: u32,
    },
    /// A video stream decoded on demand; `src_in` is the first source frame.
    Video {
        at: u32,
        dur: u32,
        src: String,
        #[serde(default)]
        src_in: u32,
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        #[serde(default)]
        fit: Fit,
        #[serde(default = "default_one")]
        opacity: f32,
        #[serde(default)]
        radius: f32,
        #[serde(default = "default_fade")]
        fade_in: u32,
        #[serde(default = "default_fade")]
        fade_out: u32,
    },
    Progress {
        at: u32,
        dur: u32,
        #[serde(default)]
        track: Option<String>,
        #[serde(default)]
        fill: Option<String>,
        #[serde(default)]
        y: Option<f32>,
    },
}

fn default_note_size() -> f32 {
    26.0
}

fn default_note_color() -> String {
    "#5d6f8c".to_string()
}

fn default_glow_color() -> String {
    "#f2c98a".to_string()
}

/// Audio bed specification; everything is synthesized in-process except the
/// reserved narration slot which mixes a PCM WAV file when provided.
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
#[serde(rename_all = "camelCase")]
pub struct AudioSpec {
    #[serde(default)]
    pub bgm: Option<BgmSpec>,
    #[serde(default)]
    pub narration: Option<NarrationSpec>,
    #[serde(default)]
    pub sfx: Vec<SfxSpec>,
    /// Imported audio placed on the timeline, mixed over the synthesized bed.
    #[serde(default)]
    pub clips: Vec<AudioClipSpec>,
}

/// One imported audio file placed at `at` frames, playing from `offset`.
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct AudioClipSpec {
    pub src: String,
    #[serde(default)]
    pub at: u32,
    /// Length in frames; omitted means "until the file ends".
    #[serde(default)]
    pub dur: Option<u32>,
    /// Source offset in frames skipped before playback starts.
    #[serde(default)]
    pub offset_frames: u32,
    #[serde(default = "default_one")]
    pub volume: f32,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct BgmSpec {
    #[serde(default = "default_bgm_volume")]
    pub volume: f32,
    /// Reserved for future moods; only `pad` exists today.
    #[serde(default = "default_bgm_mood")]
    pub mood: String,
}

fn default_bgm_volume() -> f32 {
    0.16
}

fn default_bgm_mood() -> String {
    "pad".to_string()
}

/// Narration is a slot, not a generator: OpenCut mixes the file you provide.
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct NarrationSpec {
    pub wav: String,
    #[serde(default)]
    pub delay_frames: u32,
    #[serde(default = "default_one")]
    pub volume: f32,
}

fn default_one() -> f32 {
    1.0
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct SfxSpec {
    pub at: u32,
    pub kind: String,
    #[serde(default = "default_sfx_volume")]
    pub volume: f32,
}

fn default_sfx_volume() -> f32 {
    0.5
}
