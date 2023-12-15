//! Font loading, glyph rasterization with caching, line layout, the code
//! highlighter and number formatting. Text shaping lives here so the painter
//! only ever blends cached bitmaps.

use std::collections::HashMap;

use fontdue::{Font, FontSettings};

use geometry::Rgba;
use media_image::Frame;
use media_raster::Paint;
use media_scene::{Align, Tone};

const REGULAR: &[&str] = &["C:\\Windows\\Fonts\\msyh.ttc", "C:\\Windows\\Fonts\\NotoSansCJK-Regular.ttc"];
const BOLD: &[&str] = &["C:\\Windows\\Fonts\\msyhbd.ttc", "C:\\Windows\\Fonts\\NotoSansCJK-Bold.ttc"];
const MONO: &[&str] = &["C:\\Windows\\Fonts\\consola.ttf", "C:\\Windows\\Fonts\\DejaVuSansMono.ttf"];

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum FontId {
    Regular,
    Bold,
    Mono,
}

/// Loaded fonts plus the raster cache shared by the whole render.
pub struct Fonts {
    regular: Font,
    bold: Font,
    mono: Font,
    cache: HashMap<(FontId, char, u32), CachedGlyph>,
}

/// One rasterized glyph: advance, x offset, top offset, size, coverage.
type CachedGlyph = (f32, f32, f32, f32, f32, Vec<u8>);

struct Glyph {
    advance: f32,
    xmin: f32,
    top_from_baseline: f32,
    width: f32,
    height: f32,
    bitmap: Vec<u8>,
}

impl Fonts {
    /// Loads the first available file from each role.
    pub fn load() -> Result<Self, media_types::Error> {
        let pick = |paths: &[&str]| -> Result<Vec<u8>, media_types::Error> {
            for p in paths {
                if let Ok(bytes) = std::fs::read(p) {
                    return Ok(bytes);
                }
            }
            Err(media_types::Error::io(format!("no font found among: {}", paths.join(", "))))
        };
        let settings = FontSettings { collection_index: 0, ..Default::default() };
        let regular =
            Font::from_bytes(pick(REGULAR)?, settings).map_err(|e| media_types::Error::io(format!("regular font: {e}")))?;
        let bold = Font::from_bytes(pick(BOLD)?, settings).map_err(|e| media_types::Error::io(format!("bold font: {e}")))?;
        let mono = Font::from_bytes(pick(MONO)?, settings).map_err(|e| media_types::Error::io(format!("mono font: {e}")))?;
        Ok(Self { regular, bold, mono, cache: HashMap::new() })
    }

    fn font(&self, id: FontId) -> &Font {
        match id {
            FontId::Regular => &self.regular,
            FontId::Bold => &self.bold,
            FontId::Mono => &self.mono,
        }
    }

    /// Resolves the font that can actually draw `ch`: the mono face has no CJK
    /// coverage, so those characters fall back to the regular face.
    fn resolve(&self, id: FontId, ch: char) -> FontId {
        if id == FontId::Mono && !self.mono.has_glyph(ch) && self.regular.has_glyph(ch) { FontId::Regular } else { id }
    }

    fn glyph(&mut self, id: FontId, ch: char, size: f32) -> Glyph {
        let id = self.resolve(id, ch);
        let key = (id, ch, (size * 4.0) as u32);
        if let Some(hit) = self.cache.get(&key) {
            return Glyph {
                advance: hit.0,
                xmin: hit.1,
                top_from_baseline: hit.2,
                width: hit.3,
                height: hit.4,
                bitmap: hit.5.clone(),
            };
        }
        let font = self.font(id);
        let (metrics, bitmap) = font.rasterize(ch, size);
        // `metrics.width`/`height` are the bitmap size in whole pixels and are
        // exactly the row stride fontdue packs; `bounds` is only the ink box.
        let top_from_baseline = (metrics.ymin + metrics.height as i32) as f32;
        let entry = (
            metrics.advance_width,
            metrics.xmin as f32,
            top_from_baseline,
            metrics.width as f32,
            metrics.height as f32,
            bitmap,
        );
        self.cache.insert(key, entry.clone());
        Glyph { advance: entry.0, xmin: entry.1, top_from_baseline: entry.2, width: entry.3, height: entry.4, bitmap: entry.5 }
    }

    /// Measures a single line without wrapping.
    #[must_use]
    pub fn measure(&mut self, text: &str, size: f32, bold: bool, mono: bool) -> f32 {
        let id = if mono {
            FontId::Mono
        } else if bold {
            FontId::Bold
        } else {
            FontId::Regular
        };
        let mut pen = 0.0f32;
        for ch in text.chars() {
            let g = self.glyph(id, ch, size);
            pen += g.advance;
        }
        pen
    }

    /// Lays out `text` into wrapped lines limited to `max_width` pixels.
    #[must_use]
    pub fn wrap(&mut self, text: &str, size: f32, bold: bool, mono: bool, max_width: Option<f32>) -> Vec<String> {
        let id = if mono {
            FontId::Mono
        } else if bold {
            FontId::Bold
        } else {
            FontId::Regular
        };
        let mut lines: Vec<String> = Vec::new();
        for hard in text.split('\n') {
            let Some(limit) = max_width else {
                lines.push(hard.to_string());
                continue;
            };
            let units = split_units(hard);
            let mut current = String::new();
            let mut width = 0.0f32;
            for unit in units {
                let mut uw = 0.0f32;
                for ch in unit.chars() {
                    uw += self.glyph(id, ch, size).advance;
                }
                if width + uw > limit && width > 0.0 {
                    lines.push(std::mem::take(&mut current));
                    width = 0.0;
                }
                current.push_str(&unit);
                width += uw;
            }
            if !current.is_empty() {
                lines.push(current);
            } else if lines.is_empty() || hard.is_empty() {
                lines.push(String::new());
            }
        }
        lines
    }

    /// Draws one line from `x` (left) at the given baseline y.
    pub fn draw_line(&mut self, frame: &mut Frame, text: &str, x: f32, baseline: f32, size: f32, id: FontId, color: Rgba) {
        let mut pen = x;
        for ch in text.chars() {
            let g = self.glyph(id, ch, size);
            if !g.bitmap.is_empty() {
                let gx = (pen + g.xmin).round() as i32;
                let gy = (baseline - g.top_from_baseline).round() as i32;
                for row in 0..g.height as usize {
                    let py = gy + row as i32;
                    if py < 0 || py >= frame.height as i32 {
                        continue;
                    }
                    for col in 0..g.width as usize {
                        let px = gx + col as i32;
                        if px < 0 || px >= frame.width as i32 {
                            continue;
                        }
                        let a = g.bitmap[row * g.width as usize + col] as u32;
                        if a == 0 {
                            continue;
                        }
                        let alpha = (a as f32 / 255.0 * (color.0[3] as f32 / 255.0)) * 255.0;
                        frame.blend(px as u32, py as u32, Rgba([color.0[0], color.0[1], color.0[2], alpha as u8]));
                    }
                }
            }
            pen += g.advance;
        }
    }

    /// Draws one line character by character, each glyph rising into place on
    /// a stagger. `life` is frames since the item appeared.
    #[allow(clippy::too_many_arguments)]
    pub fn draw_staggered(
        &mut self,
        frame: &mut Frame,
        text: &str,
        x: f32,
        top: f32,
        size: f32,
        bold: bool,
        color: Rgba,
        align: Align,
        stagger: u32,
        rise: f32,
        life: u32,
    ) -> f32 {
        let id = if bold { FontId::Bold } else { FontId::Regular };
        let total = self.measure(text, size, bold, false);
        let mut pen = match align {
            Align::Left => x,
            Align::Center => x - total / 2.0,
            Align::Right => x - total,
        };
        let per_char_fade = 8u32;
        let stagger = stagger.max(1);
        for (index, ch) in text.chars().enumerate() {
            let start = index as u32 * stagger;
            let p = if life <= start { 0.0 } else { ((life - start) as f32 / per_char_fade as f32).clamp(0.0, 1.0) };
            let advance = self.measure(&ch.to_string(), size, bold, false);
            if p > 0.001 {
                let eased = 1.0 - (1.0 - p).powi(3);
                let offset = (1.0 - eased) * rise;
                let alpha = color.0[3] as f32 * eased;
                let c = Rgba([color.0[0], color.0[1], color.0[2], alpha as u8]);
                self.draw_line(frame, &ch.to_string(), pen, top + size * 0.82 + offset, size, id, c);
            }
            pen += advance;
        }
        size * 1.2
    }

    /// Draws wrapped text anchored at `x`/`y` (top) with alignment; returns height.
    pub fn draw_block(
        &mut self,
        frame: &mut Frame,
        text: &str,
        x: f32,
        y: f32,
        size: f32,
        bold: bool,
        color: Rgba,
        align: Align,
        max_width: Option<f32>,
        line_height: f32,
    ) -> f32 {
        let lines = self.wrap(text, size, bold, false, max_width);
        let id = if bold { FontId::Bold } else { FontId::Regular };
        let mut height = 0.0f32;
        for line in &lines {
            let w = self.measure(line, size, bold, false);
            let lx = match align {
                Align::Left => x,
                Align::Center => x - w / 2.0,
                Align::Right => x - w,
            };
            self.draw_line(frame, line, lx, y + size * 0.82 + height, size, id, color);
            height += size * line_height;
        }
        height
    }

    /// Draws a code line with tone coloring. Returns width.
    pub fn draw_code_line(
        &mut self,
        frame: &mut Frame,
        line: &str,
        x: f32,
        baseline: f32,
        size: f32,
        theme: &media_scene::CodeTheme,
        visible: Option<usize>,
    ) {
        let tokens = highlight(line);
        let mut pen = x;
        let mut shown = 0usize;
        for (text, tone) in &tokens {
            let color = match tone {
                Tone::Keyword => geometry::parse(&theme.keyword).unwrap_or_default(),
                Tone::String => geometry::parse(&theme.string).unwrap_or_default(),
                Tone::Comment => geometry::parse(&theme.comment).unwrap_or_default(),
                Tone::Number => geometry::parse(&theme.number).unwrap_or_default(),
                Tone::Fn => geometry::parse(&theme.function).unwrap_or_default(),
                Tone::Plain => geometry::parse(&theme.plain).unwrap_or_default(),
            };
            let mut local = String::new();
            for ch in text.chars() {
                if visible.is_some_and(|v| shown >= v) {
                    break;
                }
                local.push(ch);
                shown += 1;
            }
            if local.is_empty() {
                break;
            }
            self.draw_line(frame, &local, pen, baseline, size, FontId::Mono, color);
            pen += self.measure(&local, size, false, true);
        }
    }
}

fn is_cjk(ch: char) -> bool {
    matches!(ch as u32,
        0x2E80..=0x303F | 0x3040..=0x33FF | 0x3400..=0x4DBF | 0x4E00..=0x9FFF |
        0xF900..=0xFAFF | 0xFF00..=0xFFEF | 0x20000..=0x2FA1F)
}

/// Splits a line into wrap units: Latin runs, single CJK characters, spaces
/// attach to the preceding run.
fn split_units(line: &str) -> Vec<String> {
    let mut units: Vec<String> = Vec::new();
    let mut current = String::new();
    for ch in line.chars() {
        if is_cjk(ch) {
            if !current.is_empty() {
                units.push(std::mem::take(&mut current));
            }
            units.push(ch.to_string());
        } else {
            current.push(ch);
            if ch == ' ' {
                units.push(std::mem::take(&mut current));
            }
        }
    }
    if !current.is_empty() {
        units.push(current);
    }
    units
}

/// Formats a counter value with grouping and fixed decimals.
#[must_use]
pub fn format_number(value: f64, decimals: u32) -> String {
    let scaled = value;
    let negative = scaled < 0.0;
    let abs = scaled.abs();
    let factor = 10f64.powi(decimals as i32);
    let rounded = (abs * factor).round();
    let integer = (rounded / factor).floor() as i64;
    let frac = (rounded - integer as f64 * factor).round() as i64;
    let digits = integer.abs().to_string();
    let mut grouped = String::new();
    for (i, ch) in digits.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            grouped.push(',');
        }
        grouped.push(ch);
    }
    let grouped: String = grouped.chars().rev().collect();
    let body = if decimals > 0 { format!("{grouped}.{:0width$}", frac, width = decimals as usize) } else { grouped };
    if negative { format!("-{body}") } else { body }
}

/// Splits a source line into highlighted tokens. Deliberately small: this is
/// a display highlighter, not a parser.
fn highlight(line: &str) -> Vec<(String, Tone)> {
    let keywords = [
        "const", "let", "var", "new", "import", "from", "export", "await", "async", "return", "if", "else", "for", "in", "fn",
        "pub", "struct", "class", "this", "default",
    ];
    let trimmed_start = line.len() - line.trim_start().len();
    let mut out: Vec<(String, Tone)> = Vec::new();
    if trimmed_start > 0 {
        out.push((line[..trimmed_start].to_string(), Tone::Plain));
    }
    let rest = &line[trimmed_start..];
    if rest.starts_with("//") || rest.starts_with('#') {
        out.push((rest.to_string(), Tone::Comment));
        return out;
    }
    let chars: Vec<char> = rest.chars().collect();
    let mut i = 0usize;
    let mut plain = String::new();
    let flush = |plain: &mut String, out: &mut Vec<(String, Tone)>| {
        if !plain.is_empty() {
            out.push((std::mem::take(plain), Tone::Plain));
        }
    };
    while i < chars.len() {
        let ch = chars[i];
        if ch == '"' || ch == '\'' || ch == '`' {
            flush(&mut plain, &mut out);
            let quote = ch;
            let mut s = String::from(ch);
            i += 1;
            while i < chars.len() {
                s.push(chars[i]);
                if chars[i] == quote {
                    i += 1;
                    break;
                }
                i += 1;
            }
            out.push((s, Tone::String));
            continue;
        }
        if ch == '/' && chars.get(i + 1) == Some(&'/') {
            flush(&mut plain, &mut out);
            out.push((chars[i..].iter().collect::<String>(), Tone::Comment));
            break;
        }
        if ch.is_ascii_digit() {
            flush(&mut plain, &mut out);
            let mut s = String::new();
            while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                s.push(chars[i]);
                i += 1;
            }
            out.push((s, Tone::Number));
            continue;
        }
        if ch.is_alphabetic() || ch == '_' || ch == '$' {
            let mut word = String::new();
            while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_' || chars[i] == '$') {
                word.push(chars[i]);
                i += 1;
            }
            flush(&mut plain, &mut out);
            let tone = if keywords.contains(&word.as_str()) {
                Tone::Keyword
            } else if chars.get(i) == Some(&'(') {
                Tone::Fn
            } else {
                Tone::Plain
            };
            out.push((word, tone));
            continue;
        }
        plain.push(ch);
        i += 1;
    }
    flush(&mut plain, &mut out);
    out
}
