//! 镜头级显示空间算子：soft bloom、vignette、film grain、ember pulse、slam
//! stamp、渐变映射 LUT 与转场混合。
//!
//! 全部算子的输入域是 [`InputDomain::DisplayRgba8`]——8 位显示字节按合同直接
//! 参与算术（见 [`crate::contract`]），不冒充线性物理运算。数值语义逐值吸收自
//! 帧合成器的 CPU 参考通路（`muse-media` G7 迁移），alpha 直通不写。
//!
//! 命名裁决（G4/G7 合同）：
//! - [`SoftBloom`] 是显示空间 5×5 阈值柔光，与效果栈绑定线性 HDR 物理路径的
//!   `bloom`（[`crate::bloom_pass`]）是两个不同算子，不得混用。
//! - [`Vignette`] 是暗角压暗的唯一正式公式（`smoothstep(0.25, 0.95, d·√2)`
//!   乘暗）；旧的 `media.raster::build_vignette` 掩码构造已删除，不再存在
//!   第二套暗角语义。
//! - [`FilmGrain`] 是逐像素全帧噪声；效果栈的 `grain`（[`crate::display_ops`]）
//!   是密度派生的稀疏颗粒，二者语义不同、名称不同。
//! - 噪声哈希 [`hash21`] 是显示空间噪声的唯一正式随机合同（中间 fract 形态），
//!   GPU 着色器必须逐运算复现它；CPU 与 GPU 的颗粒一致性以此为准。

use geometry_motion::smoothstep;
use media_image::Frame;

/// 显示空间噪声的正式哈希：与帧合成器 WGSL `hash21` 逐运算同序，
/// f32 确定性保证 CPU/GPU 同输入同位。
#[must_use]
pub fn hash21(x: f32, y: f32) -> f32 {
    let mut p3x = fract(x * 0.103_1);
    let mut p3y = fract(y * 0.103_1);
    let mut p3z = fract(x * 0.103_1);
    let d = p3x * (p3y + 33.33) + p3y * (p3z + 33.33) + p3z * (p3x + 33.33);
    p3x = fract(p3x + d);
    p3y = fract(p3y + d);
    p3z = fract(p3z + d);
    fract((p3x + p3y) * p3z)
}

fn fract(v: f32) -> f32 {
    v - v.floor()
}

/// 显示空间柔光：5×5 采样步距 2、逐通道硬阈 0.62、增益 ×2.5。
///
/// 这是镜头 FX 的 RGBA8 艺术效果；线性 HDR 物理 bloom 归
/// [`crate::bloom_pass`]，两者不得互相冒充。
pub struct SoftBloom {
    /// 0..1 强度，0 恒等。
    pub strength: f32,
}

impl SoftBloom {
    #[must_use]
    pub fn new(strength: f32) -> Self {
        Self { strength }
    }

    pub fn apply(&self, frame: &mut Frame) {
        let strength = self.strength.clamp(0.0, 1.0);
        if strength <= 0.0 {
            return;
        }
        let width = frame.width;
        let height = frame.height;
        let src = frame.pixels.clone();
        let threshold = 0.62_f32;
        for y in 0..height {
            for x in 0..width {
                let mut acc = [0.0_f32; 3];
                for oy in -2i32..=2 {
                    for ox in -2i32..=2 {
                        let sx = (x as i32 + ox * 2).clamp(0, width as i32 - 1) as u32;
                        let sy = (y as i32 + oy * 2).clamp(0, height as i32 - 1) as u32;
                        let base = ((sy * width + sx) * 4) as usize;
                        for c in 0..3 {
                            let v = src[base + c] as f32 / 255.0;
                            acc[c] += (v - threshold).max(0.0);
                        }
                    }
                }
                let dst = ((y * width + x) * 4) as usize;
                for c in 0..3 {
                    let add = (acc[c] / 25.0) * strength * 2.5;
                    let v = (src[dst + c] as f32 / 255.0 + add).clamp(0.0, 1.0);
                    frame.pixels[dst + c] = (v * 255.0).round() as u8;
                }
            }
        }
    }
}

/// 径向暗角：`smoothstep(0.25, 0.95, d·√2)` 乘暗。暗角唯一正式公式。
pub struct Vignette {
    /// 0..1 强度，0 恒等。
    pub strength: f32,
}

impl Vignette {
    #[must_use]
    pub fn new(strength: f32) -> Self {
        Self { strength }
    }

    pub fn apply(&self, frame: &mut Frame) {
        let strength = self.strength.clamp(0.0, 1.0);
        if strength <= 0.0 {
            return;
        }
        let width = frame.width;
        let height = frame.height;
        for y in 0..height {
            for x in 0..width {
                let u = (x as f32 + 0.5) / width as f32;
                let v = (y as f32 + 0.5) / height as f32;
                let d = ((u - 0.5).hypot(v - 0.5)) * std::f32::consts::SQRT_2;
                let edge = smoothstep(0.25, 0.95, d);
                let factor = 1.0 - strength * edge;
                let p = ((y * width + x) * 4) as usize;
                for c in 0..3 {
                    frame.pixels[p + c] = (frame.pixels[p + c] as f32 * factor).round() as u8;
                }
            }
        }
    }
}

/// 逐像素全帧胶片颗粒：`hash21(uv·1024 + t·(97, 53))·2−1` 加性噪声。
///
/// `time` 是归一化时刻（0..1），同一 time 两次渲染逐位一致（可按时刻独立
/// 求值，乱序与并行取帧可复现）。与效果栈 `grain` 的稀疏颗粒语义不同。
pub struct FilmGrain {
    /// 0..1 强度，0 恒等。
    pub strength: f32,
}

impl FilmGrain {
    #[must_use]
    pub fn new(strength: f32) -> Self {
        Self { strength }
    }

    pub fn apply(&self, frame: &mut Frame, time: f32) {
        let strength = self.strength.clamp(0.0, 1.0);
        if strength <= 0.0 {
            return;
        }
        let time = time.clamp(0.0, 1.0);
        let width = frame.width;
        let height = frame.height;
        for y in 0..height {
            for x in 0..width {
                let u = (x as f32 + 0.5) / width as f32;
                let v = (y as f32 + 0.5) / height as f32;
                let n = hash21(u * 1024.0 + time * 97.0, v * 1024.0 + time * 53.0) * 2.0 - 1.0;
                let p = ((y * width + x) * 4) as usize;
                for c in 0..3 {
                    let value = (frame.pixels[p + c] as f32 / 255.0 + n * strength).clamp(0.0, 1.0);
                    frame.pixels[p + c] = (value * 255.0).round() as u8;
                }
            }
        }
    }
}

/// 中心余烬辉光：`exp(−r²·14)`、色 (1, 0.72, 0.35)、sin 包络。
///
/// `shot_progress` 是镜头内归一化进度（FX 驱动量，不是绝对时刻）。
pub struct EmberPulse {
    /// 0..1 强度，0 恒等。
    pub strength: f32,
}

impl EmberPulse {
    #[must_use]
    pub fn new(strength: f32) -> Self {
        Self { strength }
    }

    pub fn apply(&self, frame: &mut Frame, shot_progress: f32) {
        let strength = self.strength.clamp(0.0, 1.0);
        if strength <= 0.0 {
            return;
        }
        let envelope = (shot_progress * std::f32::consts::PI).sin().max(0.0);
        let amp = strength * envelope;
        if amp <= 0.0 {
            return;
        }
        let width = frame.width;
        let height = frame.height;
        for y in 0..height {
            for x in 0..width {
                let u = (x as f32 + 0.5) / width as f32;
                let v = (y as f32 + 0.5) / height as f32;
                let r2 = (u - 0.5).powi(2) + (v - 0.5).powi(2);
                let glow = (-r2 * 14.0).exp() * amp;
                let p = ((y * width + x) * 4) as usize;
                let gains = [1.0_f32, 0.72, 0.35];
                for c in 0..3 {
                    let value = (frame.pixels[p + c] as f32 / 255.0 + glow * gains[c])
                        .clamp(0.0, 1.0);
                    frame.pixels[p + c] = (value * 255.0).round() as u8;
                }
            }
        }
    }
}

/// 几何砸字底块：scale 2.4→1.0 slam、双 smoothstep 环 mask、色 (0.98, 0.82, 0.55)。
///
/// 只画几何底块；字形栅格归 `media.text`。`shot_progress` 同上。
pub struct SlamStamp {
    /// 0..1 强度，0 恒等。
    pub strength: f32,
}

impl SlamStamp {
    #[must_use]
    pub fn new(strength: f32) -> Self {
        Self { strength }
    }

    pub fn apply(&self, frame: &mut Frame, shot_progress: f32) {
        let strength = self.strength.clamp(0.0, 1.0);
        if strength <= 0.0 {
            return;
        }
        let slam = smoothstep(0.0, 0.22, shot_progress);
        let scale = 2.4 - 1.4 * slam;
        let opacity =
            strength * slam * (0.75 + 0.25 * (shot_progress * std::f32::consts::PI).sin().abs());
        if opacity <= 0.0 {
            return;
        }
        let width = frame.width;
        let height = frame.height;
        let aspect = width as f32 / height as f32;
        for y in 0..height {
            for x in 0..width {
                let u = ((x as f32 + 0.5) / width as f32 - 0.5) * aspect;
                let v = (y as f32 + 0.5) / height as f32 - 0.5;
                let px = u * scale;
                let py = v * scale;
                let d = px.abs().max(py.abs());
                let outer = smoothstep(0.22, 0.18, d) * smoothstep(0.12, 0.14, d);
                let inner = smoothstep(0.10, 0.06, d);
                let mask = (outer * 0.85 + inner * 0.55).clamp(0.0, 1.0) * opacity;
                if mask <= 0.0 {
                    continue;
                }
                let p = ((y * width + x) * 4) as usize;
                let targets = [0.98_f32, 0.82, 0.55];
                for c in 0..3 {
                    let value = (frame.pixels[p + c] as f32 / 255.0 * (1.0 - mask)
                        + targets[c] * mask)
                        .clamp(0.0, 1.0);
                    frame.pixels[p + c] = (value * 255.0).round() as u8;
                }
            }
        }
    }
}

/// 渐变映射 LUT：256 格 luma→色彩查找表，显示空间调色（非线性物理）。
///
/// 停点表内容（品牌/创作主题）属于产品层；本算子只拥有「停点→条带→查表
/// 混合」的通用语义。条带构造：[`GradeMap::identity`]、[`GradeMap::from_stops`]、
/// [`GradeMap::from_strip`]（任意宽高 PNG 条带的中缝重采样）。
#[derive(Clone, Debug)]
pub struct GradeMap {
    /// 256 项 RGB 查找表。
    table: [[u8; 3]; 256],
}

impl GradeMap {
    /// 恒等映射（luma 直通）。
    #[must_use]
    pub fn identity() -> Self {
        let mut table = [[0u8; 3]; 256];
        for (index, entry) in table.iter_mut().enumerate() {
            let v = index as u8;
            *entry = [v, v, v];
        }
        Self { table }
    }

    /// 由停点表构造：分段线性插值到 256 格（停点须按 t 升序）。
    #[must_use]
    pub fn from_stops(stops: &[(f32, [u8; 3])]) -> Self {
        let mut table = [[0u8; 3]; 256];
        for (index, entry) in table.iter_mut().enumerate() {
            *entry = sample_stops(stops, index as f32 / 255.0);
        }
        Self { table }
    }

    /// 由解码条带构造：取中缝行重采样到 256 格（源 `resample_to_strip`）。
    #[must_use]
    pub fn from_strip(rgba: &[u8], width: u32, height: u32) -> Self {
        if width == 256 && height == 1 {
            let mut table = [[0u8; 3]; 256];
            for (index, entry) in table.iter_mut().enumerate() {
                let base = index * 4;
                *entry = [rgba[base], rgba[base + 1], rgba[base + 2]];
            }
            return Self { table };
        }
        let src_w = width.max(1) as usize;
        let y = (height / 2).min(height.saturating_sub(1)) as usize;
        let mut table = [[0u8; 3]; 256];
        for (index, entry) in table.iter_mut().enumerate() {
            let sx = (((index as f32 / 255.0) * (src_w - 1) as f32).round() as usize)
                .min(src_w - 1);
            let base = (y * src_w + sx) * 4;
            *entry = [rgba[base], rgba[base + 1], rgba[base + 2]];
        }
        Self { table }
    }

    /// 256 格查找表（供 GPU 条带上传等消费者读取）。
    #[must_use]
    pub fn table(&self) -> &[[u8; 3]; 256] {
        &self.table
    }

    /// Rec.709 luma 查表并按 `mix` 与源色线性混合（u8 域）。
    pub fn apply(&self, frame: &mut Frame, mix: f32) {
        let mix = mix.clamp(0.0, 1.0);
        if mix <= 0.0 {
            return;
        }
        for chunk in frame.pixels.chunks_exact_mut(4) {
            let r = chunk[0] as f32 / 255.0;
            let g = chunk[1] as f32 / 255.0;
            let b = chunk[2] as f32 / 255.0;
            let luma = (0.2126 * r + 0.7152 * g + 0.0722 * b).clamp(0.0, 1.0);
            let idx = (luma * 255.0).round() as usize;
            let graded = &self.table[idx.min(255)];
            for c in 0..3 {
                chunk[c] =
                    (chunk[c] as f32 * (1.0 - mix) + graded[c] as f32 * mix).round() as u8;
            }
        }
    }
}

fn sample_stops(stops: &[(f32, [u8; 3])], t: f32) -> [u8; 3] {
    let t = t.clamp(0.0, 1.0);
    if stops.is_empty() {
        return [0, 0, 0];
    }
    if t <= stops[0].0 {
        return stops[0].1;
    }
    for window in stops.windows(2) {
        let (t0, c0) = window[0];
        let (t1, c1) = window[1];
        if t <= t1 {
            let u = if (t1 - t0).abs() < f32::EPSILON {
                0.0
            } else {
                (t - t0) / (t1 - t0)
            };
            return [
                lerp_u8(c0[0], c1[0], u),
                lerp_u8(c0[1], c1[1], u),
                lerp_u8(c0[2], c1[2], u),
            ];
        }
    }
    stops[stops.len() - 1].1
}

fn lerp_u8(a: u8, b: u8, t: f32) -> u8 {
    (a as f32 + (b as f32 - a as f32) * t).round() as u8
}

/// 转场混合：线性交叉淡化（`xfade` 的正式实现）。
#[must_use]
pub fn crossfade(from: &Frame, to: &Frame, mix: f32) -> Frame {
    let mix = mix.clamp(0.0, 1.0);
    let width = from.width.min(to.width);
    let height = from.height.min(to.height);
    let mut out = Frame::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let i = ((y * width + x) * 4) as usize;
            for c in 0..4 {
                let a = from.pixels[i + c] as f32;
                let b = to.pixels[i + c] as f32;
                out.pixels[i + c] = (a * (1.0 - mix) + b * mix).round() as u8;
            }
        }
    }
    out
}

/// 转场混合：圆形 iris 开窗，当前画面自中心展开覆盖前一画面（羽化边缘）。
#[must_use]
pub fn iris_wipe(from: &Frame, to: &Frame, mix: f32) -> Frame {
    let mix = mix.clamp(0.0, 1.0);
    let width = from.width.min(to.width);
    let height = from.height.min(to.height);
    let cx = width as f32 * 0.5;
    let cy = height as f32 * 0.5;
    let max_r = (cx * cx + cy * cy).sqrt().max(1.0);
    let open = max_r * smoothstep(0.0, 1.0, mix);
    let feather = (max_r * 0.05).max(1.5);
    let mut out = Frame::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let dx = x as f32 + 0.5 - cx;
            let dy = y as f32 + 0.5 - cy;
            let d = (dx * dx + dy * dy).sqrt();
            let raw = 1.0 - ((d - (open - feather)) / (2.0 * feather)).clamp(0.0, 1.0);
            let mask = smoothstep(0.0, 1.0, raw.clamp(0.0, 1.0));
            let i = ((y * width + x) * 4) as usize;
            for c in 0..4 {
                let a = from.pixels[i + c] as f32;
                let b = to.pixels[i + c] as f32;
                out.pixels[i + c] = (a * (1.0 - mask) + b * mask).round() as u8;
            }
        }
    }
    out
}

/// 转场混合：8 条带错位擦除，自上而下逐带延迟。
#[must_use]
pub fn slices_wipe(from: &Frame, to: &Frame, mix: f32) -> Frame {
    let mix = mix.clamp(0.0, 1.0);
    let eased = smoothstep(0.0, 1.0, mix);
    let width = from.width.min(to.width);
    let height = from.height.min(to.height);
    let bands = 8.0_f32;
    let stagger = 0.45_f32;
    let span = (1.0 - stagger).max(0.15);
    let feather = 0.035_f32;
    let mut out = Frame::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let u = (x as f32 + 0.5) / width as f32;
            let v = (y as f32 + 0.5) / height as f32;
            let band = (v * bands).floor().clamp(0.0, bands - 1.0);
            let delay = band / bands * stagger;
            let local = ((eased - delay) / span).clamp(0.0, 1.0);
            let edge0 = local - feather;
            let edge1 = local + feather;
            let raw = (1.0 - ((u - edge0) / (edge1 - edge0).max(1e-4)).clamp(0.0, 1.0)).clamp(0.0, 1.0);
            let mask = smoothstep(0.0, 1.0, raw);
            let i = ((y * width + x) * 4) as usize;
            for c in 0..4 {
                let a = from.pixels[i + c] as f32;
                let b = to.pixels[i + c] as f32;
                out.pixels[i + c] = (a * (1.0 - mask) + b * mask).round() as u8;
            }
        }
    }
    out
}

/// 转场闪光：向纯白加算混合（就地）。
pub fn flash(frame: &mut Frame, amount: f32) {
    let amount = amount.clamp(0.0, 1.0);
    if amount <= 0.0 {
        return;
    }
    for chunk in frame.pixels.chunks_exact_mut(4) {
        for c in 0..3 {
            chunk[c] =
                ((chunk[c] as f32 * (1.0 - amount) + 255.0 * amount).round() as u8).min(255);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame_with(fill: [u8; 4], width: u32, height: u32) -> Frame {
        let mut frame = Frame::new(width, height);
        for chunk in frame.pixels.chunks_exact_mut(4) {
            chunk.copy_from_slice(&fill);
        }
        frame
    }

    fn fnv1a64(bytes: &[u8]) -> u64 {
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        for byte in bytes {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        hash
    }

    #[test]
    fn hash21_is_deterministic_and_bounded() {
        let a = hash21(0.5, 0.25);
        let b = hash21(0.5, 0.25);
        assert_eq!(a.to_bits(), b.to_bits());
        assert!((0.0..1.0).contains(&a), "{a}");
        assert_ne!(hash21(0.5, 0.25).to_bits(), hash21(0.5, 0.26).to_bits());
        // 固定抽查值钉死合同（WGSL 复现同序运算后逐位一致）。
        assert_eq!(hash21(1024.5, 53.0).to_bits(), 1041957960);
    }

    #[test]
    fn soft_bloom_zero_is_identity_and_threshold_gates() {
        let mut frame = frame_with([100, 100, 100, 255], 8, 8);
        SoftBloom::new(0.0).apply(&mut frame);
        assert_eq!(&frame.pixels[0..4], &[100, 100, 100, 255]);
        // 全帧低于阈值 0.62（158/255）：无能量可提取，输出不变。
        SoftBloom::new(1.0).apply(&mut frame);
        assert_eq!(&frame.pixels[0..4], &[100, 100, 100, 255]);
        // 高于阈值的白帧被提亮并截断到 255。
        let mut bright = frame_with([250, 250, 250, 255], 8, 8);
        SoftBloom::new(1.0).apply(&mut bright);
        assert_eq!(&bright.pixels[0..4], &[255, 255, 255, 255]);
    }

    #[test]
    fn vignette_darkens_edges_only() {
        let mut frame = frame_with([200, 200, 200, 255], 64, 64);
        Vignette::new(0.8).apply(&mut frame);
        let center = ((32 * 64 + 32) * 4) as usize;
        let corner = 0usize;
        assert_eq!(frame.pixels[center], 200, "center inside 0.25 radius stays");
        assert!(frame.pixels[corner] < 200, "corner darkened");
        assert_eq!(frame.pixels[center + 3], 255, "alpha untouched");
        // golden：整帧 digest 钉死公式。
        assert_eq!(fnv1a64(&frame.pixels), 8946696865805919733);
    }

    #[test]
    fn film_grain_is_time_reproducible() {
        let mut a = frame_with([128, 128, 128, 255], 32, 32);
        let mut b = frame_with([128, 128, 128, 255], 32, 32);
        FilmGrain::new(0.2).apply(&mut a, 0.5);
        FilmGrain::new(0.2).apply(&mut b, 0.5);
        assert_eq!(a.pixels, b.pixels, "same time reproduces");
        let mut c = frame_with([128, 128, 128, 255], 32, 32);
        FilmGrain::new(0.2).apply(&mut c, 0.6);
        assert_ne!(a.pixels, c.pixels, "different time shifts the pattern");
        assert_eq!(fnv1a64(&a.pixels), 9761959676559752531);
    }

    #[test]
    fn ember_pulse_envelope_peaks_mid_progress() {
        let mut frame = frame_with([0, 0, 0, 255], 32, 32);
        EmberPulse::new(1.0).apply(&mut frame, 0.5);
        let center = ((16 * 32 + 16) * 4) as usize;
        assert!(frame.pixels[center] > 200, "ember core near white");
        assert!(frame.pixels[center + 1] > 100 && frame.pixels[center + 2] < 100, "warm tint");
        let mut edge = frame_with([0, 0, 0, 255], 32, 32);
        EmberPulse::new(1.0).apply(&mut edge, 0.0);
        assert_eq!(edge.pixels[0], 0, "zero progress envelope is dark");
    }

    #[test]
    fn slam_stamp_ring_geometry_is_stable() {
        let mut frame = frame_with([0, 0, 0, 255], 64, 64);
        SlamStamp::new(1.0).apply(&mut frame, 0.5);
        // 中心内环被染成暖色，环外保持黑。
        let center = ((32 * 64 + 32) * 4) as usize;
        assert!(frame.pixels[center] > 100);
        assert!(frame.pixels[center] > frame.pixels[center + 2], "r > b warmth");
        let corner = 0usize;
        assert_eq!(frame.pixels[corner], 0, "outside the rings stays black");
        assert_eq!(fnv1a64(&frame.pixels), 3008134784848625757);
    }

    #[test]
    fn grade_map_constructors_and_apply() {
        let identity = GradeMap::identity();
        let mut frame = frame_with([40, 90, 200, 255], 4, 4);
        let before = frame.pixels.clone();
        // mix=0 恒等：早退不改像素。
        identity.apply(&mut frame, 0.0);
        assert_eq!(frame.pixels, before);
        // mix=1 恒等表：彩色被替换为 Rec.709 luma 灰度。
        identity.apply(&mut frame, 1.0);
        assert_eq!(&frame.pixels[0..4], &[87, 87, 87, 255]);
        // 双色停点：t=0 黑、t=1 白。
        let bw = GradeMap::from_stops(&[(0.0, [0, 0, 0]), (1.0, [255, 255, 255])]);
        assert_eq!(bw.table()[0], [0, 0, 0]);
        assert_eq!(bw.table()[255], [255, 255, 255]);
        assert_eq!(bw.table()[128], [128, 128, 128]);
        // 条带重采样：512 宽取中缝映射到 256 格。
        let mut strip = vec![0u8; 512 * 2 * 4];
        for x in 0..512usize {
            let v = (x / 2) as u8;
            let base = (1 * 512 + x) * 4;
            strip[base] = v;
            strip[base + 1] = v;
            strip[base + 2] = v;
            strip[base + 3] = 255;
        }
        let map = GradeMap::from_strip(&strip, 512, 2);
        assert_eq!(map.table()[0], [0, 0, 0]);
        assert_eq!(map.table()[255], [255, 255, 255]);
    }

    #[test]
    fn transition_blends_match_reference_digests() {
        let from = frame_with([0, 0, 0, 255], 16, 16);
        let to = frame_with([200, 100, 50, 255], 16, 16);
        let faded = crossfade(&from, &to, 0.5);
        assert_eq!(&faded.pixels[0..4], &[100, 50, 25, 255]);
        let iris = iris_wipe(&from, &to, 0.5);
        let center = ((8 * 16 + 8) * 4) as usize;
        assert_eq!(&iris.pixels[center..center + 4], &[200, 100, 50, 255], "center opened");
        let corner = 0usize;
        assert_eq!(&iris.pixels[corner..corner + 4], &[0, 0, 0, 255], "corner still previous");
        let slices = slices_wipe(&from, &to, 1.0);
        // 条带左缘完全擦入（羽化只影响带内右缘，几何与源实现逐值一致）。
        assert_eq!(&slices.pixels[0..4], &[200, 100, 50, 255], "left edge wiped in");
        let mut flashed = frame_with([100, 100, 100, 255], 4, 4);
        flash(&mut flashed, 0.5);
        assert_eq!(&flashed.pixels[0..4], &[178, 178, 178, 255]);
        // golden：整帧 digest 钉死转场几何。
        assert_eq!(fnv1a64(&iris.pixels), 10698823409230758749);
        assert_eq!(fnv1a64(&slices_wipe(&from, &to, 0.4).pixels), 14724886852761513233);
    }
}
