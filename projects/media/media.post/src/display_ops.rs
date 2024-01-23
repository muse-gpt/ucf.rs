//! 显示空间艺术算子：blur、色差、位移、扫光、颗粒——共五个。
//!
//! 全部算子的输入域是 [`InputDomain::DisplayRgba8`]——8 位显示字节按合同
//! 直接参与算术（见 [`crate::contract`]），不冒充线性物理运算。数值语义
//! 逐值吸收自下游效果栈：相对短边的半径合同、clamp-to-edge、straight
//! alpha 直通与帧号派生的颗粒种子全部保持原样。
//!
//! bloom 不在本模块：效果栈的 `bloom` 一律绑定线性 HDR 物理路径
//! [`crate::bloom_pass`]（阈值提取 + 多尺度金字塔 + 无早截断），其显示域
//! 进出由 [`crate::stack_bloom`] 执行成文的 EOTF 对转换。禁止在此重新
//! 引入同名显示算子——同名双 bloom 已被裁决撤销。

use geometry_types::Rgba;
use media_image::Frame;

use crate::contract::{InputDomain, relative_radius, short_side};
use crate::display_blur::display_box_blur;
use crate::effect::{Effect, EffectContext};

/// 可分离盒模糊重复三轮（近似高斯）。半径 = amount × 短边 × 0.06。
pub struct Blur {
    /// 0 保持原帧，1 是重失焦。
    pub amount: f32,
}

impl Blur {
    #[must_use]
    pub fn new(amount: f32) -> Self {
        Self { amount }
    }
}

impl Effect for Blur {
    fn name(&self) -> &'static str {
        "blur"
    }

    fn input_domain(&self) -> InputDomain {
        InputDomain::DisplayRgba8
    }

    fn apply(&self, frame: &mut Frame, _ctx: &EffectContext) {
        let radius = relative_radius(self.amount.clamp(0.0, 1.0), frame, 0.06);
        if radius <= 0 {
            return;
        }
        display_box_blur(frame, radius, 3);
    }
}

/// 水平通道分裂。`amount` 是短边比例，0..0.1。
pub struct Chroma {
    pub amount: f32,
}

impl Chroma {
    #[must_use]
    pub fn new(amount: f32) -> Self {
        Self { amount }
    }
}

impl Effect for Chroma {
    fn name(&self) -> &'static str {
        "chroma"
    }

    fn input_domain(&self) -> InputDomain {
        InputDomain::DisplayRgba8
    }

    fn apply(&self, frame: &mut Frame, _ctx: &EffectContext) {
        let offset = relative_radius(self.amount.clamp(0.0, 0.1), frame, 1.0);
        if offset == 0 {
            return;
        }
        let source = frame.pixels.clone();
        let width = frame.width as i32;
        for y in 0..frame.height as i32 {
            for x in 0..width {
                let i = ((y * width + x) * 4) as usize;
                let red_x = (x + offset).clamp(0, width - 1);
                let blue_x = (x - offset).clamp(0, width - 1);
                let red = ((y * width + red_x) * 4) as usize;
                let blue = ((y * width + blue_x) * 4) as usize;
                frame.pixels[i] = source[red];
                frame.pixels[i + 2] = source[blue + 2];
            }
        }
    }
}

/// 沿行进正弦推移像素：热浪与液态玻璃。
pub struct Displace {
    /// 最大偏移，短边比例，0..0.05。
    pub amount: f32,
    /// 画面内波数。
    pub frequency: f32,
    /// 波速，每秒波数。
    pub speed: f32,
}

impl Displace {
    #[must_use]
    pub fn new(amount: f32, frequency: f32, speed: f32) -> Self {
        Self { amount, frequency, speed }
    }
}

impl Effect for Displace {
    fn name(&self) -> &'static str {
        "displace"
    }

    fn input_domain(&self) -> InputDomain {
        InputDomain::DisplayRgba8
    }

    fn apply(&self, frame: &mut Frame, ctx: &EffectContext) {
        let amplitude = self.amount.clamp(0.0, 0.05) * short_side(frame);
        if amplitude <= 0.5 {
            return;
        }
        let source = frame.pixels.clone();
        let width = frame.width as f32;
        let height = frame.height as f32;
        let phase = ctx.seconds() * self.speed * std::f32::consts::TAU;
        for y in 0..frame.height {
            for x in 0..frame.width {
                let u = x as f32 / width;
                let v = y as f32 / height;
                let dx = (v * self.frequency * std::f32::consts::TAU + phase).sin() * amplitude;
                let dy = (u * self.frequency * std::f32::consts::TAU + phase * 0.7).cos() * amplitude;
                let sx = (x as f32 + dx).clamp(0.0, width - 1.0) as u32;
                let sy = (y as f32 + dy).clamp(0.0, height - 1.0) as u32;
                let from = ((sy * frame.width + sx) * 4) as usize;
                let to = ((y * frame.width + x) * 4) as usize;
                frame.pixels[to..to + 4].copy_from_slice(&source[from..from + 4]);
            }
        }
    }
}

/// 一条以角度扫过画面的光带。
pub struct Sweep {
    /// 光带角度（度）。
    pub angle: f32,
    /// 每秒循环数。
    pub speed: f32,
    /// 光带宽度，短边比例，0.005..1。
    pub width: f32,
    pub color: Rgba,
    /// 加色强度，0..4。
    pub strength: f32,
}

impl Sweep {
    #[must_use]
    pub fn new(angle: f32, speed: f32, width: f32, color: Rgba, strength: f32) -> Self {
        Self { angle, speed, width, color, strength }
    }
}

impl Effect for Sweep {
    fn name(&self) -> &'static str {
        "sweep"
    }

    fn input_domain(&self) -> InputDomain {
        InputDomain::DisplayRgba8
    }

    fn apply(&self, frame: &mut Frame, ctx: &EffectContext) {
        let (sin, cos) = self.angle.to_radians().sin_cos();
        let width = self.width.clamp(0.005, 1.0) * short_side(frame);
        // 行程从一侧边缘外到另一侧边缘外。
        let travel = frame.width as f32 + frame.height as f32;
        let position = ctx.seconds() * self.speed * travel;
        let wrapped = position % (travel + width * 2.0) - width;
        for y in 0..frame.height {
            for x in 0..frame.width {
                let along = x as f32 * cos + y as f32 * sin;
                let distance = (along - wrapped).abs();
                if distance >= width {
                    continue;
                }
                let falloff = 1.0 - distance / width;
                let gain = falloff * falloff * self.strength.clamp(0.0, 4.0);
                let i = ((y * frame.width + x) * 4) as usize;
                for k in 0..3 {
                    let added = (self.color.0[k] as f32 * gain) as u32;
                    frame.pixels[i + k] = (frame.pixels[i + k] as u32 + added).min(255) as u8;
                }
            }
        }
    }
}

/// 确定性逐像素颗粒。种子派生自帧号——不是用户 seed：同一帧号两次渲染颗粒
/// 完全相同，不同帧号图案不同。这是与 particles/beams 用户 seed 分立的第二种
/// 随机合同，不得静默合并。
pub struct Grain {
    /// 强度，0..1。
    pub strength: f32,
    /// 每百万像素的采样数。
    pub density: f32,
}

impl Grain {
    #[must_use]
    pub fn new(strength: f32, density: f32) -> Self {
        Self { strength, density }
    }
}

impl Effect for Grain {
    fn name(&self) -> &'static str {
        "grain"
    }

    fn input_domain(&self) -> InputDomain {
        InputDomain::DisplayRgba8
    }

    fn apply(&self, frame: &mut Frame, ctx: &EffectContext) {
        let pixels = frame.width as usize * frame.height as usize;
        let count = ((pixels as f32 / 1_000_000.0) * self.density.max(0.0) * 1000.0) as u32;
        let seed = ctx.frame.wrapping_mul(2_654_435_761).wrapping_add(1);
        for i in 0..count {
            let h = media_types::hash(seed.wrapping_add(i));
            let x = h % frame.width;
            let y = (h / frame.width.max(1)) % frame.height;
            let delta = ((h >> 16) & 0xff) as f32 / 255.0 - 0.5;
            let lift = delta * self.strength.clamp(0.0, 1.0) * 255.0;
            let index = ((y * frame.width + x) * 4) as usize;
            for k in 0..3 {
                let value = frame.pixels[index + k] as f32 + lift;
                frame.pixels[index + k] = value.clamp(0.0, 255.0) as u8;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use media_types::TimeBase;

    fn ctx(frame: u32) -> EffectContext {
        EffectContext::new(frame, TimeBase::from_fps(30))
    }

    /// 带高频红通道细节的测试帧（不依赖 raster）。
    fn scene(width: u32, height: u32) -> Frame {
        let mut frame = Frame::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let i = ((y * width + x) as usize) * 4;
                let stripe = if (x / 2) % 2 == 0 { 200 } else { 20 };
                frame.pixels[i] = stripe;
                frame.pixels[i + 1] = 30;
                frame.pixels[i + 2] = 40;
                frame.pixels[i + 3] = 255;
            }
        }
        frame
    }

    fn red_stddev(frame: &Frame) -> f32 {
        let mut sum = 0.0f64;
        let mut sum_sq = 0.0f64;
        let count = (frame.width * frame.height) as f64;
        for chunk in frame.pixels.chunks(4) {
            let value = chunk[0] as f64;
            sum += value;
            sum_sq += value * value;
        }
        let mean = sum / count;
        (sum_sq / count - mean * mean).sqrt() as f32
    }

    fn red_mean(frame: &Frame) -> f32 {
        let total: u64 = frame.pixels.chunks(4).map(|c| c[0] as u64).sum();
        (total as f64 / (frame.width * frame.height) as f64) as f32
    }

    #[test]
    fn blur_removes_detail() {
        let mut frame = scene(96, 54);
        let before = red_stddev(&frame);
        Blur::new(1.0).apply(&mut frame, &ctx(0));
        let after = red_stddev(&frame);
        assert!(after < before * 0.9, "detail {before} -> {after}");
    }

    #[test]
    fn chroma_splits_channels_without_resizing() {
        // 条纹周期 4，offset = 0.02 × 54 截断为 1：x=1（亮 200）的 R 取 x=2（暗 20）。
        let mut frame = scene(96, 54);
        let before = frame.pixels[4];
        Chroma::new(0.02).apply(&mut frame, &ctx(0));
        assert_eq!((frame.width, frame.height), (96, 54));
        assert_eq!(frame.pixels.len(), 96 * 54 * 4);
        assert_eq!(before, 200);
        assert_eq!(frame.pixels[4], 20, "red samples x + offset");
        assert_eq!(frame.pixels[6], 40, "green is never moved");
    }

    #[test]
    fn displace_is_time_dependent_and_deterministic() {
        let mut a = scene(96, 54);
        let mut b = scene(96, 54);
        let op = Displace::new(0.05, 1.5, 1.5);
        op.apply(&mut a, &ctx(0));
        op.apply(&mut b, &ctx(0));
        assert_eq!(a.pixels, b.pixels, "same moment reproduces");
        let mut c = scene(96, 54);
        op.apply(&mut c, &ctx(15));
        assert_ne!(a.pixels, c.pixels, "phase moves with the moment");
    }

    #[test]
    fn sweep_brightens_where_the_band_passes() {
        let mut frame = scene(96, 54);
        let before = red_mean(&frame);
        Sweep::new(-18.0, 0.05, 0.42, Rgba([150, 190, 255, 255]), 0.09).apply(&mut frame, &ctx(0));
        assert!(red_mean(&frame) >= before, "additive light never darkens");
    }

    #[test]
    fn grain_seeds_from_the_frame_number_not_a_user_seed() {
        // 两种随机合同的 grain 半边：同帧号逐像素一致，异帧号不同。
        let mut a = scene(96, 54);
        let mut b = scene(96, 54);
        Grain::new(0.4, 2200.0).apply(&mut a, &ctx(7));
        Grain::new(0.4, 2200.0).apply(&mut b, &ctx(7));
        assert_eq!(a.pixels, b.pixels);
        let mut c = scene(96, 54);
        Grain::new(0.4, 2200.0).apply(&mut c, &ctx(8));
        assert_ne!(a.pixels, c.pixels);
    }

    #[test]
    fn display_operators_declare_their_domain() {
        let ops: Vec<Box<dyn Effect>> = vec![
            Box::new(Blur::new(0.5)),
            Box::new(Chroma::new(0.01)),
            Box::new(Displace::new(0.01, 2.0, 1.0)),
            Box::new(Sweep::new(0.0, 1.0, 0.5, Rgba([1, 2, 3, 255]), 1.0)),
            Box::new(Grain::new(0.2, 900.0)),
        ];
        assert_eq!(ops.iter().map(|op| op.name().to_string()).collect::<Vec<_>>(), vec!["blur", "chroma", "displace", "sweep", "grain"]);
        for op in &ops {
            assert_eq!(op.input_domain(), InputDomain::DisplayRgba8);
        }
    }
}
