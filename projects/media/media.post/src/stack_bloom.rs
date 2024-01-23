//! 效果栈的 bloom：显示域输入，线性 HDR 物理执行，成文的 EOTF 对转换边界。
//!
//! 这是栈 `bloom` kind 的唯一正式实现，合同成文于 [`crate::contract`]
//! 「栈 bloom 的域转换合同」：当前帧先经 sRGB EOTF 解码到线性光，再乘以
//! [`BLOOM_HDR_HEADROOM`] 上转换为可超过显示白的 HDR 能量，执行
//! [`crate::bloom_pass`]（阈值提取 + 多尺度金字塔 + 线性加回、无早截断），
//! 最后按逆序（除以同一 headroom、施加 sRGB OETF）回写显示字节。
//! 转换对是 EOTF 的互逆，不是 tone mapping；输出边界的单次 tone mapping
//! 合同不受影响。alpha 字节原样直通，回写的唯一截断是最终字节量化。

use geometry_types::LinearRgb;
use media_image::{Frame, Framebuffer};

use crate::contract::InputDomain;
use crate::effect::{Effect, EffectContext};
use crate::post::{bloom_pass, DEFAULT_BLOOM_INTENSITY, DEFAULT_BLOOM_THRESHOLD};

/// 显示白（线性 1.0）上转换到的 HDR 能量。
///
/// 2.0 使默认阈值 [`DEFAULT_BLOOM_THRESHOLD`]（1.15）落在显示线性亮度
/// 0.575（sRGB 字节约 200）之上：只有真高光参与泛光，阈值语义与
/// [`crate::bloom_pass`] 的 HDR 合同一致。
pub const BLOOM_HDR_HEADROOM: f64 = 2.0;

/// sRGB 编码值 → 线性光（IEC 61966-2-1 EOTF）。
fn srgb_eotf(encoded: f64) -> f64 {
    if encoded <= 0.04045 {
        encoded / 12.92
    } else {
        ((encoded + 0.055) / 1.055).powf(2.4)
    }
}

/// 线性光 → sRGB 编码值（上式的严格逆，即 OETF）。
fn srgb_oetf(linear: f64) -> f64 {
    if linear <= 0.003_130_8 {
        linear * 12.92
    } else {
        1.055 * linear.powf(1.0 / 2.4) - 0.055
    }
}

/// 显示字节 → 线性 HDR 上转换（EOTF 后乘 headroom）。
fn up_convert(byte: u8) -> f64 {
    srgb_eotf(f64::from(byte) / 255.0) * BLOOM_HDR_HEADROOM
}

/// 线性 HDR → 显示字节回写（除 headroom 后 OETF，唯一截断在字节量化）。
fn down_convert(linear: f64) -> u8 {
    let encoded = srgb_oetf(linear / BLOOM_HDR_HEADROOM).clamp(0.0, 1.0);
    (encoded * 255.0).round() as u8
}

/// 阈值亮部、多尺度金字塔、线性加回——物理 bloom 经显式 EOTF 对进出效果栈。
///
/// 没有半径参数：扩散尺度完全由金字塔层级合同表达，与
/// [`crate::bloom_pass`] 一致。
pub struct Bloom {
    /// 线性 HDR 亮度阈值；作者面默认 [`DEFAULT_BLOOM_THRESHOLD`]。
    pub threshold: f64,
    /// 加回强度；作者面默认 [`DEFAULT_BLOOM_INTENSITY`]。
    pub intensity: f64,
}

impl Bloom {
    #[must_use]
    pub fn new(threshold: f64, intensity: f64) -> Self {
        Self { threshold, intensity }
    }

    /// 物理路径的默认参数（与 CPU/UCF 双路径同源）。
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(DEFAULT_BLOOM_THRESHOLD, DEFAULT_BLOOM_INTENSITY)
    }
}

impl Effect for Bloom {
    fn name(&self) -> &'static str {
        "bloom"
    }

    fn input_domain(&self) -> InputDomain {
        InputDomain::DisplayRgba8
    }

    fn apply(&self, frame: &mut Frame, _ctx: &EffectContext) {
        let width = frame.width;
        let height = frame.height;
        if width == 0 || height == 0 {
            return;
        }
        let mut fb = Framebuffer::new(width, height);
        for index in 0..width as usize * height as usize {
            let i = index * 4;
            fb.pixels[index] = LinearRgb::new(
                up_convert(frame.pixels[i]),
                up_convert(frame.pixels[i + 1]),
                up_convert(frame.pixels[i + 2]),
            );
        }
        bloom_pass(&mut fb, self.threshold, self.intensity);
        for index in 0..width as usize * height as usize {
            let i = index * 4;
            let px = fb.pixels[index];
            frame.pixels[i] = down_convert(px.r);
            frame.pixels[i + 1] = down_convert(px.g);
            frame.pixels[i + 2] = down_convert(px.b);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use media_types::TimeBase;

    use crate::effect::EffectStack;

    fn ctx() -> EffectContext {
        EffectContext::new(0, TimeBase::from_fps(30))
    }

    fn black_frame(width: u32, height: u32) -> Frame {
        let mut frame = Frame::new(width, height);
        for chunk in frame.pixels.chunks_exact_mut(4) {
            chunk[3] = 255;
        }
        frame
    }

    #[test]
    fn the_eotf_pair_round_trips_every_byte_exactly() {
        // 转换对是严格互逆：无 bloom 能量时栈 bloom 是字节恒等。
        for byte in 0..=255u8 {
            assert_eq!(down_convert(up_convert(byte)), byte, "byte {byte}");
        }
    }

    #[test]
    fn stack_bloom_spreads_linear_energy_without_early_clamp() {
        // 物理路径特征：饱和高光在字节域仍是 255（不反向变暗），能量经
        // 金字塔多尺度扩散到 4px 外的邻域——显示域早截断实现做不到这一点。
        let mut frame = black_frame(64, 64);
        frame.pixels[(32 * 64 + 32) * 4..(32 * 64 + 32) * 4 + 3].copy_from_slice(&[255, 255, 255]);
        EffectStack::new().push(Bloom::with_defaults()).apply(&mut frame, &ctx());
        let center = (32 * 64 + 32) as usize * 4;
        assert_eq!(frame.pixels[center], 255, "saturated core stays saturated");
        for [dx, dy] in [[4, 0], [0, 4], [3, 2]] {
            let i = ((32 + dy) * 64 + (32 + dx)) as usize * 4;
            assert!(frame.pixels[i] > 0, "energy reaches ({dx},{dy})");
        }
        let far = (2 * 64 + 2) as usize * 4;
        assert_eq!(frame.pixels[far], 0, "the pyramid reach is bounded");
    }

    #[test]
    fn below_threshold_pixels_pass_through_unchanged() {
        // 灰 190：上转换亮度约 1.02 < 1.15，提取为零，字节恒等。
        // 旧显示域 bloom（threshold 0.45 × 255 = 115）会辉光这一帧——
        // 阈值语义的域判别证据。
        let mut frame = Frame::new(32, 32);
        for chunk in frame.pixels.chunks_exact_mut(4) {
            chunk[0] = 190;
            chunk[1] = 190;
            chunk[2] = 190;
            chunk[3] = 255;
        }
        let before = frame.pixels.clone();
        EffectStack::new().push(Bloom::with_defaults()).apply(&mut frame, &ctx());
        assert_eq!(frame.pixels, before);
    }

    #[test]
    fn threshold_response_is_a_soft_knee_not_a_binary_hard_copy() {
        // 均匀灰场的抬起量按超出阈值的比例增长：灰 190（上转换亮度约
        // 1.03 < 1.15）恒等，灰 210（约 1.29）轻抬，灰 230（约 1.58）抬得
        // 更多且仍远离硬截断的 255。显示域旧实现是「阈值上原值硬拷贝 +
        // 加回 min(255)」，给不出这种阶梯。
        let lift_of = |byte: u8| -> i32 {
            let mut frame = Frame::new(32, 32);
            for chunk in frame.pixels.chunks_exact_mut(4) {
                chunk[0] = byte;
                chunk[1] = byte;
                chunk[2] = byte;
                chunk[3] = 255;
            }
            EffectStack::new().push(Bloom::with_defaults()).apply(&mut frame, &ctx());
            i32::from(frame.pixels[0]) - i32::from(byte)
        };
        assert_eq!(lift_of(190), 0, "below the HDR threshold nothing glows");
        let warm = lift_of(210);
        let hot = lift_of(230);
        assert!(warm > 0, "just above the threshold glows a little");
        assert!(hot > warm, "the knee is proportional, got {warm} vs {hot}");
        assert!(i32::from(230) + hot < 255, "no early clamp to white");
    }

    #[test]
    fn alpha_is_never_touched() {
        let mut frame = Frame::new(8, 8);
        for chunk in frame.pixels.chunks_exact_mut(4) {
            chunk[0] = 255;
            chunk[3] = 128;
        }
        EffectStack::new().push(Bloom::with_defaults()).apply(&mut frame, &ctx());
        assert!(frame.pixels.chunks_exact(4).all(|c| c[3] == 128));
    }

    #[test]
    fn zero_intensity_is_identity() {
        let mut frame = black_frame(16, 16);
        frame.pixels[(8 * 16 + 8) * 4..(8 * 16 + 8) * 4 + 3].copy_from_slice(&[255, 255, 255]);
        let before = frame.pixels.clone();
        EffectStack::new().push(Bloom::new(DEFAULT_BLOOM_THRESHOLD, 0.0)).apply(&mut frame, &ctx());
        assert_eq!(frame.pixels, before);
    }
}
