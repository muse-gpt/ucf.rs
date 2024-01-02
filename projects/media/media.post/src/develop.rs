//! 显影配方：exposure、brightness、contrast、saturation、temperature 与 blur。
//!
//! 输入域是 [`InputDomain::DisplayRgba8`]（见 [`crate::contract`]）：配方按既有
//! 作者面在 8 位显示字节上工作，不冒充线性物理运算。配方是数值而不是滑杆历史，
//! 同一配方作用于同一帧必得同一像素——develop 步骤可复现、可测。
//!
//! 顺序即语义，固定不变：任一 tone 参数非零先跑 tone，`blur > 0` 再进模糊。
//! 模糊走 [`crate::display_box_blur`] 的唯一实现（单次横纵两趟），半径系数 0.03
//! 是该算子合同的一部分。

use media_image::Frame;

use crate::contract::{InputDomain, relative_radius};
use crate::display_blur::display_box_blur;

/// 显影模糊的短边系数（相对半径合同，见 [`crate::contract`]）。
pub const DEVELOP_BLUR_COEFFICIENT: f32 = 0.03;

/// 一次确定性的显影配方。
#[derive(Clone, Copy, Debug)]
pub struct Adjustments {
    /// 曝光档数（`2^stops`），-4..4。`0` 保持原帧。
    pub exposure: f32,
    /// 加性提升，-1..1，在曝光与对比之后应用。
    pub brightness: f32,
    /// -1 压向中灰，1 使距离加倍。
    pub contrast: f32,
    /// -1 到灰，+1 增强。
    pub saturation: f32,
    /// -1 冷蓝，+1 暖琥珀。
    pub temperature: f32,
    /// 0..1 相对盒模糊半径（短边 ×0.03）。
    pub blur: f32,
}

impl Default for Adjustments {
    fn default() -> Self {
        Self {
            exposure: 0.0,
            brightness: 0.0,
            contrast: 0.0,
            saturation: 0.0,
            temperature: 0.0,
            blur: 0.0,
        }
    }
}

impl Adjustments {
    /// 配方消费的输入域，恒为显示 RGBA8。
    #[must_use]
    pub fn input_domain(&self) -> InputDomain {
        InputDomain::DisplayRgba8
    }

    /// 按固定顺序就地应用配方。
    pub fn apply(&self, frame: &mut Frame) {
        if self.exposure != 0.0
            || self.brightness != 0.0
            || self.contrast != 0.0
            || self.saturation != 0.0
            || self.temperature != 0.0
        {
            tone(frame, self);
        }
        if self.blur > 0.0 {
            let radius = relative_radius(self.blur.clamp(0.0, 1.0), frame, DEVELOP_BLUR_COEFFICIENT);
            if radius > 0 {
                display_box_blur(frame, radius, 1);
            }
        }
    }
}

fn tone(frame: &mut Frame, recipe: &Adjustments) {
    let exposure = 2f32.powf(recipe.exposure.clamp(-4.0, 4.0));
    let brightness = recipe.brightness.clamp(-1.0, 1.0) * 0.5;
    let contrast = 1.0 + recipe.contrast.clamp(-1.0, 1.0);
    let saturation = 1.0 + recipe.saturation.clamp(-1.0, 1.0);
    let warm = recipe.temperature.clamp(-1.0, 1.0);
    for index in (0..frame.pixels.len()).step_by(4) {
        let mut rgb = [
            frame.pixels[index] as f32 / 255.0,
            frame.pixels[index + 1] as f32 / 255.0,
            frame.pixels[index + 2] as f32 / 255.0,
        ];
        for channel in &mut rgb {
            *channel = ((*channel * exposure - 0.5) * contrast + 0.5 + brightness).clamp(0.0, 1.0);
        }
        let luma = rgb[0] * 0.299 + rgb[1] * 0.587 + rgb[2] * 0.114;
        for channel in &mut rgb {
            *channel = (luma + (*channel - luma) * saturation).clamp(0.0, 1.0);
        }
        // Temperature moves red and blue in opposite directions.
        rgb[0] = (rgb[0] + warm * 0.06).clamp(0.0, 1.0);
        rgb[2] = (rgb[2] - warm * 0.06).clamp(0.0, 1.0);
        for k in 0..3 {
            frame.pixels[index + k] = (rgb[k] * 255.0).round() as u8;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grey(value: u8) -> Frame {
        let mut frame = Frame::new(4, 4);
        for chunk in frame.pixels.chunks_exact_mut(4) {
            chunk.copy_from_slice(&[value, value, value, 255]);
        }
        frame
    }

    #[test]
    fn brightness_lifts_the_mean() {
        let mut frame = grey(80);
        Adjustments {
            brightness: 0.25,
            ..Adjustments::default()
        }
        .apply(&mut frame);
        assert!(frame.pixels[0] > 80, "{}", frame.pixels[0]);
    }

    #[test]
    fn contrast_pushes_dark_pixels_lower() {
        let mut frame = grey(60);
        Adjustments {
            contrast: 0.6,
            ..Adjustments::default()
        }
        .apply(&mut frame);
        assert!(frame.pixels[0] < 60, "{}", frame.pixels[0]);
    }

    #[test]
    fn saturation_minus_one_produces_grey() {
        let mut frame = Frame::new(2, 2);
        for chunk in frame.pixels.chunks_exact_mut(4) {
            chunk.copy_from_slice(&[200, 60, 40, 255]);
        }
        Adjustments {
            saturation: -1.0,
            ..Adjustments::default()
        }
        .apply(&mut frame);
        assert_eq!(frame.pixels[0], frame.pixels[1]);
        assert_eq!(frame.pixels[1], frame.pixels[2]);
    }

    #[test]
    fn exposure_is_monotonic() {
        let mut dark = grey(30);
        let mut light = grey(200);
        let recipe = Adjustments {
            exposure: 0.4,
            ..Adjustments::default()
        };
        recipe.apply(&mut dark);
        recipe.apply(&mut light);
        assert!(light.pixels[0] > dark.pixels[0]);
    }

    #[test]
    fn the_tone_formula_is_pinned_to_its_absorbed_values() {
        // Golden of the exposed authoring surface: grey 80 with brightness
        // 0.25 is exactly 80/255 + 0.125 -> round(0.43872... * 255) = 112.
        let mut frame = grey(80);
        Adjustments {
            brightness: 0.25,
            ..Adjustments::default()
        }
        .apply(&mut frame);
        assert_eq!(frame.pixels[0], 112);
        // Warmth shifts red and blue by the pinned 0.06 constant.
        let mut frame = grey(128);
        Adjustments {
            temperature: 1.0,
            ..Adjustments::default()
        }
        .apply(&mut frame);
        assert!(frame.pixels[0] > 128 && frame.pixels[2] < 128);
    }

    #[test]
    fn the_identity_recipe_and_subpixel_blur_change_nothing() {
        let mut frame = grey(99);
        let before = frame.pixels.clone();
        Adjustments::default().apply(&mut frame);
        assert_eq!(frame.pixels, before);
        // blur 0.5 on a 100-wide field: radius = 0.5 * 100 * 0.03 = 1.5 -> 1,
        // and the cross golden matches `display_box_blur`'s single pass.
        let mut frame = Frame::new(100, 100);
        for chunk in frame.pixels.chunks_exact_mut(4) {
            chunk[3] = 255;
        }
        let hot = ((50 * 100 + 50) as usize) * 4;
        frame.pixels[hot..hot + 3].copy_from_slice(&[255, 255, 255]);
        Adjustments {
            blur: 0.5,
            ..Adjustments::default()
        }
        .apply(&mut frame);
        assert_eq!(&frame.pixels[hot..hot + 3], &[28, 28, 28]);
    }

    #[test]
    fn the_recipe_declares_the_display_domain() {
        assert_eq!(Adjustments::default().input_domain(), InputDomain::DisplayRgba8);
    }
}
