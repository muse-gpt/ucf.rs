//! 显示空间可分离盒模糊的唯一正式实现。
//!
//! 吸收自两条下游线路并逐值一致：显影配方的单次横纵两趟（短边 ×0.03）与特效栈
//! blur 的三轮横纵（短边 ×0.06）。二者共享同一单趟语义——clamp-to-edge、
//! u32 累加、对窗口跨度整除取均值、只动 RGB 三个通道、straight alpha 直通、
//! 每趟读旧快照写新帧。线性 HDR 的 3×3 金字塔盒是另一域的另一算子，不在此处。

use media_image::Frame;

/// 对 RGBA8 帧就地执行 `passes` 轮可分离盒模糊，每轮先横后纵各一趟。
///
/// `radius <= 0` 或 `passes == 0` 时是恒等直通。像素半径的换算与钳制由调用算子
/// 通过 [`crate::relative_radius`] 完成，本函数只消费像素半径。
pub fn display_box_blur(frame: &mut Frame, radius: i32, passes: u32) {
    if radius <= 0 || passes == 0 {
        return;
    }
    for _ in 0..passes {
        blur_pass(frame, radius, true);
        blur_pass(frame, radius, false);
    }
}

/// 单趟可分离盒模糊：`horizontal` 为真时沿行、否则沿列。
fn blur_pass(frame: &mut Frame, radius: i32, horizontal: bool) {
    let width = frame.width as i32;
    let height = frame.height as i32;
    let source = frame.pixels.clone();
    let span = radius * 2 + 1;
    for y in 0..height {
        for x in 0..width {
            let mut acc = [0u32; 3];
            for offset in -radius..=radius {
                let (sx, sy) = if horizontal {
                    ((x + offset).clamp(0, width - 1), y)
                } else {
                    (x, (y + offset).clamp(0, height - 1))
                };
                let index = ((sy * width + sx) * 4) as usize;
                for k in 0..3 {
                    acc[k] += source[index + k] as u32;
                }
            }
            let index = ((y * width + x) * 4) as usize;
            for k in 0..3 {
                frame.pixels[index + k] = (acc[k] / span as u32) as u8;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dark_frame(width: u32, height: u32) -> Frame {
        let mut frame = Frame::new(width, height);
        for chunk in frame.pixels.chunks_exact_mut(4) {
            chunk[3] = 255;
        }
        frame
    }

    fn set(frame: &mut Frame, x: u32, y: u32, rgb: [u8; 3]) {
        let i = ((y * frame.width + x) as usize) * 4;
        frame.pixels[i..i + 3].copy_from_slice(&rgb);
    }

    fn get(frame: &Frame, x: u32, y: u32) -> [u8; 4] {
        let i = ((y * frame.width + x) as usize) * 4;
        frame.pixels[i..i + 4].try_into().unwrap()
    }

    #[test]
    fn single_pass_spreads_a_hot_pixel_deterministically() {
        // Golden of the absorbed develop recipe blur: one 255 pixel in a dark
        // field, radius 1, one H+V pass. After H the row reads 85,85,85; the V
        // pass then leaves 255/3/3 = 28 at the centre and along its cross.
        let mut frame = dark_frame(100, 100);
        set(&mut frame, 50, 50, [255, 255, 255]);
        display_box_blur(&mut frame, 1, 1);
        assert_eq!(&get(&frame, 50, 50)[0..3], &[28, 28, 28]);
        assert_eq!(&get(&frame, 49, 50)[0..3], &[28, 28, 28]);
        assert_eq!(&get(&frame, 50, 51)[0..3], &[28, 28, 28]);
        assert_eq!(&get(&frame, 48, 50)[0..3], &[0, 0, 0], "energy stays inside the window");
    }

    #[test]
    fn edges_sample_clamp_to_the_border() {
        // Corner energy is re-read at the border, never wrapped: the H pass
        // turns (0,0)=255 into [170, 85, 0, 0] on row 0, and the V pass then
        // gives (0,0) = (170 + 170 + 0) / 3 = 113.
        let mut frame = dark_frame(4, 4);
        set(&mut frame, 0, 0, [255, 255, 255]);
        display_box_blur(&mut frame, 1, 1);
        assert_eq!(&get(&frame, 0, 0)[0..3], &[113, 113, 113]);
    }

    #[test]
    fn alpha_is_never_touched() {
        let mut frame = Frame::new(8, 8);
        for chunk in frame.pixels.chunks_exact_mut(4) {
            chunk[3] = 7;
        }
        set(&mut frame, 4, 4, [200, 100, 50]);
        display_box_blur(&mut frame, 2, 3);
        for i in (0..frame.pixels.len()).step_by(4) {
            assert_eq!(frame.pixels[i + 3], 7, "straight alpha passes through");
        }
    }

    #[test]
    fn zero_radius_or_passes_is_the_identity() {
        let mut frame = dark_frame(4, 4);
        set(&mut frame, 1, 1, [255, 0, 0]);
        let before = frame.pixels.clone();
        display_box_blur(&mut frame, 0, 3);
        display_box_blur(&mut frame, 2, 0);
        assert_eq!(frame.pixels, before);
    }

    #[test]
    fn three_rounds_approximate_a_gaussian_spread() {
        // The absorbed stack blur contract: repeated H+V rounds widen the
        // footprint beyond a single pass while keeping the peak lower.
        let mut single = dark_frame(64, 64);
        let mut triple = dark_frame(64, 64);
        for frame in [&mut single, &mut triple] {
            set(frame, 32, 32, [255, 255, 255]);
        }
        display_box_blur(&mut single, 2, 1);
        display_box_blur(&mut triple, 2, 3);
        let peak = |frame: &Frame| u32::from(frame.pixels[((32 * 64 + 32) as usize) * 4]);
        assert!(peak(&triple) < peak(&single), "more rounds lower the peak");
        assert!(triple.pixels.iter().step_by(4).filter(|v| **v > 0).count() > single.pixels.iter().step_by(4).filter(|v| **v > 0).count());
    }
}
