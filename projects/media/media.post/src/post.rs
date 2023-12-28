// 源自 TON blackhole 渲染器（MIT, © TON Team），按 MPL-2.0 分发。

//! HDR 后处理：多尺度泛光与色调映射（UCF `ton_bloom` / `ton_tone_map` CPU 参考）。

use geometry_types::LinearRgb;
use media_image::{Framebuffer, FramebufferEncoding};

/// 默认曝光乘子（线性 HDR，在色调映射之前应用）。
pub const DEFAULT_EXPOSURE: f64 = 0.82;

/// 默认泛光阈值：线性 HDR 下高于显示白（1.0）的能量才参与泛光。
///
/// 这是 CPU 与 UCF 双路径共同的唯一默认来源（合并前 CPU 硬编码 0.85、UCF
/// 携带 1.15 的不一致已消除）；UCF 帧图 `ton_bloom` 任务与该值一致。
pub const DEFAULT_BLOOM_THRESHOLD: f64 = 1.15;

/// 默认泛光强度，与 [`DEFAULT_BLOOM_THRESHOLD`] 同源。
pub const DEFAULT_BLOOM_INTENSITY: f64 = 0.18;

/// ACES 拟合色调映射（单通道）。
pub fn aces_filmic(x: f64) -> f64 {
    let x = x.max(0.0);
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    ((x * (a * x + b)) / (x * (c * x + d) + e)).clamp(0.0, 1.0)
}

/// 线性 RGB → 显示 RGB（UCF `ton_tone_map`）。
pub fn tone_map_rgb(color: LinearRgb) -> LinearRgb {
    LinearRgb::new(
        aces_filmic(color.r),
        aces_filmic(color.g),
        aces_filmic(color.b),
    )
}

fn luminance(c: LinearRgb) -> f64 {
    0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b
}

/// 阈值亮部提取（native 参考，与 UCF `post_bloom_extract` 语义一致）。
pub fn extract_bright_pass(pixels: &[LinearRgb], threshold: f64) -> Vec<LinearRgb> {
    pixels
        .iter()
        .map(|px| {
            let l = luminance(*px);
            if l > threshold {
                let scale = (l - threshold) / l.max(1.0e-9);
                px.scale(scale)
            } else {
                LinearRgb::BLACK
            }
        })
        .collect()
}

/// 将已提取的亮部场经多尺度模糊后线性加回帧缓冲。
pub fn composite_bloom_from_bright(fb: &mut Framebuffer, bright: &[LinearRgb], intensity: f64) {
    let bloom = multi_scale_bloom(bright, fb.width, fb.height);
    for (dst, add) in fb.pixels.iter_mut().zip(bloom.iter()) {
        *dst = dst.add(add.scale(intensity));
    }
}

/// 多尺度 HDR 泛光：阈值提取 → 1×/2×/4× 盒模糊合成 → 线性加回（不截断）。
pub fn bloom_pass(fb: &mut Framebuffer, threshold: f64, intensity: f64) {
    let w = fb.width;
    let h = fb.height;
    if w == 0 || h == 0 {
        return;
    }
    let bright = extract_bright_pass(&fb.pixels, threshold);
    composite_bloom_from_bright(fb, &bright, intensity);
}

fn multi_scale_bloom(src: &[LinearRgb], width: u32, height: u32) -> Vec<LinearRgb> {
    let l0 = box_blur_3x3(src, width, height);
    let (w2, h2, down) = downsample_half(src, width, height);
    let l1 = if w2 > 0 && h2 > 0 {
        upsample_add(
            &box_blur_3x3(&down, w2, h2),
            w2,
            h2,
            width,
            height,
            0.55,
        )
    } else {
        vec![LinearRgb::BLACK; src.len()]
    };
    let (w4, h4, down2) = downsample_half(&down, w2, h2);
    let l2 = if w4 > 0 && h4 > 0 {
        let mid = upsample_add(&box_blur_3x3(&down2, w4, h4), w4, h4, w2, h2, 1.0);
        upsample_add(&mid, w2, h2, width, height, 0.35)
    } else {
        vec![LinearRgb::BLACK; src.len()]
    };
    src.iter()
        .zip(l0.iter())
        .zip(l1.iter())
        .zip(l2.iter())
        .map(|(((base, a), b), c)| base.add(*a).add(*b).add(*c))
        .collect()
}

fn downsample_half(src: &[LinearRgb], width: u32, height: u32) -> (u32, u32, Vec<LinearRgb>) {
    let w2 = width / 2;
    let h2 = height / 2;
    if w2 == 0 || h2 == 0 {
        return (0, 0, Vec::new());
    }
    let mut out = Vec::with_capacity((w2 * h2) as usize);
    for y in 0..h2 {
        for x in 0..w2 {
            let sx = x * 2;
            let sy = y * 2;
            let i00 = sy as usize * width as usize + sx as usize;
            let i10 = i00 + 1;
            let i01 = i00 + width as usize;
            let i11 = i01 + 1;
            let sum = src[i00].add(src[i10]).add(src[i01]).add(src[i11]);
            out.push(sum.scale(0.25));
        }
    }
    (w2, h2, out)
}

fn upsample_add(
    src: &[LinearRgb],
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    weight: f64,
) -> Vec<LinearRgb> {
    let mut out = vec![LinearRgb::BLACK; (dst_w * dst_h) as usize];
    if src_w == 0 || src_h == 0 {
        return out;
    }
    for y in 0..dst_h {
        for x in 0..dst_w {
            let sx = (x * src_w / dst_w).min(src_w - 1);
            let sy = (y * src_h / dst_h).min(src_h - 1);
            let sample = src[sy as usize * src_w as usize + sx as usize];
            out[y as usize * dst_w as usize + x as usize] = sample.scale(weight);
        }
    }
    out
}

fn box_blur_3x3(src: &[LinearRgb], width: u32, height: u32) -> Vec<LinearRgb> {
    let w = width as usize;
    let h = height as usize;
    let mut out = vec![LinearRgb::BLACK; src.len()];
    for y in 0..h {
        for x in 0..w {
            let mut sum = LinearRgb::BLACK;
            let mut count = 0u32;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let nx = (x as i32 + dx).clamp(0, w as i32 - 1) as usize;
                    let ny = (y as i32 + dy).clamp(0, h as i32 - 1) as usize;
                    sum = sum.add(src[ny * w + nx]);
                    count += 1;
                }
            }
            let inv = 1.0 / count as f64;
            out[y * w + x] = LinearRgb::new(sum.r * inv, sum.g * inv, sum.b * inv);
        }
    }
    out
}

/// 对帧缓冲就地应用色调映射，并标记为显示空间。
pub fn apply_tone_map(fb: &mut Framebuffer) {
    for px in &mut fb.pixels {
        *px = tone_map_rgb(*px);
    }
    fb.encoding = FramebufferEncoding::Display;
}

/// 后处理链：线性 HDR 上泛光（默认参数见 [`DEFAULT_BLOOM_THRESHOLD`] 与
/// [`DEFAULT_BLOOM_INTENSITY`]），再一次色调映射。
pub fn apply_post_process(fb: &mut Framebuffer, bloom: bool) {
    if bloom {
        bloom_pass(fb, DEFAULT_BLOOM_THRESHOLD, DEFAULT_BLOOM_INTENSITY);
    }
    apply_tone_map(fb);
}

/// 编码为 RGBA8：线性 HDR 帧会先执行一次色调映射。
#[must_use]
pub fn to_rgba8(fb: &Framebuffer) -> Vec<u8> {
    let mut out = Vec::with_capacity(fb.pixels.len() * 4);
    for px in &fb.pixels {
        let display = match fb.encoding {
            FramebufferEncoding::LinearHdr => tone_map_rgb(*px),
            FramebufferEncoding::Display => *px,
        };
        let [r, g, b] = display.to_u8();
        out.extend_from_slice(&[r, g, b, 255]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tone_map_clamps_hot_pixel() {
        let mapped = tone_map_rgb(LinearRgb::new(4.0, 2.0, 0.5));
        assert!(mapped.r <= 1.0 && mapped.g <= 1.0 && mapped.b <= 1.0);
        assert!(mapped.r > 0.9);
    }

    #[test]
    fn bloom_spreads_energy_to_neighbors_without_early_clamp() {
        let mut fb = Framebuffer::new(64, 64);
        fb.set(32, 32, LinearRgb::new(4.0, 2.0, 0.5));
        bloom_pass(&mut fb, 0.5, 0.5);
        let center = fb.pixels[32 * 64 + 32];
        let neighbor = fb.pixels[32 * 64 + 28];
        assert!(neighbor.r > 0.0);
        assert!(center.r > 1.0);
    }

    #[test]
    fn multi_scale_bloom_produces_nonzero_blur_field() {
        let src = vec![LinearRgb::BLACK; 64 * 64];
        let mut bright = src.clone();
        bright[32 * 64 + 32] = LinearRgb::new(3.0, 2.0, 1.0);
        let blurred = multi_scale_bloom(&bright, 64, 64);
        assert!(blurred.iter().any(|p| p.r > 0.0));
    }
}
