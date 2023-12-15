// 源自 TON blackhole 渲染器（MIT, © TON Team），按 MPL-2.0 分发。

//! 线性 HDR 帧缓冲存储：像素容器与原始 clamp 转换。
//! 色调映射与泛光等处理语义归 `media.post`。

use geometry_types::LinearRgb;

/// 帧缓冲颜色空间：Beauty 输出为线性 HDR，色调映射后标记为显示空间。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FramebufferEncoding {
    /// 线性 HDR 辐射（可含 >1 能量）。
    #[default]
    LinearHdr,
    /// 已色调映射，可直接编码为 8 位显示。
    Display,
}

#[derive(Debug, Clone)]
pub struct Framebuffer {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<LinearRgb>,
    pub encoding: FramebufferEncoding,
}

impl Framebuffer {
    pub fn new(width: u32, height: u32) -> Self {
        let len = (width as usize) * (height as usize);
        Self {
            width,
            height,
            pixels: vec![LinearRgb::BLACK; len],
            encoding: FramebufferEncoding::LinearHdr,
        }
    }

    pub fn set(&mut self, x: u32, y: u32, color: LinearRgb) {
        if x < self.width && y < self.height {
            self.pixels[y as usize * self.width as usize + x as usize] = color;
        }
    }

    /// 线性 HDR 原始值（不色调映射），用于 HDR 诊断输出。
    #[must_use]
    pub fn to_rgba8_linear(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.pixels.len() * 4);
        for px in &self.pixels {
            let c = px.clamp01();
            out.extend_from_slice(&[
                (c.r * 255.0).round() as u8,
                (c.g * 255.0).round() as u8,
                (c.b * 255.0).round() as u8,
                255,
            ]);
        }
        out
    }
}
