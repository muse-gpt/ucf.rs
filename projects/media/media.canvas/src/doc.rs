//! 位图文档：画布尺寸、背景与图层栈之上的图像放置。
//!
//! [`media_image::Frame`] 存像素，[`LayerStack`] 是图层像素、opacity、可见性与
//! 混合模式的唯一所有者；本文档只加画布尺寸、背景色与「把解码图像放到新层」
//! 的放置规则。图层模型在 Rust 侧完整可用，宿主作者面暴露多少由各产品决定，
//! 本 crate 不为任何宿主裁剪或扩张它。

use geometry::Rgba;
use media_image::Frame;

use crate::layer::LayerStack;

/// 一张位图文档：画布尺寸、背景与有序图层栈。
pub struct ImageDoc {
    pub width: u32,
    pub height: u32,
    pub background: Rgba,
    stack: LayerStack,
}

impl ImageDoc {
    /// 新建文档；尺寸至少 1×1（与 [`LayerStack`] 同一钳制）。
    #[must_use]
    pub fn new(width: u32, height: u32, background: Rgba) -> Self {
        let (width, height) = (width.max(1), height.max(1));
        Self {
            width,
            height,
            background,
            stack: LayerStack::new(width, height),
        }
    }

    /// 加一个透明层，返回索引。
    pub fn add_layer(&mut self, name: impl Into<String>) -> usize {
        self.stack.add_layer(name)
    }

    /// 把一张已解码图像放到新层：左上对齐、裁剪到画布与图像的交集、
    /// 不缩放，RGBA 四通道原样拷贝（源 alpha 直通）。
    pub fn add_image(&mut self, name: impl Into<String>, image: &Frame) -> usize {
        let index = self.stack.add_layer(name);
        let target = self
            .stack
            .layer_mut(index)
            .expect("the layer was just added");
        let width = target.width.min(image.width);
        let height = target.height.min(image.height);
        for y in 0..height {
            for x in 0..width {
                let from = ((y * image.width + x) * 4) as usize;
                let to = ((y * target.width + x) * 4) as usize;
                target.pixels[to..to + 4].copy_from_slice(&image.pixels[from..from + 4]);
            }
        }
        index
    }

    #[must_use]
    pub fn layer(&self, index: usize) -> Option<&crate::layer::Layer> {
        self.stack.layer(index)
    }

    pub fn layer_mut(&mut self, index: usize) -> Option<&mut Frame> {
        self.stack.layer_mut(index)
    }

    /// 图层本身，用于 opacity、可见性与混合模式。
    pub fn layer_entry(&mut self, index: usize) -> Option<&mut crate::layer::Layer> {
        self.stack.layer_entry(index)
    }

    /// 栈本身，供想自己合成的宿主使用。
    #[must_use]
    pub fn stack_mut(&mut self) -> &mut LayerStack {
        &mut self.stack
    }

    /// 把全部可见层合成到背景上。
    #[must_use]
    pub fn render(&self) -> Frame {
        self.stack.composite(self.background)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use media_raster::Paint;

    #[test]
    fn a_document_composites_its_layers() {
        let mut doc = ImageDoc::new(4, 4, Rgba([0, 0, 0, 255]));
        let base = doc.add_layer("base");
        doc.layer_mut(base).expect("frame").fill(Rgba([200, 0, 0, 255]));
        let overlay = doc.add_layer("overlay");
        doc.layer_mut(overlay)
            .expect("frame")
            .fill(Rgba([0, 0, 200, 255]));
        doc.layer_entry(overlay).expect("layer").opacity = 0.5;
        let frame = doc.render();
        assert!(frame.pixels[0] > 80 && frame.pixels[2] > 80, "{:?}", &frame.pixels[0..3]);
    }

    #[test]
    fn hidden_layers_do_not_reach_the_output() {
        let mut doc = ImageDoc::new(2, 2, Rgba([10, 10, 10, 255]));
        let layer = doc.add_layer("hidden");
        doc.layer_mut(layer).expect("frame").fill(Rgba([255, 255, 255, 255]));
        doc.layer_entry(layer).expect("layer").visible = false;
        let frame = doc.render();
        assert_eq!(&frame.pixels[0..3], &[10, 10, 10]);
    }

    #[test]
    fn add_image_places_top_left_cropped_and_unscaled() {
        // A 4x4 source onto a 3x2 canvas: rows 0..2, columns 0..3 survive;
        // the source's own alpha passes through untouched.
        let mut doc = ImageDoc::new(3, 2, Rgba([0, 0, 0, 255]));
        let mut source = Frame::new(4, 4);
        for y in 0..4u32 {
            for x in 0..4u32 {
                let i = ((y * 4 + x) as usize) * 4;
                source.pixels[i] = (x * 60) as u8;
                source.pixels[i + 1] = (y * 60) as u8;
                source.pixels[i + 2] = 7;
                source.pixels[i + 3] = 128;
            }
        }
        let index = doc.add_image("plate", &source);
        let layer = doc.layer(index).expect("layer");
        assert_eq!(layer.frame.width, 3);
        assert_eq!(layer.frame.height, 2);
        // (2,1): copied from source (2,1).
        let i = ((1 * 3 + 2) as usize) * 4;
        assert_eq!(&layer.frame.pixels[i..i + 4], &[120, 60, 7, 128]);
        // Rows beyond the canvas were dropped.
        assert_eq!(layer.frame.pixels.len(), 3 * 2 * 4);
    }

    #[test]
    fn dimensions_floor_to_one() {
        let doc = ImageDoc::new(0, 0, Rgba([0, 0, 0, 255]));
        assert_eq!((doc.width, doc.height), (1, 1));
        let frame = doc.render();
        assert_eq!((frame.width, frame.height), (1, 1));
    }

    #[test]
    fn the_background_reaches_pixels_no_layer_covers() {
        let mut doc = ImageDoc::new(2, 2, Rgba([5, 6, 7, 255]));
        let layer = doc.add_layer("partial");
        doc.layer_mut(layer).expect("frame").fill(Rgba([90, 0, 0, 255]));
        // Punch a hole by zeroing one pixel's alpha.
        let frame = {
            let target = doc.layer_mut(layer).expect("frame");
            target.pixels[3] = 0;
            doc.render()
        };
        assert_eq!(&frame.pixels[0..4], &[5, 6, 7, 255], "the hole shows the background");
        assert_eq!(&frame.pixels[4..8], &[90, 0, 0, 255], "the covered pixel takes the layer");
    }
}
