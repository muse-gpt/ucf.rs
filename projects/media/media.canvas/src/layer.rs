//! Layer stacks and blend modes.

use geometry::Rgba;
use media_image::Frame;
use media_raster::Paint;

/// How a layer combines with what is already there.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BlendMode {
    #[default]
    Normal,
    Multiply,
    Screen,
    Add,
}

/// One raster layer.
pub struct Layer {
    pub name: String,
    pub frame: Frame,
    /// 0..1.
    pub opacity: f32,
    pub blend: BlendMode,
    pub visible: bool,
}

impl Layer {
    #[must_use]
    pub fn new(name: impl Into<String>, width: u32, height: u32) -> Self {
        Self {
            name: name.into(),
            frame: Frame::new(width, height),
            opacity: 1.0,
            blend: BlendMode::Normal,
            visible: true,
        }
    }
}

/// An ordered stack of layers, bottom first.
pub struct LayerStack {
    pub width: u32,
    pub height: u32,
    layers: Vec<Layer>,
}

impl LayerStack {
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width: width.max(1),
            height: height.max(1),
            layers: Vec::new(),
        }
    }

    /// Appends a transparent layer and returns its index.
    pub fn add_layer(&mut self, name: impl Into<String>) -> usize {
        self.layers.push(Layer::new(name, self.width, self.height));
        self.layers.len() - 1
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    #[must_use]
    pub fn layer(&self, index: usize) -> Option<&Layer> {
        self.layers.get(index)
    }

    pub fn layer_mut(&mut self, index: usize) -> Option<&mut Frame> {
        self.layers.get_mut(index).map(|layer| &mut layer.frame)
    }

    pub fn layer_entry(&mut self, index: usize) -> Option<&mut Layer> {
        self.layers.get_mut(index)
    }

    /// Composites every visible layer bottom-up over `background`.
    #[must_use]
    pub fn composite(&self, background: Rgba) -> Frame {
        let mut out = Frame::new(self.width, self.height);
        out.fill(background);
        for layer in &self.layers {
            if !layer.visible || layer.opacity <= 0.001 {
                continue;
            }
            for i in (0..out.pixels.len()).step_by(4) {
                let alpha = (layer.frame.pixels[i + 3] as f32 / 255.0) * layer.opacity;
                if alpha <= 0.001 {
                    continue;
                }
                for k in 0..3 {
                    let src = layer.frame.pixels[i + k] as f32;
                    let dst = out.pixels[i + k] as f32;
                    let blended = match layer.blend {
                        BlendMode::Normal => src,
                        BlendMode::Multiply => src * dst / 255.0,
                        BlendMode::Screen => 255.0 - (255.0 - src) * (255.0 - dst) / 255.0,
                        BlendMode::Add => (src + dst).min(255.0),
                    };
                    out.pixels[i + k] = (dst + (blended - dst) * alpha).clamp(0.0, 255.0) as u8;
                }
                out.pixels[i + 3] = 255;
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn painted(width: u32, height: u32, color: Rgba, opacity: f32, blend: BlendMode) -> LayerStack {
        let mut stack = LayerStack::new(width, height);
        let index = stack.add_layer("paint");
        let frame = stack.layer_mut(index).expect("layer");
        frame.fill(color);
        let entry = stack.layer_entry(index).expect("layer");
        entry.opacity = opacity;
        entry.blend = blend;
        stack
    }

    #[test]
    fn a_single_layer_replaces_the_background() {
        let stack = painted(4, 4, Rgba([200, 40, 40, 255]), 1.0, BlendMode::Normal);
        let composed = stack.composite(Rgba([0, 0, 0, 255]));
        assert_eq!(&composed.pixels[0..3], &[200, 40, 40]);
    }

    #[test]
    fn opacity_mixes_toward_the_background() {
        let stack = painted(2, 2, Rgba([200, 200, 200, 255]), 0.5, BlendMode::Normal);
        let composed = stack.composite(Rgba([0, 0, 0, 255]));
        let value = composed.pixels[0];
        assert!((90..=110).contains(&value), "{value}");
    }

    #[test]
    fn multiply_darkens_and_screen_brightens() {
        let background = Rgba([128, 128, 128, 255]);
        let multiply = painted(2, 2, Rgba([128, 128, 128, 255]), 1.0, BlendMode::Multiply);
        let screened = painted(2, 2, Rgba([128, 128, 128, 255]), 1.0, BlendMode::Screen);
        let dark = multiply.composite(background).pixels[0];
        let light = screened.composite(background).pixels[0];
        assert!(dark < background.0[0], "{dark}");
        assert!(light > background.0[0], "{light}");
    }

    #[test]
    fn hidden_layers_are_skipped() {
        let mut stack = painted(2, 2, Rgba([255, 255, 255, 255]), 1.0, BlendMode::Normal);
        stack.layer_entry(0).expect("layer").visible = false;
        let composed = stack.composite(Rgba([10, 20, 30, 255]));
        assert_eq!(&composed.pixels[0..3], &[10, 20, 30]);
    }

    #[test]
    fn layer_order_is_bottom_first() {
        let mut stack = LayerStack::new(2, 2);
        let bottom = stack.add_layer("bottom");
        stack.layer_mut(bottom).expect("layer").fill(Rgba([255, 0, 0, 255]));
        let top = stack.add_layer("top");
        stack.layer_mut(top).expect("layer").fill(Rgba([0, 0, 255, 255]));
        let composed = stack.composite(Rgba([0, 0, 0, 255]));
        assert_eq!(&composed.pixels[0..3], &[0, 0, 255]);
    }
}
