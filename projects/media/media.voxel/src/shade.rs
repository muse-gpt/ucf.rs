//! Face brightness as an explicit preview parameter.
//!
//! The absorbed preview multiplied each channel by one of six fixed
//! constants chosen by face direction. Those constants are a display
//! preset, not a physical material or a light model, so they no longer
//! live on the mesh: a [`Shading`] strategy is passed in when the preview
//! plan is built, and [`Shading::FixedSixDir`] reproduces the preset
//! byte-for-byte.

use geometry::Rgba;

use crate::mesh::Direction;

/// How a face colour becomes a painted colour.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Shading {
    /// The absorbed preset: +Y `1.0`, -Y `0.45`, +X `0.72`, -X `0.62`,
    /// +Z `0.58`, -Z `0.66`. Named for what it is — six display constants
    /// keyed by direction — never presented as a lighting model.
    FixedSixDir,
    /// One multiplier for every direction.
    Uniform(f32),
}

impl Default for Shading {
    fn default() -> Self {
        Self::FixedSixDir
    }
}

impl Shading {
    /// The brightness multiplier for one direction.
    #[must_use]
    pub fn factor(self, direction: Direction) -> f32 {
        match self {
            Self::FixedSixDir => match direction {
                Direction::PosY => 1.0,
                Direction::NegY => 0.45,
                Direction::PosX => 0.72,
                Direction::NegX => 0.62,
                Direction::PosZ => 0.58,
                Direction::NegZ => 0.66,
            },
            Self::Uniform(value) => value,
        }
    }

    /// Paint one face: channel-wise multiply with truncating `u8`
    /// conversion and alpha forced to `255`. The forced alpha is the
    /// documented preview contract inherited from the source — the preview
    /// is an opaque picture; voxel alpha is not carried into it.
    #[must_use]
    pub fn paint(self, direction: Direction, color: Rgba) -> Rgba {
        let shade = self.factor(direction);
        Rgba([(color.0[0] as f32 * shade) as u8, (color.0[1] as f32 * shade) as u8, (color.0[2] as f32 * shade) as u8, 255])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOURCE_COLOR: Rgba = Rgba([200, 120, 60, 255]);

    #[test]
    fn fixed_six_dir_carries_the_absorbed_constants() {
        let shading = Shading::default();
        assert_eq!(shading, Shading::FixedSixDir, "the default is the absorbed preset");
        assert_eq!(shading.factor(Direction::PosY), 1.0);
        assert_eq!(shading.factor(Direction::NegY), 0.45);
        assert_eq!(shading.factor(Direction::PosX), 0.72);
        assert_eq!(shading.factor(Direction::NegX), 0.62);
        assert_eq!(shading.factor(Direction::PosZ), 0.58);
        assert_eq!(shading.factor(Direction::NegZ), 0.66);
    }

    #[test]
    fn the_default_preset_paints_the_old_bytes() {
        // Every painted channel is the source formula
        // `(channel as f32 * shade) as u8` with alpha forced to 255,
        // evaluated with the same f32 multipliers the binding used.
        let shading = Shading::default();
        assert_eq!(shading.paint(Direction::PosY, SOURCE_COLOR), Rgba([200, 120, 60, 255]));
        assert_eq!(shading.paint(Direction::NegY, SOURCE_COLOR), Rgba([90, 54, 27, 255]));
        assert_eq!(shading.paint(Direction::PosX, SOURCE_COLOR), Rgba([144, 86, 43, 255]));
        assert_eq!(shading.paint(Direction::NegX, SOURCE_COLOR), Rgba([124, 74, 37, 255]));
        assert_eq!(shading.paint(Direction::PosZ, SOURCE_COLOR), Rgba([116, 69, 34, 255]));
        assert_eq!(shading.paint(Direction::NegZ, SOURCE_COLOR), Rgba([132, 79, 39, 255]));
    }

    #[test]
    fn paint_truncates_and_forces_alpha() {
        // Truncation, not rounding: 255 * 0.45 = 114.75 lands on 114, and a
        // transparent voxel still paints opaque.
        let shading = Shading::FixedSixDir;
        assert_eq!(shading.paint(Direction::NegY, Rgba([255, 255, 255, 0])), Rgba([114, 114, 114, 255]));
        let shading = Shading::Uniform(0.5);
        assert_eq!(shading.paint(Direction::PosY, Rgba([100, 200, 50, 17])), Rgba([50, 100, 25, 255]));
    }

    #[test]
    fn uniform_shading_scales_every_direction_equally() {
        let shading = Shading::Uniform(0.5);
        for direction in [Direction::PosY, Direction::NegY, Direction::PosX, Direction::NegX, Direction::PosZ, Direction::NegZ]
        {
            assert_eq!(shading.factor(direction), 0.5);
            assert_eq!(shading.paint(direction, SOURCE_COLOR), Rgba([100, 60, 30, 255]));
        }
    }
}
