//! Image layout: how a decoded frame maps into its box. Shared by the
//! raster painter and the scene IR so neither pulls the other in.

use serde::{Deserialize, Serialize};

/// How a decoded frame is mapped into its layout box.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum Fit {
    /// Scale to fit inside the box, preserving aspect ratio (letterbox).
    #[default]
    Contain,
    /// Fill the box, preserving aspect ratio, cropping the overflow.
    Cover,
    /// Fill the box exactly, ignoring aspect ratio.
    Stretch,
}
