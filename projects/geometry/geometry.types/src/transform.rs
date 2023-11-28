//! Affine transforms and the camera view.

use crate::geom::Vec2;

/// A 2D affine transform stored as the top two rows of a 3x3 matrix.
///
/// Points are column vectors: `[a c tx; b d ty; 0 0 1]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Affine2 {
    pub a: f32,
    pub b: f32,
    pub c: f32,
    pub d: f32,
    pub tx: f32,
    pub ty: f32,
}

impl Default for Affine2 {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Affine2 {
    pub const IDENTITY: Self = Self { a: 1.0, b: 0.0, c: 0.0, d: 1.0, tx: 0.0, ty: 0.0 };

    #[must_use]
    pub fn translate(x: f32, y: f32) -> Self {
        Self { tx: x, ty: y, ..Self::IDENTITY }
    }

    #[must_use]
    pub fn scale(x: f32, y: f32) -> Self {
        Self { a: x, d: y, ..Self::IDENTITY }
    }

    #[must_use]
    pub fn rotate(degrees: f32) -> Self {
        let (sin, cos) = degrees.to_radians().sin_cos();
        Self { a: cos, b: sin, c: -sin, d: cos, tx: 0.0, ty: 0.0 }
    }

    /// Scale, then rotate, then translate around `pivot`.
    #[must_use]
    pub fn around(pivot: Vec2, scale: f32, degrees: f32) -> Self {
        Self::translate(pivot[0], pivot[1])
            .then(&Self::rotate(degrees))
            .then(&Self::scale(scale, scale))
            .then(&Self::translate(-pivot[0], -pivot[1]))
    }

    /// Applies `rhs` first, then `self`.
    #[must_use]
    pub fn then(&self, rhs: &Self) -> Self {
        Self {
            a: self.a * rhs.a + self.c * rhs.b,
            b: self.b * rhs.a + self.d * rhs.b,
            c: self.a * rhs.c + self.c * rhs.d,
            d: self.b * rhs.c + self.d * rhs.d,
            tx: self.a * rhs.tx + self.c * rhs.ty + self.tx,
            ty: self.b * rhs.tx + self.d * rhs.ty + self.ty,
        }
    }

    #[must_use]
    pub fn apply(&self, point: Vec2) -> Vec2 {
        [self.a * point[0] + self.c * point[1] + self.tx, self.b * point[0] + self.d * point[1] + self.ty]
    }

    /// Returns `None` when the matrix is singular.
    #[must_use]
    pub fn inverse(&self) -> Option<Self> {
        let det = self.a * self.d - self.b * self.c;
        if det.abs() < 1e-9 {
            return None;
        }
        let inv = 1.0 / det;
        Some(Self {
            a: self.d * inv,
            b: -self.b * inv,
            c: -self.c * inv,
            d: self.a * inv,
            tx: (self.c * self.ty - self.d * self.tx) * inv,
            ty: (self.b * self.tx - self.a * self.ty) * inv,
        })
    }
}

/// A camera pose in normalized view space.
///
/// `focus` is the world point that lands in the middle of the frame and `zoom`
/// is the scale factor; `rotate` is in degrees. The transform is resolution
/// independent, so the same pose works for preview and final output.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Camera {
    pub zoom: f32,
    pub focus: Vec2,
    pub rotate: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self { zoom: 1.0, focus: [0.5, 0.5], rotate: 0.0 }
    }
}

impl Camera {
    /// The transform from normalized world space to normalized view space.
    #[must_use]
    pub fn transform(&self) -> Affine2 {
        let zoom = if self.zoom.abs() < 1e-6 { 1e-6 } else { self.zoom };
        Affine2::translate(0.5, 0.5)
            .then(&Affine2::scale(zoom, zoom))
            .then(&Affine2::rotate(-self.rotate))
            .then(&Affine2::translate(-self.focus[0], -self.focus[1]))
    }

    /// Maps a world point into view space.
    #[must_use]
    pub fn project(&self, point: Vec2) -> Vec2 {
        self.transform().apply(point)
    }

    /// Maps a view point back into world space.
    #[must_use]
    pub fn unproject(&self, point: Vec2) -> Vec2 {
        self.transform().inverse().map_or(point, |inverse| inverse.apply(point))
    }

    /// Normalized half-extent of the visible world at the current zoom.
    #[must_use]
    pub fn half_extent(&self) -> f32 {
        0.5 / self.zoom.abs().max(1e-6)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn camera_round_trips_through_the_inverse() {
        let camera = Camera { zoom: 1.7, focus: [0.42, 0.58], rotate: 12.0 };
        let world = [0.3, 0.7];
        let view = camera.project(world);
        let back = camera.unproject(view);
        assert!((back[0] - world[0]).abs() < 1e-4, "{back:?}");
        assert!((back[1] - world[1]).abs() < 1e-4, "{back:?}");
    }

    #[test]
    fn focus_lands_in_the_middle_of_the_frame() {
        let camera = Camera { zoom: 2.0, focus: [0.25, 0.75], rotate: 0.0 };
        let center = camera.project([0.25, 0.75]);
        assert!((center[0] - 0.5).abs() < 1e-5);
        assert!((center[1] - 0.5).abs() < 1e-5);
    }
}
