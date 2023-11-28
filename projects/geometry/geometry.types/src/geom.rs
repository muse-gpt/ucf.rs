//! 2D geometry primitives.

/// A point or vector in 2D.
pub type Vec2 = [f32; 2];

/// A width/height pair.
pub type Size = [f32; 2];

/// An axis-aligned rectangle: origin plus size.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl Rect {
    #[must_use]
    pub fn new(x: f32, y: f32, w: f32, h: f32) -> Self {
        Self { x, y, w, h }
    }

    #[must_use]
    pub fn center(&self) -> Vec2 {
        [self.x + self.w / 2.0, self.y + self.h / 2.0]
    }

    #[must_use]
    pub fn contains(&self, point: Vec2) -> bool {
        point[0] >= self.x && point[0] <= self.x + self.w && point[1] >= self.y && point[1] <= self.y + self.h
    }
}

/// Linear interpolation between two scalars.
#[must_use]
pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Euclidean distance between two points.
#[must_use]
pub fn distance(a: Vec2, b: Vec2) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    (dx * dx + dy * dy).sqrt()
}
