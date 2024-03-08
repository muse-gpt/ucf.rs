//! Annotation outline geometry: the closed paths a caption box, a balloon and
//! a balloon tail are stroked along, absorbed value-for-value from the
//! OpenComic core.

/// Half-width of the tail base where it meets the box edge.
pub const TAIL_BASE_HALF_WIDTH: f32 = 7.0;
/// Floor for the tail ray length, keeping a tail at the box centre finite.
pub const TAIL_MIN_LENGTH: f32 = 1.0;
/// Axis component below which the ray is treated as parallel to that axis.
pub const TAIL_AXIS_EPSILON: f32 = 1e-4;

/// The five-point closed loop of a plain rectangle, in the source stroke
/// order (top-left → top-right → bottom-right → bottom-left → close).
#[must_use]
pub fn rect_path(rect: [f32; 4]) -> Vec<[f32; 2]> {
    let [x, y, w, h] = rect;
    vec![[x, y], [x + w, y], [x + w, y + h], [x, y + h], [x, y]]
}

/// Rounded outline as a corner-cut octagon — the source approximation, kept
/// deliberately: cheaper than a curved stroke and pixel-locked by the export
/// golden. The radius clamps to half the shorter side.
#[must_use]
pub fn round_rect_path(rect: [f32; 4], radius: f32) -> Vec<[f32; 2]> {
    let [x, y, w, h] = rect;
    let r = radius.min(w / 2.0).min(h / 2.0);
    vec![
        [x + r, y],
        [x + w - r, y],
        [x + w, y + r],
        [x + w, y + h - r],
        [x + w - r, y + h],
        [x + r, y + h],
        [x, y + h - r],
        [x, y + r],
        [x + r, y],
    ]
}

/// The two base points of a balloon tail: the ray from the box centre to the
/// tail point crosses the box edge at `t = min(w/2/|ux|, h/2/|uy|)` and the
/// base is the perpendicular segment of half-width [`TAIL_BASE_HALF_WIDTH`]
/// around that crossing. A tail at the exact centre degenerates to two equal
/// points at the centre (finite, as in the source).
#[must_use]
pub fn tail_base(rect: [f32; 4], tail: [f32; 2]) -> [[f32; 2]; 2] {
    let [x, y, w, h] = rect;
    let (cx, cy) = (x + w / 2.0, y + h / 2.0);
    let (dx, dy) = (tail[0] - cx, tail[1] - cy);
    let len = (dx * dx + dy * dy).sqrt().max(TAIL_MIN_LENGTH);
    let (ux, uy) = (dx / len, dy / len);
    let tx = if ux.abs() > TAIL_AXIS_EPSILON { w / 2.0 / ux.abs() } else { f32::MAX };
    let ty = if uy.abs() > TAIL_AXIS_EPSILON { h / 2.0 / uy.abs() } else { f32::MAX };
    let t = tx.min(ty);
    let (bx, by) = (cx + ux * t, cy + uy * t);
    let (px, py) = (-uy, ux);
    [
        [bx + px * TAIL_BASE_HALF_WIDTH, by + py * TAIL_BASE_HALF_WIDTH],
        [bx - px * TAIL_BASE_HALF_WIDTH, by - py * TAIL_BASE_HALF_WIDTH],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rect_path_is_the_five_point_closed_loop() {
        assert_eq!(rect_path([1.0, 2.0, 3.0, 4.0]), vec![[1.0, 2.0], [4.0, 2.0], [4.0, 6.0], [1.0, 6.0], [1.0, 2.0]]);
    }

    #[test]
    fn octagon_clamps_the_radius() {
        // radius 40 on a 100 × 60 box clamps to h/2 = 30.
        let path = round_rect_path([0.0, 0.0, 100.0, 60.0], 40.0);
        assert_eq!(path.len(), 9);
        assert_eq!(path[0], [30.0, 0.0]);
        assert_eq!(path[2], [100.0, 30.0]);
        assert_eq!(path[4], [70.0, 60.0]);
        assert_eq!(path[6], [0.0, 30.0]);
        assert_eq!(path[8], path[0], "the octagon closes");
    }

    #[test]
    fn tail_base_crosses_the_nearest_edge() {
        // Straight down from a 100 × 60 box: crossing (50, 60), base along x.
        assert_eq!(tail_base([0.0, 0.0, 100.0, 60.0], [50.0, 100.0]), [[43.0, 60.0], [57.0, 60.0]]);
        // Straight right: crossing (100, 30), base along y.
        assert_eq!(tail_base([0.0, 0.0, 100.0, 60.0], [200.0, 30.0]), [[100.0, 37.0], [100.0, 23.0]]);
        // A diagonal ray exits the right edge first (t = 50/ux < 30/uy):
        // crossing (100, 45), base perpendicular to (0.958, 0.287).
        let [b0, b1] = tail_base([0.0, 0.0, 100.0, 60.0], [150.0, 60.0]);
        assert!((b0[0] - 97.989).abs() < 0.01 && (b0[1] - 51.705).abs() < 0.01, "{b0:?}");
        assert!((b1[0] - 102.011).abs() < 0.01 && (b1[1] - 38.295).abs() < 0.01, "{b1:?}");
    }

    #[test]
    fn a_tail_at_the_centre_stays_finite() {
        let [b0, b1] = tail_base([0.0, 0.0, 100.0, 60.0], [50.0, 30.0]);
        assert_eq!((b0, b1), ([50.0, 30.0], [50.0, 30.0]));
    }
}
