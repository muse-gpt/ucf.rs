//! Path interpolation and polygon queries.

use crate::geom::Vec2;

/// Smooths a polyline with a closed-form Catmull-Rom spline.
///
/// `segments` is the number of samples per input segment. Open paths repeat
/// their end point so the curve starts and ends exactly on the input.
#[must_use]
pub fn smooth_polyline(points: &[Vec2], closed: bool, segments: u32) -> Vec<Vec2> {
    if points.len() < 2 {
        return points.to_vec();
    }
    let count = points.len();
    let at = |i: isize| -> Vec2 {
        let index =
            if closed { ((i % count as isize) + count as isize) % count as isize } else { i.clamp(0, count as isize - 1) };
        points[index as usize]
    };
    let mut out = Vec::with_capacity(count * segments as usize + 1);
    let last = if closed { count as isize } else { count as isize - 1 };
    for i in 0..last {
        let (p0, p1, p2, p3) = (at(i - 1), at(i), at(i + 1), at(i + 2));
        for s in 0..segments {
            let t = s as f32 / segments as f32;
            let t2 = t * t;
            let t3 = t2 * t;
            let mut point = [0.0f32; 2];
            for k in 0..2 {
                point[k] = 0.5
                    * ((2.0 * p1[k])
                        + (-p0[k] + p2[k]) * t
                        + (2.0 * p0[k] - 5.0 * p1[k] + 4.0 * p2[k] - p3[k]) * t2
                        + (-p0[k] + 3.0 * p1[k] - 3.0 * p2[k] + p3[k]) * t3);
            }
            out.push(point);
        }
    }
    if closed {
        out.push(out[0]);
    } else {
        out.push(points[count - 1]);
    }
    out
}

/// Closed outline of a rounded rectangle in absolute pixel coordinates.
///
/// The walk starts on the top edge at `(x + r, y)` and runs clockwise: every
/// straight edge ends where its corner arc begins, and each arc is sampled
/// `segments` times. The final point repeats the first, so the ring is closed
/// for both polygon fills and polyline strokes. With `radius` 0 the result is
/// the four-corner rectangle. This is the shared geometric reading of a
/// rounded-rect stroke: raster and vector encoders derive their outlines from
/// the same call and stay consistent.
#[must_use]
pub fn rounded_rect_outline(origin: Vec2, size: Vec2, radius: f32, segments: u32) -> Vec<Vec2> {
    let (x, y) = (origin[0], origin[1]);
    let (w, h) = (size[0], size[1]);
    let r = radius.max(0.0).min(w / 2.0).min(h / 2.0);
    if r <= 0.0 {
        return vec![[x, y], [x + w, y], [x + w, y + h], [x, y + h], [x, y]];
    }
    let segments = segments.max(1);
    // (centre, start angle degrees) of the four corner arcs in clockwise order.
    let corners = [
        ([x + w - r, y + r], 270.0f32),
        ([x + w - r, y + h - r], 0.0),
        ([x + r, y + h - r], 90.0),
        ([x + r, y + r], 180.0),
    ];
    let mut out = Vec::with_capacity(4 * segments as usize + 4);
    out.push([x + r, y]);
    for ([cx, cy], start) in corners {
        // End of the straight edge that precedes this arc.
        let edge_end = [cx + r * start.to_radians().cos(), cy + r * start.to_radians().sin()];
        if (edge_end[0] - out[out.len() - 1][0]).abs() > f32::EPSILON
            || (edge_end[1] - out[out.len() - 1][1]).abs() > f32::EPSILON
        {
            out.push(edge_end);
        }
        for i in 1..=segments {
            let angle = (start + 90.0 * i as f32 / segments as f32).to_radians();
            out.push([cx + r * angle.cos(), cy + r * angle.sin()]);
        }
    }
    out
}

/// Arithmetic mean of the points; the fallback pivot for per-path transforms.
#[must_use]
pub fn centroid(points: &[Vec2]) -> Vec2 {
    if points.is_empty() {
        return [0.0, 0.0];
    }
    let count = points.len() as f32;
    [points.iter().map(|p| p[0]).sum::<f32>() / count, points.iter().map(|p| p[1]).sum::<f32>() / count]
}

/// Even-odd point-in-polygon test.
#[must_use]
pub fn point_in_polygon(points: &[Vec2], point: Vec2) -> bool {
    let mut inside = false;
    let mut j = points.len().wrapping_sub(1);
    for i in 0..points.len() {
        let (pi, pj) = (points[i], points[j]);
        if (pi[1] > point[1]) != (pj[1] > point[1]) {
            let t = (point[1] - pi[1]) / (pj[1] - pi[1]);
            if point[0] < pi[0] + t * (pj[0] - pi[0]) {
                inside = !inside;
            }
        }
        j = i;
    }
    inside
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_radius_is_the_corner_rectangle() {
        let outline = rounded_rect_outline([2.0, 3.0], [20.0, 10.0], 0.0, 12);
        assert_eq!(
            outline,
            vec![[2.0, 3.0], [22.0, 3.0], [22.0, 13.0], [2.0, 13.0], [2.0, 3.0]]
        );
    }

    #[test]
    fn radius_is_clamped_to_the_short_half_edge() {
        let wide = rounded_rect_outline([0.0, 0.0], [40.0, 20.0], 99.0, 4);
        // Clamped to h/2 = 10: the shape becomes a capsule spanning the box.
        let right = wide.iter().fold([0.0f32; 2], |a, p| if p[0] > a[0] { *p } else { a });
        assert!((right[0] - 40.0).abs() < 1e-4, "{right:?}");
        let left = wide.iter().fold([1e9, 1e9], |a, p| if p[0] < a[0] { *p } else { a });
        assert!(left[0].abs() < 1e-4, "{left:?}");
        let top = wide.iter().fold([1e9, 1e9], |a, p| if p[1] < a[1] { *p } else { a });
        assert!(top[1].abs() < 1e-4, "{top:?}");
        // An unclamped radius would throw samples outside the box.
        for point in &wide {
            assert!(
                point[0] >= -1e-4 && point[0] <= 40.0 + 1e-4 && point[1] >= -1e-4 && point[1] <= 20.0 + 1e-4,
                "{point:?} outside the canvas box"
            );
        }
    }

    #[test]
    fn outline_is_closed_and_arcs_are_sampled() {
        let outline = rounded_rect_outline([0.0, 0.0], [30.0, 20.0], 5.0, 8);
        assert_eq!(outline.len(), 4 * 8 + 5, "start plus 4 edge starts plus 4 arcs of 8");
        assert_eq!(outline.first(), outline.last(), "the ring closes on itself");
        assert_eq!(outline[0], [5.0, 0.0], "the walk starts on the top edge");
        // The top-right arc runs from (25, 0) to (30, 5) around (25, 5):
        // one edge start plus eight samples, all on the circle.
        for point in &outline[1..=9] {
            let dx = point[0] - 25.0;
            let dy = point[1] - 5.0;
            assert!((dx * dx + dy * dy - 25.0).abs() < 1e-3, "{point:?} off the corner arc");
        }
        assert!((outline[9][0] - 30.0).abs() < 1e-4 && (outline[9][1] - 5.0).abs() < 1e-4, "the arc lands on the right edge: {:?}", outline[9]);
    }

    #[test]
    fn negative_radius_behaves_like_zero() {
        assert_eq!(
            rounded_rect_outline([1.0, 1.0], [8.0, 6.0], -4.0, 12),
            rounded_rect_outline([1.0, 1.0], [8.0, 6.0], 0.0, 12)
        );
    }
}
