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
