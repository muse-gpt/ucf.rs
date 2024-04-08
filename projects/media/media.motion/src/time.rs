//! Mapping between project frames and the local frames a track is authored in.

use serde::{Deserialize, Serialize};

/// How a window repeats after its own length has elapsed.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum LoopMode {
    /// Plays once, then stops existing.
    #[default]
    Once,
    /// Restarts from the first frame.
    Repeat,
    /// Plays forward, then backward, then forward again.
    PingPong,
}

/// A visible window in project frames plus how it repeats.
///
/// The window is the only place a loop may happen. Evaluation outside it
/// returns `None`, so a looping element cannot silently leak past the shot it
/// was authored for.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TimeWindow {
    /// First project frame in the window.
    #[serde(default)]
    pub start: u32,
    /// Window length in frames; zero is rejected by [`TimeWindow::validate`].
    pub len: u32,
    #[serde(default)]
    pub loop_mode: LoopMode,
}

impl TimeWindow {
    /// A single-pass window.
    pub fn once(start: u32, len: u32) -> Self {
        Self { start, len, loop_mode: LoopMode::Once }
    }

    /// The local frame for `global`, or `None` when the window does not cover it.
    ///
    /// Sub-frame precision is preserved: the caller passes the fractional frame
    /// it is evaluating, and a loop wraps it without rounding.
    #[must_use]
    pub fn local(&self, global: f32) -> Option<f32> {
        if self.len == 0 || global < self.start as f32 {
            return None;
        }
        let offset = global - self.start as f32;
        let len = self.len as f32;
        match self.loop_mode {
            LoopMode::Once => (offset < len).then_some(offset),
            LoopMode::Repeat => Some(offset % len),
            LoopMode::PingPong => {
                let cycle = offset % (len * 2.0);
                Some(if cycle <= len { cycle } else { len * 2.0 - cycle })
            }
        }
    }

    /// Reports a window that cannot be evaluated.
    pub fn validate(&self, where_: &str, problems: &mut Vec<String>) {
        if self.len == 0 {
            problems.push(format!("{where_}: time window length must be positive"));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn once_stops_at_the_end() {
        let window = TimeWindow::once(10, 5);
        assert_eq!(window.local(9.0), None);
        assert_eq!(window.local(10.0), Some(0.0));
        assert_eq!(window.local(12.5), Some(2.5));
        assert_eq!(window.local(15.0), None);
    }

    #[test]
    fn repeat_wraps_and_pingpong_reverses() {
        let repeat = TimeWindow { start: 0, len: 4, loop_mode: LoopMode::Repeat };
        assert_eq!(repeat.local(6.0), Some(2.0));
        let ping = TimeWindow { start: 0, len: 4, loop_mode: LoopMode::PingPong };
        assert_eq!(ping.local(2.0), Some(2.0));
        assert_eq!(ping.local(6.0), Some(2.0));
        assert_eq!(ping.local(4.0), Some(4.0));
    }

    #[test]
    fn zero_length_is_rejected() {
        let mut problems = Vec::new();
        TimeWindow::once(0, 0).validate("node a", &mut problems);
        assert_eq!(problems.len(), 1);
    }
}
