//! Frame-budget helpers for interactive scheduling thin gates.

/// Nominal frame budget for 120 Hz interactive work (microseconds).
pub const FRAME_BUDGET_120HZ_MICROS: u64 = 8_333;

/// Whether `deadline_micros` fits inside a 120 Hz frame budget.
pub fn fits_120hz_frame(deadline_micros: u64) -> bool {
    deadline_micros <= FRAME_BUDGET_120HZ_MICROS
}
