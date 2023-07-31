//! Task scheduler and [`Backend`] trait.
#![warn(missing_docs)]

mod backend;
mod budget;
mod deadline;
mod error;
mod exec;
mod noop;
mod scheduler;

pub use backend::Backend;
pub use budget::{fits_120hz_frame, FRAME_BUDGET_120HZ_MICROS};
pub use deadline::effective_deadlines;
pub use error::{Error, ErrorCode, Result};
pub use exec::{
    ExecEvent, ExecStream, ExecutionBindings, ExternalBuffer, ImmediateBridge, ImmediateEvent,
    ImmediateStream, StreamEventBridge,
};
pub use noop::NoopBackend;
pub use scheduler::Scheduler;
