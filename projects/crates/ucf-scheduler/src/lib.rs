//! Task scheduler and [`Backend`] trait.
#![warn(missing_docs)]

mod backend;
mod deadline;
mod error;
mod noop;
mod scheduler;

pub use backend::Backend;
pub use deadline::effective_deadlines;
pub use error::{Error, Result};
pub use noop::NoopBackend;
pub use scheduler::Scheduler;
