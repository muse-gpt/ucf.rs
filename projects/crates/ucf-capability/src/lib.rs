//! Backend capability / feature flags.
#![warn(missing_docs)]

mod feature;
mod report;
mod set;

pub use feature::Feature;
pub use report::{BackendCapabilities, CapabilityReport};
pub use set::FeatureSet;
