use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::Feature;

/// Ordered set of [`Feature`] flags advertised by a backend.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeatureSet {
    features: BTreeSet<Feature>,
}

impl FeatureSet {
    /// Empty feature set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert `feature` and return `self` for chaining.
    pub fn with(mut self, feature: Feature) -> Self {
        self.features.insert(feature);
        self
    }

    /// Whether `feature` is present.
    pub fn has(&self, feature: Feature) -> bool {
        self.features.contains(&feature)
    }

    /// Iterate features in sorted order.
    pub fn iter(&self) -> impl Iterator<Item = &Feature> {
        self.features.iter()
    }

    /// Number of advertised features.
    pub fn len(&self) -> usize {
        self.features.len()
    }

    /// Whether no features are advertised.
    pub fn is_empty(&self) -> bool {
        self.features.is_empty()
    }
}
