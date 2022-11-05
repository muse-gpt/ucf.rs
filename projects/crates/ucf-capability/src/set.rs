use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::Feature;

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeatureSet {
    features: BTreeSet<Feature>,
}

impl FeatureSet {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with(mut self, feature: Feature) -> Self {
        self.features.insert(feature);
        self
    }

    pub fn has(&self, feature: Feature) -> bool {
        self.features.contains(&feature)
    }

    pub fn iter(&self) -> impl Iterator<Item = &Feature> {
        self.features.iter()
    }
}
