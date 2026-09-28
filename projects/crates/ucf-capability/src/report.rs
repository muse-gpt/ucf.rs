//! Human-readable capability reports assembled from backend [`FeatureSet`]s.

use crate::FeatureSet;

/// One backend row in a [`CapabilityReport`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendCapabilities {
    /// Backend name (e.g. `cpu`, `dx12`).
    pub name: String,
    /// Advertised feature set.
    pub features: FeatureSet,
}

/// Aggregated capability probe result across registered / available backends.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CapabilityReport {
    /// Backends in registration / probe order.
    pub backends: Vec<BackendCapabilities>,
}

impl CapabilityReport {
    /// Build a report from `(name, features)` pairs.
    pub fn from_backends(
        items: impl IntoIterator<Item = (impl Into<String>, FeatureSet)>,
    ) -> Self {
        Self {
            backends: items
                .into_iter()
                .map(|(name, features)| BackendCapabilities {
                    name: name.into(),
                    features,
                })
                .collect(),
        }
    }

    /// Stable text lines: `name: feat1, feat2` or `name: (none)`.
    pub fn lines(&self) -> Vec<String> {
        self.backends
            .iter()
            .map(|row| {
                let mut feats: Vec<String> = row
                    .features
                    .iter()
                    .map(|f| format!("{f:?}"))
                    .collect();
                if feats.is_empty() {
                    format!("{}: (none)", row.name)
                } else {
                    feats.sort();
                    format!("{}: {}", row.name, feats.join(", "))
                }
            })
            .collect()
    }

    /// Whether a backend name appears (exact match).
    pub fn contains_backend(&self, name: &str) -> bool {
        self.backends.iter().any(|b| b.name == name)
    }
}
