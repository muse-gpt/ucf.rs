//! Stable identity for animatable nodes.

use serde::{Deserialize, Serialize};

/// A stable name for one node, scene or component.
///
/// The name is authored once and survives edits: revisions, locks, revisions
/// of a component or a review comment refer to it instead of to "the third
/// text in scene 2". Empty or whitespace-bearing names are rejected at the
/// boundary instead of becoming an unreferenceable object.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[serde(transparent)]
pub struct NodeId(String);

impl NodeId {
    /// Validates and wraps an authored identifier.
    pub fn new(raw: impl Into<String>) -> Result<Self, media_types::Error> {
        let raw = raw.into();
        if raw.is_empty() {
            return Err(media_types::Error::invalid("node id must not be empty"));
        }
        if raw.chars().any(char::is_whitespace) {
            return Err(media_types::Error::invalid(format!("node id `{raw}` must not contain whitespace")));
        }
        Ok(Self(raw))
    }

    /// The identifier as a string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_empty_and_whitespace_ids() {
        assert!(NodeId::new("").is_err());
        assert!(NodeId::new("left panel").is_err());
        assert_eq!(NodeId::new("left-panel").unwrap().as_str(), "left-panel");
    }
}
