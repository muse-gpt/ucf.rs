//! Thin binary `.ucf` envelope: magic + wire major + JSON graph body.
//!
//! This is a **transport gate**, not a frozen native wire schema. The payload
//! remains JSON so field evolution stays on the existing serde path.

use crate::error::{Error, Result};
use crate::graph::Graph;

/// File magic: ASCII `UCF` followed by NUL.
pub const MAGIC: &[u8; 4] = b"UCF\0";

/// Current wire major accepted by [`encode`] / [`decode`].
pub const WIRE_MAJOR: u32 = 1;

/// Encode `graph` as `.ucf` bytes (`MAGIC` + LE `WIRE_MAJOR` + JSON).
pub fn encode(graph: &Graph) -> Result<Vec<u8>> {
    let json = serde_json::to_vec(graph).map_err(|e| Error::Serde(e.to_string()))?;
    let mut out = Vec::with_capacity(8 + json.len());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&WIRE_MAJOR.to_le_bytes());
    out.extend_from_slice(&json);
    Ok(out)
}

/// Decode `.ucf` bytes produced by [`encode`] (or an equal layout).
pub fn decode(bytes: &[u8]) -> Result<Graph> {
    if bytes.len() < 8 {
        return Err(Error::TruncatedBinary(format!(
            "need at least 8 header bytes, got {}",
            bytes.len()
        )));
    }
    if &bytes[0..4] != MAGIC.as_slice() {
        return Err(Error::InvalidMagic);
    }
    let major = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    if major != WIRE_MAJOR {
        return Err(Error::UnsupportedWireMajor {
            found: major,
            supported: WIRE_MAJOR,
        });
    }
    serde_json::from_slice(&bytes[8..]).map_err(|e| Error::Serde(e.to_string()))
}
