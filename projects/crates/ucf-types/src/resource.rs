use serde::{Deserialize, Serialize};

use crate::ids::ResourceId;

/// Physical or logical storage tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Domain {
    Register,
    L1,
    L2,
    Hbm,
    Vram,
    Host,
    Ssd,
    Network,
}

/// How a shader or fixed-function stage may touch a resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Access {
    Read,
    Write,
    ReadWrite,
    Sample,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResourceKind {
    Buffer,
    Texture,
    Tensor,
    AccelStruct,
    Sampler,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResourceNode {
    pub id: ResourceId,
    pub kind: ResourceKind,
    pub domain: Domain,
    pub access: Access,
    /// Logical byte size when known; `None` means unbounded / streaming.
    pub byte_size: Option<u64>,
}
