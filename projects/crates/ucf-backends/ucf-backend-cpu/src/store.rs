use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use ucf_types::ResourceId;

use crate::params::BackendError;

/// In-memory byte buffers keyed by [`ResourceId`].
#[derive(Debug, Default, Clone)]
pub struct HostStore {
    buffers: BTreeMap<ResourceId, Vec<u8>>,
}

impl HostStore {
    /// Empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Ensure a buffer of `bytes` exists (zero-filled on first allocate).
    pub fn ensure(&mut self, id: ResourceId, bytes: usize) -> Result<(), BackendError> {
        self.buffers.entry(id).or_insert_with(|| vec![0u8; bytes]);
        let len = self.buffers.get(&id).map(|b| b.len()).unwrap_or(0);
        if len != bytes {
            return Err(BackendError(format!(
                "resource {} size mismatch: have {len}, need {bytes}",
                id.0
            )));
        }
        Ok(())
    }

    /// Raw mutable bytes for a resource.
    pub fn bytes_mut(&mut self, id: ResourceId) -> Result<&mut [u8], BackendError> {
        self.buffers
            .get_mut(&id)
            .map(|b| b.as_mut_slice())
            .ok_or_else(|| BackendError(format!("resource {} not allocated", id.0)))
    }

    /// Raw bytes for a resource.
    pub fn bytes(&self, id: ResourceId) -> Result<&[u8], BackendError> {
        self.buffers
            .get(&id)
            .map(|b| b.as_slice())
            .ok_or_else(|| BackendError(format!("resource {} not allocated", id.0)))
    }

    /// Byte length of an allocated resource.
    pub fn len_bytes(&self, id: ResourceId) -> Result<usize, BackendError> {
        Ok(self.bytes(id)?.len())
    }

    /// Interpret a buffer as little-endian `f32` values.
    pub fn read_f32(&self, id: ResourceId) -> Result<Vec<f32>, BackendError> {
        let bytes = self.bytes(id)?;
        if bytes.len() % 4 != 0 {
            return Err(BackendError(format!(
                "resource {} length {} is not a multiple of 4",
                id.0,
                bytes.len()
            )));
        }
        Ok(bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    /// Overwrite a buffer with little-endian `f32` values (length must match).
    pub fn write_f32(&mut self, id: ResourceId, values: &[f32]) -> Result<(), BackendError> {
        let bytes = self.bytes_mut(id)?;
        let need = values.len() * 4;
        if bytes.len() != need {
            return Err(BackendError(format!(
                "resource {} expected {need} bytes for {} f32, have {}",
                id.0,
                values.len(),
                bytes.len()
            )));
        }
        for (slot, value) in bytes.chunks_exact_mut(4).zip(values.iter()) {
            slot.copy_from_slice(&value.to_le_bytes());
        }
        Ok(())
    }
}

/// Shared handle so callers can seed and read back while the backend is owned by [`ucf_runtime::Runtime`].
pub type SharedHostStore = Arc<Mutex<HostStore>>;

/// Create a new shared host store.
pub fn shared_store() -> SharedHostStore {
    Arc::new(Mutex::new(HostStore::new()))
}
