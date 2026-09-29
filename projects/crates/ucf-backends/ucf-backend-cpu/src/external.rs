//! Host-side [`ExternalBuffer`] for the CPU reference backend.

use std::any::Any;
use std::sync::Arc;

use ucf_scheduler::{Error as SchedulerError, ExternalBuffer, Result as SchedulerResult};
use ucf_types::ResourceId;

use crate::params::BackendError;
use crate::store::SharedHostStore;

/// External host buffer living in a [`SharedHostStore`].
///
/// Must share the same store Arc as the [`crate::CpuBackend`] that binds it.
pub struct HostExternalBuffer {
    store: SharedHostStore,
    id: ResourceId,
    byte_size: u64,
}

impl HostExternalBuffer {
    /// Allocate (or re-ensure) `byte_size` bytes for `id` in `store`.
    pub fn allocate(
        store: SharedHostStore,
        id: ResourceId,
        byte_size: u64,
    ) -> SchedulerResult<Arc<Self>> {
        store
            .lock()
            .map_err(|_| SchedulerError::Backend("cpu".into(), "host store poisoned".into()))?
            .ensure(id, byte_size as usize)
            .map_err(|e: BackendError| SchedulerError::Backend("cpu".into(), e.0))?;
        Ok(Arc::new(Self {
            store,
            id,
            byte_size,
        }))
    }

    /// Resource id this buffer backs.
    pub fn resource_id(&self) -> ResourceId {
        self.id
    }

    /// Shared store handle.
    pub fn store(&self) -> &SharedHostStore {
        &self.store
    }
}

impl ExternalBuffer for HostExternalBuffer {
    fn backend_name(&self) -> &str {
        "cpu"
    }

    fn byte_size(&self) -> u64 {
        self.byte_size
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
