//! Downstream-facing execution bindings: external buffers, streams, and events.
//!
//! These types are **runtime** contracts. They are not part of the `.ucf` wire
//! format — adapters (DXO / Spark) bind live device objects after decode.

use std::any::Any;
use std::collections::BTreeMap;
use std::sync::Arc;

use ucf_ir::ResourceId;

use crate::error::{Error, Result};

/// Backend-owned buffer that an application already allocated (device or host).
///
/// Adapters downcast via [`ExternalBuffer::as_any`] to recover CUDA device
/// pointers, Titan `Buffer` handles, wgpu buffers, etc. UCF itself never depends
/// on those crates.
pub trait ExternalBuffer: Send + Sync {
    /// Backend label this buffer belongs to (`cpu`, `cuda`, …).
    fn backend_name(&self) -> &str;

    /// Declared byte size (must match the graph [`ucf_types::ResourceNode`] size).
    fn byte_size(&self) -> u64;

    /// Type-erasure hook for backend-private downcasts.
    fn as_any(&self) -> &dyn Any;
}

/// Queue / stream the adapter wants UCF submits to use.
///
/// When set, backends must not invent an unrelated default stream for work that
/// participates in the adapter's dependency model.
pub trait ExecStream: Send + Sync {
    /// Backend label (`cpu`, `cuda`, …).
    fn backend_name(&self) -> &str;

    /// Type-erasure hook.
    fn as_any(&self) -> &dyn Any;
}

/// Synchronization token recorded or waited on a stream.
pub trait ExecEvent: Send + Sync {
    /// Backend label.
    fn backend_name(&self) -> &str;

    /// Type-erasure hook.
    fn as_any(&self) -> &dyn Any;
}

/// Cross-stream dependency encoding (upload → compute → readback).
pub trait StreamEventBridge: Send + Sync {
    /// Make `stream` wait until `event` completes.
    fn wait_event(&self, stream: &dyn ExecStream, event: &dyn ExecEvent) -> Result<()>;

    /// Record `event` after prior work on `stream`.
    fn record_event(&self, stream: &dyn ExecStream, event: &dyn ExecEvent) -> Result<()>;
}

/// Host CPU immediate stream (no GPU queue).
#[derive(Debug, Default)]
pub struct ImmediateStream;

impl ExecStream for ImmediateStream {
    fn backend_name(&self) -> &str {
        "cpu"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Host CPU immediate event (already complete once recorded).
#[derive(Debug, Default)]
pub struct ImmediateEvent;

impl ExecEvent for ImmediateEvent {
    fn backend_name(&self) -> &str {
        "cpu"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// No-op bridge for CPU / software paths (ordering is submission order).
#[derive(Debug, Default)]
pub struct ImmediateBridge;

impl StreamEventBridge for ImmediateBridge {
    fn wait_event(&self, stream: &dyn ExecStream, event: &dyn ExecEvent) -> Result<()> {
        if stream.backend_name() != event.backend_name() {
            return Err(Error::Backend(
                "cpu".into(),
                format!(
                    "wait_event backend mismatch: stream={} event={}",
                    stream.backend_name(),
                    event.backend_name()
                ),
            ));
        }
        Ok(())
    }

    fn record_event(&self, stream: &dyn ExecStream, event: &dyn ExecEvent) -> Result<()> {
        if stream.backend_name() != event.backend_name() {
            return Err(Error::Backend(
                "cpu".into(),
                format!(
                    "record_event backend mismatch: stream={} event={}",
                    stream.backend_name(),
                    event.backend_name()
                ),
            ));
        }
        Ok(())
    }
}

/// Live bindings supplied by a downstream adapter for one prepare/run cycle.
#[derive(Default, Clone)]
pub struct ExecutionBindings {
    /// External buffers keyed by IR resource id.
    pub buffers: BTreeMap<ResourceId, Arc<dyn ExternalBuffer>>,
    /// Preferred submission stream (optional).
    pub stream: Option<Arc<dyn ExecStream>>,
}

impl ExecutionBindings {
    /// Empty bindings (backend allocates everything).
    pub fn new() -> Self {
        Self::default()
    }

    /// Bind an external buffer to a resource id.
    pub fn bind_buffer(&mut self, id: ResourceId, buffer: Arc<dyn ExternalBuffer>) {
        self.buffers.insert(id, buffer);
    }

    /// Prefer this stream for subsequent submits.
    pub fn set_stream(&mut self, stream: Arc<dyn ExecStream>) {
        self.stream = Some(stream);
    }

    /// Lookup a bound buffer.
    pub fn buffer(&self, id: ResourceId) -> Option<&Arc<dyn ExternalBuffer>> {
        self.buffers.get(&id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyBuf {
        bytes: u64,
    }

    impl ExternalBuffer for DummyBuf {
        fn backend_name(&self) -> &str {
            "cpu"
        }
        fn byte_size(&self) -> u64 {
            self.bytes
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    #[test]
    fn bindings_round_trip_buffer() {
        let mut b = ExecutionBindings::new();
        b.bind_buffer(ResourceId(1), Arc::new(DummyBuf { bytes: 16 }));
        assert_eq!(b.buffer(ResourceId(1)).unwrap().byte_size(), 16);
    }

    #[test]
    fn immediate_bridge_rejects_cross_backend() {
        let bridge = ImmediateBridge;
        let stream = ImmediateStream;
        struct Other;
        impl ExecEvent for Other {
            fn backend_name(&self) -> &str {
                "cuda"
            }
            fn as_any(&self) -> &dyn Any {
                self
            }
        }
        let err = bridge
            .wait_event(&stream, &Other)
            .expect_err("mismatch");
        assert!(matches!(err, Error::Backend(_, _)));
    }
}
