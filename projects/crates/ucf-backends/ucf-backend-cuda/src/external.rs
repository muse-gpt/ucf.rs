//! CUDA [`ExternalBuffer`] / [`ExecStream`] / [`ExecEvent`] adapters.

use std::any::Any;
use std::sync::Arc;

use ucf_scheduler::{
    Error as SchedulerError, ExecEvent, ExecStream, ExternalBuffer, Result as SchedulerResult,
    StreamEventBridge,
};

use crate::driver::{CudaDriver, CUdeviceptr, CUevent, CUstream, DriverError};

fn map_driver(err: DriverError) -> SchedulerError {
    SchedulerError::Backend("cuda".into(), err.to_string())
}

/// Device-resident buffer bound into a UCF resource id (may be owned by DXO / Titan).
pub struct CudaExternalBuffer {
    ptr: CUdeviceptr,
    byte_size: u64,
}

impl CudaExternalBuffer {
    /// Borrow an existing device pointer (caller retains free responsibility).
    pub fn from_device_ptr(ptr: CUdeviceptr, byte_size: u64) -> Arc<Self> {
        Arc::new(Self { ptr, byte_size })
    }

    /// Raw device pointer.
    pub fn device_ptr(&self) -> CUdeviceptr {
        self.ptr
    }
}

impl ExternalBuffer for CudaExternalBuffer {
    fn backend_name(&self) -> &str {
        "cuda"
    }

    fn byte_size(&self) -> u64 {
        self.byte_size
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// CUDA stream handle for [`ExecutionBindings`](ucf_scheduler::ExecutionBindings).
pub struct CudaExecStream {
    stream: CUstream,
}

impl CudaExecStream {
    /// Wrap a raw `CUstream` (must remain valid for the binding lifetime).
    pub fn from_raw(stream: CUstream) -> Arc<Self> {
        Arc::new(Self { stream })
    }

    /// Raw stream pointer.
    pub fn raw(&self) -> CUstream {
        self.stream
    }
}

impl ExecStream for CudaExecStream {
    fn backend_name(&self) -> &str {
        "cuda"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// Stream pointers are owned by the CUDA context / adapter and used on the submit thread.
unsafe impl Send for CudaExecStream {}
unsafe impl Sync for CudaExecStream {}

/// CUDA event for cross-stream waits.
pub struct CudaExecEvent {
    event: CUevent,
}

impl CudaExecEvent {
    /// Wrap a raw `CUevent`.
    pub fn from_raw(event: CUevent) -> Arc<Self> {
        Arc::new(Self { event })
    }

    /// Raw event pointer.
    pub fn raw(&self) -> CUevent {
        self.event
    }
}

impl ExecEvent for CudaExecEvent {
    fn backend_name(&self) -> &str {
        "cuda"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// Event pointers are owned by the CUDA context / adapter and used on the submit thread.
unsafe impl Send for CudaExecEvent {}
unsafe impl Sync for CudaExecEvent {}

/// Records / waits CUDA events on streams belonging to one driver context.
pub struct CudaStreamEventBridge<'a> {
    driver: &'a CudaDriver,
}

impl<'a> CudaStreamEventBridge<'a> {
    /// Bridge bound to a live driver.
    pub fn new(driver: &'a CudaDriver) -> Self {
        Self { driver }
    }
}

impl StreamEventBridge for CudaStreamEventBridge<'_> {
    fn wait_event(&self, stream: &dyn ExecStream, event: &dyn ExecEvent) -> SchedulerResult<()> {
        let stream = stream
            .as_any()
            .downcast_ref::<CudaExecStream>()
            .ok_or_else(|| {
                SchedulerError::Backend(
                    "cuda".into(),
                    format!(
                        "wait_event expects CudaExecStream, got backend `{}`",
                        stream.backend_name()
                    ),
                )
            })?;
        let event = event
            .as_any()
            .downcast_ref::<CudaExecEvent>()
            .ok_or_else(|| {
                SchedulerError::Backend(
                    "cuda".into(),
                    format!(
                        "wait_event expects CudaExecEvent, got backend `{}`",
                        event.backend_name()
                    ),
                )
            })?;
        self.driver
            .stream_wait_event(stream.raw(), event.raw())
            .map_err(map_driver)
    }

    fn record_event(&self, stream: &dyn ExecStream, event: &dyn ExecEvent) -> SchedulerResult<()> {
        let stream = stream
            .as_any()
            .downcast_ref::<CudaExecStream>()
            .ok_or_else(|| {
                SchedulerError::Backend(
                    "cuda".into(),
                    format!(
                        "record_event expects CudaExecStream, got backend `{}`",
                        stream.backend_name()
                    ),
                )
            })?;
        let event = event
            .as_any()
            .downcast_ref::<CudaExecEvent>()
            .ok_or_else(|| {
                SchedulerError::Backend(
                    "cuda".into(),
                    format!(
                        "record_event expects CudaExecEvent, got backend `{}`",
                        event.backend_name()
                    ),
                )
            })?;
        self.driver
            .event_record(event.raw(), stream.raw())
            .map_err(map_driver)
    }
}
