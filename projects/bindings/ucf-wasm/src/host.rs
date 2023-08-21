//! Host import surface for Web (`ucf_host` wasm module).

#[cfg(target_arch = "wasm32")]
#[link(wasm_import_module = "ucf_host")]
extern "C" {
    /// Bind a host-side external buffer handle to logical `resource_id`.
    ///
    /// Returns `0` on success or a negative stable error code.
    pub fn ucf_host_bind_external_buffer(resource_id: u32, handle_id: u32) -> i32;
}

/// Forward bind request to the JS / TS host registry.
pub fn bind_external_buffer_via_host(resource_id: u32, handle_id: u32) -> i32 {
    #[cfg(target_arch = "wasm32")]
    unsafe {
        ucf_host_bind_external_buffer(resource_id, handle_id)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = (resource_id, handle_id);
        -20
    }
}
