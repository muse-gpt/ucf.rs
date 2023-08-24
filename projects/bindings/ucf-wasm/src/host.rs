//! Host import surface for Web (`ucf_host` wasm module).

#[cfg(target_arch = "wasm32")]
#[link(wasm_import_module = "ucf_host")]
extern "C" {
    /// Bind a host-side external buffer handle to logical `resource_id`.
    ///
    /// Returns `0` on success or a negative stable error code.
    pub fn ucf_host_bind_external_buffer(resource_id: u32, handle_id: u32) -> i32;

    /// L2：宿主侧 WebGPU 清屏（设备须已由 TS 初始化）。
    pub fn ucf_host_l2_color_fill_webgpu(
        width: u32,
        height: u32,
        r: u8,
        g: u8,
        b: u8,
        a: u8,
    ) -> i32;

    /// L2：宿主侧 WebGPU Raster 清屏（与 Fill 同色参数，语义为 Raster）。
    pub fn ucf_host_l2_raster_clear_webgpu(
        width: u32,
        height: u32,
        r: u8,
        g: u8,
        b: u8,
        a: u8,
    ) -> i32;
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

/// L2 WebGPU 清屏委托宿主（浏览器 WebGPU API，非 `wgpu` crate）。
pub fn l2_color_fill_webgpu_via_host(width: u32, height: u32, rgba: [u8; 4]) -> i32 {
    #[cfg(target_arch = "wasm32")]
    unsafe {
        ucf_host_l2_color_fill_webgpu(width, height, rgba[0], rgba[1], rgba[2], rgba[3])
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = (width, height, rgba);
        -21
    }
}

/// L2 WebGPU Raster 清屏委托宿主。
pub fn l2_raster_clear_webgpu_via_host(width: u32, height: u32, rgba: [u8; 4]) -> i32 {
    #[cfg(target_arch = "wasm32")]
    unsafe {
        ucf_host_l2_raster_clear_webgpu(width, height, rgba[0], rgba[1], rgba[2], rgba[3])
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = (width, height, rgba);
        -22
    }
}
