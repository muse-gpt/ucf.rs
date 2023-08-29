/**
 * L2 宿主 WebGPU 桥（直接浏览器 API，无 Rust `wgpu` crate）。
 */

export interface WebGpuHostState {
    device: GPUDevice;
    context: GPUCanvasContext;
    format: GPUTextureFormat;
}

let hostState: WebGpuHostState | undefined;

/** 为 canvas 配置 WebGPU 上下文；须在调用 L2 wasm 导出前完成。 */
export async function initWebGpuHost(canvas: HTMLCanvasElement): Promise<WebGpuHostState> {
    const gpu = navigator.gpu;
    if (!gpu) {
        throw new Error("WebGPU unavailable");
    }
    const adapter = await gpu.requestAdapter();
    if (!adapter) {
        throw new Error("WebGPU adapter unavailable");
    }
    const device = await adapter.requestDevice();
    const context = canvas.getContext("webgpu");
    if (!context) {
        throw new Error("webgpu canvas context unavailable");
    }
    const format = gpu.getPreferredCanvasFormat();
    context.configure({
        device,
        format,
        alphaMode: "premultiplied",
    });
    hostState = { device, context, format };
    return hostState;
}

export function getWebGpuHostState(): WebGpuHostState | undefined {
    return hostState;
}

/** wasm import `ucf_host_l2_color_fill_webgpu` 实现。 */
export function hostL2ColorFillWebGpu(
    width: number,
    height: number,
    r: number,
    g: number,
    b: number,
    a: number,
): number {
    if (width === 0 || height === 0) {
        return 0;
    }
    const state = hostState;
    if (!state) {
        return -30;
    }
    try {
        const texture = state.context.getCurrentTexture();
        const encoder = state.device.createCommandEncoder();
        const pass = encoder.beginRenderPass({
            colorAttachments: [
                {
                    view: texture.createView(),
                    clearValue: {
                        r: r / 255,
                        g: g / 255,
                        b: b / 255,
                        a: a / 255,
                    },
                    loadOp: "clear",
                    storeOp: "store",
                },
            ],
        });
        pass.end();
        state.device.queue.submit([encoder.finish()]);
        return 0;
    } catch {
        return -31;
    }
}

/** wasm import `ucf_host_l2_raster_clear_webgpu`（与 Fill 同色清屏，Raster 语义）。 */
export function hostL2RasterClearWebGpu(
    width: number,
    height: number,
    r: number,
    g: number,
    b: number,
    a: number,
): number {
    return hostL2ColorFillWebGpu(width, height, r, g, b, a);
}

/** 与 `ucf_emitter` / CPU `raster_tri` 薄门对齐的 WGSL（`vertex_index` 大三角）。 */
const RASTER_TRI_WGSL = `
struct VSOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) col: vec4<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VSOut {
    var verts = array<vec2<f32>, 3>(
        vec2<f32>(-0.8, -0.8),
        vec2<f32>( 0.8, -0.8),
        vec2<f32>( 0.0,  0.8)
    );
    var o: VSOut;
    o.pos = vec4<f32>(verts[vid], 0.0, 1.0);
    o.col = vec4<f32>(1.0, 1.0, 0.0, 1.0);
    return o;
}

@fragment
fn fs_main(@location(0) col: vec4<f32>) -> @location(0) vec4<f32> {
    return col;
}
`;

let rasterTriPipeline: GPURenderPipeline | undefined;

function ensureRasterTriPipeline(state: WebGpuHostState): GPURenderPipeline {
    if (rasterTriPipeline) {
        return rasterTriPipeline;
    }
    const module = state.device.createShaderModule({ code: RASTER_TRI_WGSL });
    rasterTriPipeline = state.device.createRenderPipeline({
        layout: "auto",
        vertex: {
            module,
            entryPoint: "vs_main",
        },
        fragment: {
            module,
            entryPoint: "fs_main",
            targets: [{ format: state.format }],
        },
        primitive: {
            topology: "triangle-list",
        },
    });
    return rasterTriPipeline;
}

/** wasm import `ucf_host_l2_raster_tri_webgpu`：黑底 + NDC 黄三角。 */
export function hostL2RasterTriWebGpu(width: number, height: number): number {
    if (width === 0 || height === 0) {
        return 0;
    }
    const state = hostState;
    if (!state) {
        return -30;
    }
    try {
        const texture = state.context.getCurrentTexture();
        const pipeline = ensureRasterTriPipeline(state);
        const encoder = state.device.createCommandEncoder();
        const pass = encoder.beginRenderPass({
            colorAttachments: [
                {
                    view: texture.createView(),
                    clearValue: { r: 0, g: 0, b: 0, a: 1 },
                    loadOp: "clear",
                    storeOp: "store",
                },
            ],
        });
        pass.setPipeline(pipeline);
        pass.draw(3);
        pass.end();
        state.device.queue.submit([encoder.finish()]);
        return 0;
    } catch {
        return -31;
    }
}
