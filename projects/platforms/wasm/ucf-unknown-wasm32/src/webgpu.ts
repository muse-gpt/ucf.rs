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
