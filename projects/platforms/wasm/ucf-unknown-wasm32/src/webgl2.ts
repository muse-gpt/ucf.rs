/**
 * L1 宿主 WebGL2 桥（Raster 清屏子集，无 Rust `wgpu` crate）。
 */

export interface WebGl2HostState {
    gl: WebGL2RenderingContext;
}

let hostState: WebGl2HostState | undefined;

/** 为 canvas 获取 WebGL2 上下文；须在 L1 清屏前完成。 */
export function initWebGl2Host(canvas: HTMLCanvasElement): WebGl2HostState {
    const gl = canvas.getContext("webgl2", {
        alpha: true,
        antialias: false,
        depth: false,
        stencil: false,
    });
    if (!gl) {
        throw new Error("WebGL2 unavailable");
    }
    hostState = { gl };
    return hostState;
}

export function getWebGl2HostState(): WebGl2HostState | undefined {
    return hostState;
}

/** L1 清屏：全屏 `gl.clearColor`。 */
export function hostL1ColorFillWebGl2(r: number, g: number, b: number, a: number): number {
    const state = hostState;
    if (!state) {
        return -40;
    }
    const gl = state.gl;
    try {
        gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
        gl.clearColor(r / 255, g / 255, b / 255, a / 255);
        gl.clear(gl.COLOR_BUFFER_BIT);
        return 0;
    } catch {
        return -41;
    }
}
