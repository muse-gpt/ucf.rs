/**
 * UCF Wasm 平台袋：`ucf-wasm` cdylib + 宿主句柄表。
 */

import { hostL2ColorFillWebGpu, hostL2RasterClearWebGpu, initWebGpuHost } from "./webgpu.js";
import { hostL1ColorFillWebGl2, initWebGl2Host } from "./webgl2.js";

export { getWebGpuHostState, hostL2ColorFillWebGpu, hostL2RasterClearWebGpu, initWebGpuHost } from "./webgpu.js";
export type { WebGpuHostState } from "./webgpu.js";
export { getWebGl2HostState, hostL1ColorFillWebGl2, initWebGl2Host } from "./webgl2.js";
export type { WebGl2HostState } from "./webgl2.js";

/** 自动降级链实际使用的呈现档位。 */
export type UcfPresentTier = "L2" | "L1" | "L0";

export const rustTarget = "wasm32-unknown-unknown" as const;
export const platformPackage = "ucf-unknown-wasm32" as const;

/** 宿主侧外部资源句柄（WebGPU buffer / texture 等后置）。 */
export interface UcfHostHandle {
    kind: "buffer" | "texture" | "unknown";
    label?: string;
    /** 宿主私有对象（浏览器 `GPUBuffer` 等）。 */
    native?: unknown;
}

/** 逻辑 `resource_id` → 宿主句柄表（不进 `.ucf` wire）。 */
export class UcfHostHandleRegistry {
    readonly #handles = new Map<number, UcfHostHandle>();
    readonly #bindings = new Map<number, number>();

    register(handleId: number, handle: UcfHostHandle): void {
        this.#handles.set(handleId, handle);
    }

    get(handleId: number): UcfHostHandle | undefined {
        return this.#handles.get(handleId);
    }

    bindResource(resourceId: number, handleId: number): number {
        if (!this.#handles.has(handleId)) {
            return -1;
        }
        this.#bindings.set(resourceId, handleId);
        return 0;
    }

    boundHandle(resourceId: number): number | undefined {
        return this.#bindings.get(resourceId);
    }
}

export interface UcfWasmInfo {
    name: string;
    version: string;
    npm_package: string;
    versionCode: number;
    targetTriple: string;
    hasCpuBackend: boolean;
}

export interface UcfWasmBindings {
    info(): UcfWasmInfo;
    runApplicationContractMatmul(): Float32Array;
    applicationContractDiagMask(): number;
    bindExternalBuffer(resourceId: number, handleId: number): number;
    colorFillRgba8(
        width: number,
        height: number,
        rgba: [number, number, number, number],
        outPtr: number,
    ): number;
    l2ColorFillWebgpu(
        width: number,
        height: number,
        rgba: [number, number, number, number],
    ): number;
    l2RasterClearWebgpu(
        width: number,
        height: number,
        rgba: [number, number, number, number],
    ): number;
    rasterClearRgba8(
        width: number,
        height: number,
        rgba: [number, number, number, number],
        outPtr: number,
    ): number;
    memory: WebAssembly.Memory | undefined;
}

export interface LoadUcfOptions {
    wasmUrl?: string | URL;
    module?: WebAssembly.Module;
    /** 可选句柄表；提供时注入 `ucf_host` import。 */
    registry?: UcfHostHandleRegistry;
}

type WasmExports = {
    memory: WebAssembly.Memory;
    ucf_version_code: () => number;
    ucf_target_triple: () => number;
    ucf_has_cpu_backend: () => number;
    ucf_application_contract_matmul: (outPtr: number) => number;
    ucf_application_contract_diag_mask: (outPtr: number) => number;
    ucf_bind_external_buffer: (resourceId: number, handleId: number) => number;
    ucf_color_fill_rgba8: (
        width: number,
        height: number,
        r: number,
        g: number,
        b: number,
        a: number,
        outPtr: number,
    ) => number;
    ucf_l2_color_fill_webgpu: (
        width: number,
        height: number,
        r: number,
        g: number,
        b: number,
        a: number,
    ) => number;
    ucf_l2_raster_clear_webgpu: (
        width: number,
        height: number,
        r: number,
        g: number,
        b: number,
        a: number,
    ) => number;
    ucf_raster_clear_rgba8: (
        width: number,
        height: number,
        r: number,
        g: number,
        b: number,
        a: number,
        outPtr: number,
    ) => number;
};

function decodeVersion(code: number): string {
    const major = Math.floor(code / 1_000_000);
    const minor = Math.floor((code % 1_000_000) / 1_000);
    const patch = code % 1_000;
    return `${major}.${minor}.${patch}`;
}

function readCString(memory: WebAssembly.Memory, ptr: number): string {
    const view = new Uint8Array(memory.buffer);
    let end = ptr;
    while (view[end] !== 0) {
        end += 1;
    }
    return new TextDecoder().decode(view.subarray(ptr, end));
}

function buildImports(registry: UcfHostHandleRegistry): WebAssembly.Imports {
    return {
        ucf_host: {
            ucf_host_bind_external_buffer: (resourceId: number, handleId: number): number =>
                registry.bindResource(resourceId, handleId),
            ucf_host_l2_color_fill_webgpu: (
                width: number,
                height: number,
                r: number,
                g: number,
                b: number,
                a: number,
            ): number => hostL2ColorFillWebGpu(width, height, r, g, b, a),
            ucf_host_l2_raster_clear_webgpu: (
                width: number,
                height: number,
                r: number,
                g: number,
                b: number,
                a: number,
            ): number => hostL2RasterClearWebGpu(width, height, r, g, b, a),
        },
    };
}

async function instantiate(
    options: LoadUcfOptions,
): Promise<WasmExports | undefined> {
    const url = options.wasmUrl ?? new URL("../ucf_wasm_bg.wasm", import.meta.url);
    const imports = options.registry ? buildImports(options.registry) : buildImports(new UcfHostHandleRegistry());

    try {
        if (options.module) {
            const instance = await WebAssembly.instantiate(options.module, imports);
            return instance.exports as unknown as WasmExports;
        }
        if (typeof fetch === "function") {
            const result = await WebAssembly.instantiateStreaming(fetch(url.toString()), imports);
            return result.instance.exports as unknown as WasmExports;
        }
        const { readFile } = await import("node:fs/promises");
        const { fileURLToPath } = await import("node:url");
        const buf = await readFile(fileURLToPath(url));
        const result = await WebAssembly.instantiate(buf, imports);
        return result.instance.exports as unknown as WasmExports;
    } catch {
        return undefined;
    }
}

/**
 * 加载 `ucf-wasm` 产物并暴露应用契约探针。
 */
export async function loadUcf(options: LoadUcfOptions = {}): Promise<UcfWasmBindings> {
    const exports = await instantiate(options);
    if (
        !exports?.memory ||
        !exports.ucf_application_contract_matmul ||
        !exports.ucf_version_code
    ) {
        throw new Error("ucf_wasm_bg.wasm unavailable: run build:wasm first");
    }

    const memory = exports.memory;
    const runApplicationContractMatmul = (): Float32Array => {
        const outPtr = 256;
        const status = exports.ucf_application_contract_matmul(outPtr);
        if (status !== 0) {
            throw new Error(`ucf_application_contract_matmul failed: ${status}`);
        }
        return new Float32Array(memory.buffer.slice(outPtr, outPtr + 16));
    };

    return {
        memory,
        info: () => ({
            name: "Unified Compute Fabric",
            version: decodeVersion(exports.ucf_version_code()),
            npm_package: platformPackage,
            versionCode: exports.ucf_version_code(),
            targetTriple: readCString(memory, exports.ucf_target_triple()),
            hasCpuBackend: exports.ucf_has_cpu_backend() === 1,
        }),
        runApplicationContractMatmul,
        applicationContractDiagMask: () => {
            const outPtr = 512;
            const status = exports.ucf_application_contract_diag_mask(outPtr);
            if (status !== 0) {
                throw new Error(`ucf_application_contract_diag_mask failed: ${status}`);
            }
            return new DataView(memory.buffer).getUint32(outPtr, true);
        },
        bindExternalBuffer: (resourceId, handleId) =>
            exports.ucf_bind_external_buffer(resourceId, handleId),
        colorFillRgba8: (width, height, rgba, outPtr) =>
            exports.ucf_color_fill_rgba8(
                width,
                height,
                rgba[0],
                rgba[1],
                rgba[2],
                rgba[3],
                outPtr,
            ),
        l2ColorFillWebgpu: (width, height, rgba) => {
            if (!exports.ucf_l2_color_fill_webgpu) {
                throw new Error("ucf_l2_color_fill_webgpu missing: rebuild wasm");
            }
            return exports.ucf_l2_color_fill_webgpu(
                width,
                height,
                rgba[0],
                rgba[1],
                rgba[2],
                rgba[3],
            );
        },
        l2RasterClearWebgpu: (width, height, rgba) => {
            if (!exports.ucf_l2_raster_clear_webgpu) {
                throw new Error("ucf_l2_raster_clear_webgpu missing: rebuild wasm");
            }
            return exports.ucf_l2_raster_clear_webgpu(
                width,
                height,
                rgba[0],
                rgba[1],
                rgba[2],
                rgba[3],
            );
        },
        rasterClearRgba8: (width, height, rgba, outPtr) =>
            exports.ucf_raster_clear_rgba8(
                width,
                height,
                rgba[0],
                rgba[1],
                rgba[2],
                rgba[3],
                outPtr,
            ),
    };
}

/** L0 降级：把 wasm 线性内存中的 RGBA8 画到 2D canvas。 */
export function presentRgba8ToCanvas2d(
    ctx: CanvasRenderingContext2D,
    memory: WebAssembly.Memory,
    outPtr: number,
    width: number,
    height: number,
): void {
    const bytes = new Uint8ClampedArray(memory.buffer, outPtr, width * height * 4);
    ctx.putImageData(new ImageData(bytes, width, height), 0, 0);
}

/**
 * Fill + readback + `putImageData` 一步（L0 薄门）。
 */
export async function presentColorFillL0(
    canvas: HTMLCanvasElement,
    rgba: [number, number, number, number],
    options: LoadUcfOptions = {},
): Promise<UcfWasmBindings> {
    const ucf = await loadUcf(options);
    const width = canvas.width;
    const height = canvas.height;
    const outPtr = 4096;
    const status = ucf.colorFillRgba8(width, height, rgba, outPtr);
    if (status !== 0) {
        throw new Error(`ucf_color_fill_rgba8 failed: ${status}`);
    }
    const need = width * height * 4;
    if (outPtr + need > ucf.memory!.buffer.byteLength) {
        throw new Error("ucf wasm memory too small for rgba8 readback");
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) {
        throw new Error("canvas 2d context unavailable");
    }
    presentRgba8ToCanvas2d(ctx, ucf.memory!, outPtr, width, height);
    return ucf;
}

/**
 * L2：UCF CPU Fill 校验 + 宿主 WebGPU 清屏（须先 `initWebGpuHost`）。
 */
export async function presentColorFillL2(
    canvas: HTMLCanvasElement,
    rgba: [number, number, number, number],
    options: LoadUcfOptions = {},
): Promise<UcfWasmBindings> {
    await initWebGpuHost(canvas);
    const ucf = await loadUcf(options);
    const status = ucf.l2ColorFillWebgpu(canvas.width, canvas.height, rgba);
    if (status !== 0) {
        throw new Error(`ucf_l2_color_fill_webgpu failed: ${status}`);
    }
    return ucf;
}

/**
 * L1：UCF CPU Fill 探针 + 宿主 WebGL2 清屏。
 */
export async function presentColorFillL1(
    canvas: HTMLCanvasElement,
    rgba: [number, number, number, number],
    options: LoadUcfOptions = {},
): Promise<UcfWasmBindings> {
    initWebGl2Host(canvas);
    const ucf = await loadUcf(options);
    const probe = ucf.colorFillRgba8(1, 1, rgba, 4096);
    if (probe !== 0) {
        throw new Error(`ucf_color_fill_rgba8 probe failed: ${probe}`);
    }
    const code = hostL1ColorFillWebGl2(rgba[0], rgba[1], rgba[2], rgba[3]);
    if (code !== 0) {
        throw new Error(`hostL1ColorFillWebGl2 failed: ${code}`);
    }
    return ucf;
}

export interface PresentColorFillAutoResult {
    ucf: UcfWasmBindings;
    tier: UcfPresentTier;
}

function probeWebGl2(): boolean {
    if (typeof document === "undefined") {
        return false;
    }
    const probe = document.createElement("canvas");
    return !!probe.getContext("webgl2");
}

/**
 * L0 经离屏 canvas 再 `drawImage` 到目标（目标已绑定 WebGPU 时降级用）。
 */
export async function presentColorFillL0Blit(
    target: HTMLCanvasElement,
    rgba: [number, number, number, number],
    options: LoadUcfOptions = {},
): Promise<UcfWasmBindings> {
    const scratch = document.createElement("canvas");
    scratch.width = target.width;
    scratch.height = target.height;
    const ucf = await presentColorFillL0(scratch, rgba, options);
    const ctx = target.getContext("2d");
    if (!ctx) {
        throw new Error("target canvas 2d context unavailable for L0 blit");
    }
    ctx.drawImage(scratch, 0, 0);
    return ucf;
}

/**
 * L2 → L1 → L0 自动降级清屏。
 */
export async function presentColorFillAuto(
    canvas: HTMLCanvasElement,
    rgba: [number, number, number, number],
    options: LoadUcfOptions = {},
): Promise<PresentColorFillAutoResult> {
    const webGpu = typeof navigator !== "undefined" && !!navigator.gpu;
    const webGl2 = probeWebGl2();

    if (webGpu) {
        try {
            const ucf = await presentColorFillL2(canvas, rgba, options);
            return { ucf, tier: "L2" };
        } catch {
            // canvas 可能已绑定 WebGPU，不再尝试 L1
        }
    } else if (webGl2) {
        try {
            const ucf = await presentColorFillL1(canvas, rgba, options);
            return { ucf, tier: "L1" };
        } catch {
            // fall through
        }
    }

    try {
        const ucf = await presentColorFillL0(canvas, rgba, options);
        return { ucf, tier: "L0" };
    } catch {
        const ucf = await presentColorFillL0Blit(canvas, rgba, options);
        return { ucf, tier: "L0" };
    }
}

/**
 * L2：UCF CPU `Raster` 清屏校验 + 宿主 WebGPU 清屏。
 */
export async function presentRasterClearL2(
    canvas: HTMLCanvasElement,
    rgba: [number, number, number, number],
    options: LoadUcfOptions = {},
): Promise<UcfWasmBindings> {
    await initWebGpuHost(canvas);
    const ucf = await loadUcf(options);
    const status = ucf.l2RasterClearWebgpu(canvas.width, canvas.height, rgba);
    if (status !== 0) {
        throw new Error(`ucf_l2_raster_clear_webgpu failed: ${status}`);
    }
    return ucf;
}

/**
 * L0：`Raster` CPU 读回 + canvas 2D。
 */
export async function presentRasterClearL0(
    canvas: HTMLCanvasElement,
    rgba: [number, number, number, number],
    options: LoadUcfOptions = {},
): Promise<UcfWasmBindings> {
    const ucf = await loadUcf(options);
    const width = canvas.width;
    const height = canvas.height;
    const outPtr = 8192;
    const status = ucf.rasterClearRgba8(width, height, rgba, outPtr);
    if (status !== 0) {
        throw new Error(`ucf_raster_clear_rgba8 failed: ${status}`);
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) {
        throw new Error("canvas 2d context unavailable");
    }
    presentRgba8ToCanvas2d(ctx, ucf.memory!, outPtr, width, height);
    return ucf;
}

export interface PresentRasterClearAutoResult {
    ucf: UcfWasmBindings;
    tier: UcfPresentTier;
}

export async function presentRasterClearAuto(
    canvas: HTMLCanvasElement,
    rgba: [number, number, number, number],
    options: LoadUcfOptions = {},
): Promise<PresentRasterClearAutoResult> {
    const webGpu = typeof navigator !== "undefined" && !!navigator.gpu;
    const webGl2 = probeWebGl2();

    if (webGpu) {
        try {
            const ucf = await presentRasterClearL2(canvas, rgba, options);
            return { ucf, tier: "L2" };
        } catch {
            // fall through
        }
    } else if (webGl2) {
        try {
            initWebGl2Host(canvas);
            const ucf = await loadUcf(options);
            const probe = ucf.rasterClearRgba8(1, 1, rgba, 4096);
            if (probe !== 0) {
                throw new Error(`raster probe failed: ${probe}`);
            }
            const code = hostL1ColorFillWebGl2(rgba[0], rgba[1], rgba[2], rgba[3]);
            if (code !== 0) {
                throw new Error(`hostL1ColorFillWebGl2 failed: ${code}`);
            }
            return { ucf, tier: "L1" };
        } catch {
            // fall through
        }
    }

    try {
        const ucf = await presentRasterClearL0(canvas, rgba, options);
        return { ucf, tier: "L0" };
    } catch {
        const ucf = await presentColorFillL0Blit(canvas, rgba, options);
        return { ucf, tier: "L0" };
    }
}

export default {
    loadUcf,
    UcfHostHandleRegistry,
    presentRgba8ToCanvas2d,
    presentColorFillL0,
    presentColorFillL1,
    presentColorFillL2,
    presentColorFillL0Blit,
    presentColorFillAuto,
    presentRasterClearL0,
    presentRasterClearL2,
    presentRasterClearAuto,
    initWebGpuHost,
    initWebGl2Host,
    rustTarget,
    platformPackage,
};
