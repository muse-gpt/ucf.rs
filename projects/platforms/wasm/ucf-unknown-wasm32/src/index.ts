/**
 * UCF Wasm 平台袋：`ucf-wasm` cdylib + 宿主句柄表。
 */

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
    };
}

export default { loadUcf, UcfHostHandleRegistry, rustTarget, platformPackage };
