# `@ucf/unknown-wasm32`

UCF Wasm 平台袋：TypeScript 加载器 + `ucf_wasm_bg.wasm`（来自 Rust crate `ucf-wasm`）。

## 构建

```bash
rustup target add wasm32-unknown-unknown
pnpm --filter @ucf/unknown-wasm32 run build:wasm
pnpm --filter @ucf/unknown-wasm32 run build
```

## 用法

```js
import { loadUcf, UcfHostHandleRegistry } from "@ucf/unknown-wasm32";

const ucf = await loadUcf();
console.log(ucf.info());
const matmul = ucf.runApplicationContractMatmul();
console.log(matmul); // [3, 3, 7, 7]

const registry = new UcfHostHandleRegistry();
registry.register(9, { kind: "buffer", label: "demo" });
await loadUcf({ registry });
```

宿主通过 wasm import `ucf_host.ucf_host_bind_external_buffer` 维护 GPU 句柄表。

L0 降级（CPU readback → canvas 2D）：

```js
import { presentColorFillL0 } from "@ucf/unknown-wasm32";

const canvas = document.querySelector("canvas");
await presentColorFillL0(canvas, [32, 64, 128, 255]);
```

L2（UCF CPU Fill + 宿主 WebGPU 清屏，无 Rust `wgpu` crate）：

```js
import { presentColorFillL2 } from "@ucf/unknown-wasm32";

const canvas = document.querySelector("canvas");
await presentColorFillL2(canvas, [0, 0, 255, 255]);
```

L1 / 自动降级（WebGL2 → L0）：

```js
import { presentColorFillL1, presentColorFillAuto } from "@ucf/unknown-wasm32";

const { ucf, tier } = await presentColorFillAuto(canvas, [0, 128, 0, 255]);
console.log(tier); // "L2" | "L1" | "L0"
```

设计见工作区 `规划设计/ucf/44-ucf-web-platform.md`。
