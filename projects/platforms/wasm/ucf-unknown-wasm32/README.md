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

宿主通过 wasm import `ucf_host.ucf_host_bind_external_buffer` 维护 GPU 句柄表（L2 `browser-gpu` 后置接线）。

设计见工作区 `规划设计/ucf/44-ucf-web-platform.md`。
