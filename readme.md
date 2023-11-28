# ucf.rs

**UCF — Unified Compute Fabric**

A language-neutral, backend-neutral heterogeneous compute layer. Graphics, deep learning, inference, cloud gaming, and similar workloads are expressed as a resource graph, task graph, and dependency graph; the scheduler maps them onto CUDA, DX12, Vulkan, ROCm, and other backends.

## Status

Stage 0 skeleton: IR, scheduler, runtime, and backend stubs (CUDA / DX12 / Vulkan / ROCm).

## Layout

```text
projects/crates/
├── ucf/              # Public facade (application entry)
├── hardware-types/        # Shared value types
├── hardware-ir/           # Graph IR
├── hardware-capability/   # Feature detection
├── hardware-optimize/     # Optimization descriptors
├── hardware-scheduler/    # Scheduler + Backend trait
├── hardware-emitter/      # Built-in PTX / DXIL emitter
├── hardware-runtime/      # Runtime
└── ucf-backends/
    ├── hardware-backend-cuda/
    ├── hardware-backend-dx12/
    ├── hardware-backend-vulkan/
    └── hardware-backend-rocm/

projects/bindings/
    hardware-wasm/           # Wasm cdylib（Web CPU 应用契约；TS 加载器由下游产品封装）

projects/examples/      # Runnable IR examples (one crate per graph, src/main.rs)
```

`.ucf` is the binary wire format. Run a single example:

```bash
cargo run -p example-cfd-navier-stokes
cargo run -p example-deep-learning-attention
```

## Build

```bash
cargo build
cargo test
cargo run -p hardware-runtime --example matmul_cuda --features cuda
```

Applications should depend on `ucf`:

```toml
hardware = { path = "../projects/crates/ucf", features = ["cuda", "dx12"] }
```

## License

See [License.md](./License.md).
