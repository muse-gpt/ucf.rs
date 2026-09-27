# ucf.rs

**UCF — Unified Compute Fabric**

A language-neutral, backend-neutral heterogeneous compute layer. Graphics, deep learning, inference, cloud gaming, and similar workloads are expressed as a resource graph, task graph, and dependency graph; the scheduler maps them onto CUDA, DX12, Vulkan, ROCm, and other backends.

## Status

Stage 0 skeleton: IR, scheduler, runtime, and backend stubs (CUDA / DX12 / Vulkan / ROCm).

## Layout

```text
projects/crates/
├── ucf/              # Public facade (application entry)
├── ucf-types/        # Shared value types
├── ucf-ir/           # Graph IR
├── ucf-capability/   # Feature detection
├── ucf-optimize/     # Optimization descriptors
├── ucf-scheduler/    # Scheduler + Backend trait
├── ucf-emitter/      # Built-in PTX / DXIL emitter
├── ucf-runtime/      # Runtime
└── ucf-backends/
    ├── ucf-backend-cuda/
    ├── ucf-backend-dx12/
    ├── ucf-backend-vulkan/
    └── ucf-backend-rocm/

packages/examples/      # Runnable IR examples (one crate per graph, src/main.rs)
packages/examples/kit/  # GraphBuilder + noop dry-run helpers
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
cargo run -p ucf-runtime --example matmul_cuda --features cuda
```

Applications should depend on `ucf`:

```toml
ucf = { path = "../projects/crates/ucf", features = ["cuda", "dx12"] }
```

## License

See [License.md](./License.md).
