# ucf.rs

**UCF — Unified Compute Fabric（统一计算织构）**

语言无关、后端无关的异构计算中间层。把图形、深度学习、推理、云游戏等负载统一为资源图 + 任务图 + 依赖图，由调度器映射到 CUDA、DX12、Vulkan、ROCm 等后端。

## 状态

阶段 0 骨架：IR、调度器、运行时、后端 stub（CUDA / DX12 / Vulkan / ROCm）。

## 仓库布局

```text
projects/crates/
├── ucf/              # 门面库（应用入口）
├── ucf-types/        # 共享底库
├── ucf-ir/           # 图 IR
├── ucf-capability/   # 能力探测
├── ucf-optimize/     # 优化描述符
├── ucf-scheduler/    # 调度器 + Backend trait
├── ucf-runtime/      # 运行时
└── backends/
    ├── ucf-backend-cuda/
    ├── ucf-backend-dx12/
    ├── ucf-backend-vulkan/
    └── ucf-backend-rocm/
```

设计文档在仓库外维护，不随开源 crate 发布。

## 构建

```bash
cargo build
cargo test
cargo run -p ucf-runtime --example matmul_cuda --features cuda
```

应用侧推荐依赖 `ucf`：

```toml
ucf = { path = "../projects/crates/ucf", features = ["cuda", "dx12"] }
```

## License

See [License.md](./License.md).
