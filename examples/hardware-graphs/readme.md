# hardware-graphs 图构建示例 workspace

这里收纳原 `projects/examples` 的硬件图建模示例。每个 crate 用 `hardware::prelude::*` 的 `GraphBuilder` 构造一张图，然后调用 `dry_run(&graph())` 走完图校验与调度演练。

**能力边界（必读）：**

- 这些示例只验证图构建与 dry-run，不证明 CPU/GPU 真实执行，也不产出任何计算结果或产物。
- 图节点与资源名只是不透明标签，禁止把任何示例宣传为 CFD、数据库、深度学习训练或硬件执行能力。
- 真实后端执行与产物验收由 `projects/hardware` 各 crate 的测试单独负责，不在此报告。

## 运行

```bash
cd examples/hardware-graphs
cargo run -p chain-buffers
cargo run -p chain-raster
```

## 目录与旧名对照

G1 整理：原 27 个示例全部是「buffer/texture/tensor 资源 + 计算/执行器节点 + 线性依赖链」的同族图结构，彼此只在节点名、字符串和节点数量上不同。按实际示范的图结构重命名，并合并了结构相同的重复样例（每组保留一份代表图，其余删除）。

| 新目录 | 示范的图结构 | 保留自（旧目录） | 合并删除（旧目录） |
| --- | --- | --- | --- |
| `chain-buffers` | Vram buffer 上的线性 Custom 计算链 | `database-query` | `cfd-lbm`、`cfd-navier-stokes`、`science-conjugate-gradient`、`science-fft`、`science-molecular-dynamics`、`science-monte-carlo`、`simulation-cloth`、`simulation-sph` |
| `chain-copy-task` | 由 op 名推断 Copy 与 Custom 任务类型的 buffer 链 | `simulation-rigid-body` | `simulation-nbody` |
| `chain-textures` | Vram texture 上的线性计算链 | `graphics-post-process` | `image-video-image-pipeline` |
| `chain-tensors` | Vram tensor 上的线性计算链 | `deep-learning-attention` | — |
| `chain-buffer-texture` | buffer 与 texture 混合资源的计算链 | `image-video-video-pipeline` | — |
| `chain-raster` | `gpu_raster` 显式光栅任务链（交互延迟目标） | `graphics-raster` | — |
| `chain-inferred-raster` | 由 op 名推断 Raster 任务类型的混合资源链 | `graphics-ray-tracing` | — |
| `chain-matmul` | `gpu_matmul` 节点与 Custom 节点串联的 tensor 链 | `deep-learning-train-step` | `deep-learning-conv2d` |
| `chain-cross-executor` | GPU 节点与显式互联 executor 节点交替的链 | `distributed-ddp-train` | `distributed-tensor-parallel` |
| `chain-host-executors` | Host 域 buffer 上的纯 executor 链（Host/Fpga） | `exotic-hardware-fpga-accel` | — |
| `chain-hbm-executors` | Hbm/Vram 域 tensor 上的 Copy/MatMul executor 链 | `exotic-hardware-pim-matmul` | — |
| `chain-render-present` | 物理、光栅、NPU 与显示 executor 的延迟链 | `ai-fusion-physics-render-denoise` | `ai-fusion-dlss` |
| `chain-streaming-qos` | Host/Vram 混合资源与 Streaming/Interactive 截止期链 | `robotics-robot-loop` | — |
| `independent-nodes` | 无依赖边的独立节点图（混合资源类型） | `mixed-cloud-game-and-inference` | — |

原 `projects/examples` 中的示例只含 `Cargo.toml`、`readme.md` 与 `src/main.rs`，没有可断言的图校验测试，因此无测试断言需要迁入 `projects/hardware`；图结构覆盖由本 workspace 的 CI 运行阶段保持。
