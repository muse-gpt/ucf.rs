# ucf-program

UCF 通用程序 IR 与 CPU 语义参考解释器。

- **L0**：表达式（算术、向量、数学函数、`select`）
- **L1**：结构化程序（局部变量、`if`、有上限 `for`/`while`、`return`、函数调用）
- **L2（最小）**：`Global1D` 并行实例索引

任务图（`ucf-ir`）描述调度与资源；本 crate 描述**如何计算**。后端 lowering 与 GPU 路径应消费 `Program`，而非领域专用算子名。
