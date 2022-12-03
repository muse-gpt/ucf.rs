# UCF IR Examples

Each subdirectory is an independent workspace crate. `src/main.rs` builds a real `ucf_ir::Graph` and dry-runs it through the noop scheduler.

Shared helpers: [`kit/`](./kit/) (`GraphBuilder` + `NoopBackend`).

```bash
cargo run -p example-cfd-navier-stokes
cargo run -p example-robotics-robot-loop
```
