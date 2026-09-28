# UCF IR Examples

Each subdirectory is an independent workspace crate. `src/main.rs` builds a `Graph` with `GraphBuilder`, then dry-runs it through `ucf::prelude::*` (`dry_run` + `NoopBackend`).

```bash
cargo run -p example-cfd-navier-stokes
cargo run -p example-robotics-robot-loop
```
