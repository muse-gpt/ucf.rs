# ucf-ir

Language-neutral IR for Unified Compute Fabric: resource graph, task graph, and dependency edges.

- JSON serde round-trip (`tests/roundtrip.rs`)
- Thin binary `.ucf` envelope: magic `UCF\0` + LE wire major + JSON body (`encode` / `decode`, `tests/binary_ucf.rs`)
