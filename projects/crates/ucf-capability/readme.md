# ucf-capability

Backend feature detection and degradation-chain hooks for UCF.

- [`Feature`] / [`FeatureSet`] — advertised capability flags
- [`CapabilityReport`] — aggregate probe rows with stable `lines()` text
- `pick_*_strategy` — descriptor / pipeline / sync degradation picks

Unit: `tests/report.rs`, `tests/degrade.rs`. Live soft-skip probe: `ucf/tests/capability_probe.rs`.
