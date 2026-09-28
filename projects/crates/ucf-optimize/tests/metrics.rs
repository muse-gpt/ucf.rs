use ucf_optimize::{attention_flops, AttentionMetrics, matmul_flops, MatmulMetrics};

#[test]
fn matmul_flops_is_two_mnk() {
    assert_eq!(matmul_flops(2, 2, 2), 16);
    assert_eq!(matmul_flops(64, 64, 64), 2 * 64 * 64 * 64);
}

#[test]
fn matmul_metrics_throughput_and_mfu() {
    let metrics = MatmulMetrics::new(2, 2, 2, 1_000_000); // 1 ms, 16 FLOPs
    let fps = metrics.flop_per_s();
    assert!((fps - 16_000.0).abs() < 1e-6);
    assert_eq!(metrics.mfu(32_000.0), Some(0.5));
    assert_eq!(metrics.mfu(0.0), None);
}

#[test]
fn attention_flops_is_four_bhssd() {
    // batch=1 heads=1 seq=2 dim=2 → 4*1*1*2*2*2 = 32
    assert_eq!(attention_flops(1, 1, 2, 2), 32);
    let metrics = AttentionMetrics::new(1, 1, 2, 2, 1_000_000);
    assert_eq!(metrics.flops, 32);
    assert!((metrics.flop_per_s() - 32_000.0).abs() < 1e-6);
    assert_eq!(metrics.mfu(64_000.0), Some(0.5));
}
