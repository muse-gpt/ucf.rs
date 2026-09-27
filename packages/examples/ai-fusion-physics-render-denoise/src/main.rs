use ucf::prelude::*;
use ucf_example_kit::{GraphBuilder, dry_run};

fn graph() -> Graph {
    let mut b = GraphBuilder::new();
    b.buffer(1, Domain::Vram);
    b.texture(2, Domain::Vram);
    b.texture(3, Domain::Vram);
    b.texture(4, Domain::Vram);
    b.gpu(101, "PhysicsKernel");
    b.gpu_raster(102, "RasterShader");
    b.executor(103, "DenoiseNet", "Npu", TaskKind::Custom("DenoiseNet".into()), Objective::MinLatency { deadline_micros: 8_000 }, Priority::Interactive);
    b.executor(104, "Present", "Display", TaskKind::Raster, Objective::MinLatency { deadline_micros: 16_000 }, Priority::Interactive);
    b.chain(&[101, 102, 103, 104]);
    b.build()
}

fn main() {
    dry_run(&graph()).expect("example");
}
