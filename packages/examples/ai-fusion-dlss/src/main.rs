use ucf::prelude::*;

fn graph() -> Graph {
    let mut b = GraphBuilder::new();
    b.texture(1, Domain::Vram);
    b.texture(2, Domain::Vram);
    b.buffer(3, Domain::Vram);
    b.texture(4, Domain::Vram);
    b.gpu_raster(101, "RasterShader");
    b.executor(102, "DenoiseNet", "Npu", TaskKind::Custom("DenoiseNet".into()), Objective::MinLatency { deadline_micros: 8_000 }, Priority::Interactive);
    b.executor(103, "Present", "Display", TaskKind::Raster, Objective::MinLatency { deadline_micros: 16_000 }, Priority::Interactive);
    b.chain(&[101, 102, 103]);
    b.build()
}

fn main() {
    dry_run(&graph()).expect("example");
}
