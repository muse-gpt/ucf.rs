use ucf::prelude::*;
use ucf_example_kit::{GraphBuilder, dry_run};

fn graph() -> Graph {
    let mut b = GraphBuilder::new();
    b.buffer(1, Domain::Vram);
    b.texture(2, Domain::Vram);
    b.tensor(3, Domain::Vram);
    b.tensor(4, Domain::Vram);
    b.gpu_raster(101, "RasterShader");
    b.executor(102, "LLM", "Gpu", TaskKind::Dispatch, Objective::MaxThroughput, Priority::Batch);
    b.build()
}

fn main() {
    dry_run(&graph()).expect("example");
}
