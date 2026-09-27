use ucf::prelude::*;
use ucf_example_kit::{GraphBuilder, dry_run};

fn graph() -> Graph {
    let mut b = GraphBuilder::new();
    b.buffer(1, Domain::Vram);
    b.buffer(2, Domain::Vram);
    b.buffer(3, Domain::Vram);
    b.buffer(4, Domain::Vram);
    b.gpu(101, "Advect");
    b.gpu(102, "Advect");
    b.gpu(103, "Divergence");
    b.gpu(104, "Jacobi");
    b.gpu(105, "Project");
    b.chain(&[101, 103, 104, 105, 102]);
    b.build()
}

fn main() {
    dry_run(&graph()).expect("example");
}
