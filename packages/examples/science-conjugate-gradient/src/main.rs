use ucf::prelude::*;

fn graph() -> Graph {
    let mut b = GraphBuilder::new();
    b.buffer(1, Domain::Vram);
    b.buffer(2, Domain::Vram);
    b.buffer(3, Domain::Vram);
    b.buffer(4, Domain::Vram);
    b.buffer(5, Domain::Vram);
    b.buffer(6, Domain::Vram);
    b.gpu(101, "SpMV");
    b.gpu(102, "Dot");
    b.gpu(103, "AXPY");
    b.gpu(104, "Update");
    b.chain(&[101, 102, 103, 104]);
    b.build()
}

fn main() {
    dry_run(&graph()).expect("example");
}
