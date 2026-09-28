use ucf::prelude::*;

fn graph() -> Graph {
    let mut b = GraphBuilder::new();
    b.buffer(1, Domain::Vram);
    b.buffer(2, Domain::Vram);
    b.buffer(3, Domain::Vram);
    b.buffer(4, Domain::Vram);
    b.gpu(101, "Predict");
    b.gpu(102, "SolveSprings");
    b.gpu(103, "Collide");
    b.gpu(104, "Update");
    b.chain(&[101, 102, 103, 104]);
    b.build()
}

fn main() {
    dry_run(&graph()).expect("example");
}
