use ucf::prelude::*;

fn graph() -> Graph {
    let mut b = GraphBuilder::new();
    b.buffer(1, Domain::Vram);
    b.buffer(2, Domain::Vram);
    b.buffer(3, Domain::Vram);
    b.buffer(4, Domain::Vram);
    b.buffer(5, Domain::Vram);
    b.gpu(101, "NeighborList");
    b.gpu(102, "LennardJones");
    b.gpu(103, "Verlet");
    b.chain(&[101, 102, 103]);
    b.build()
}

fn main() {
    dry_run(&graph()).expect("example");
}
