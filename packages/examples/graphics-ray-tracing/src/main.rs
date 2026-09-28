use ucf::prelude::*;

fn graph() -> Graph {
    let mut b = GraphBuilder::new();
    b.buffer(1, Domain::Vram);
    b.buffer(2, Domain::Vram);
    b.texture(3, Domain::Vram);
    b.gpu(101, "RayGen");
    b.gpu(102, "Traverse");
    b.gpu(103, "ClosestHit");
    b.chain(&[101, 102, 103]);
    b.build()
}

fn main() {
    dry_run(&graph()).expect("example");
}
