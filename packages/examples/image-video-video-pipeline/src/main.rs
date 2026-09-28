use ucf::prelude::*;

fn graph() -> Graph {
    let mut b = GraphBuilder::new();
    b.buffer(1, Domain::Vram);
    b.texture(2, Domain::Vram);
    b.texture(3, Domain::Vram);
    b.buffer(4, Domain::Vram);
    b.gpu(101, "Decode");
    b.gpu(102, "Scale");
    b.gpu(103, "Encode");
    b.chain(&[101, 102, 103]);
    b.build()
}

fn main() {
    dry_run(&graph()).expect("example");
}
