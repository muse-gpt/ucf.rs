use ucf::prelude::*;

fn graph() -> Graph {
    let mut b = GraphBuilder::new();
    b.texture(1, Domain::Vram);
    b.texture(2, Domain::Vram);
    b.texture(3, Domain::Vram);
    b.gpu(101, "Downsample");
    b.gpu(102, "Blur");
    b.gpu(103, "Composite");
    b.gpu(104, "Tonemap");
    b.chain(&[101, 102, 103, 104]);
    b.build()
}

fn main() {
    dry_run(&graph()).expect("example");
}
