use ucf::prelude::*;

fn graph() -> Graph {
    let mut b = GraphBuilder::new();
    b.buffer(1, Domain::Vram);
    b.buffer(2, Domain::Vram);
    b.buffer(3, Domain::Vram);
    b.gpu(101, "FFTStage");
    b.gpu(102, "BitReverse");
    b.chain(&[101, 102]);
    b.build()
}

fn main() {
    dry_run(&graph()).expect("example");
}
