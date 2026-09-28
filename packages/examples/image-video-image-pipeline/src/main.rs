use ucf::prelude::*;

fn graph() -> Graph {
    let mut b = GraphBuilder::new();
    b.texture(1, Domain::Vram);
    b.texture(2, Domain::Vram);
    b.texture(3, Domain::Vram);
    b.gpu(101, "GaussianBlur");
    b.gpu(102, "UnsharpMask");
    b.gpu(103, "ToneCurve");
    b.chain(&[101, 102, 103]);
    b.build()
}

fn main() {
    dry_run(&graph()).expect("example");
}
