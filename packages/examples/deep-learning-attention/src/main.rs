use ucf::prelude::*;
use ucf_example_kit::{GraphBuilder, dry_run};

fn graph() -> Graph {
    let mut b = GraphBuilder::new();
    b.tensor(1, Domain::Vram);
    b.tensor(2, Domain::Vram);
    b.tensor(3, Domain::Vram);
    b.tensor(4, Domain::Vram);
    b.tensor(5, Domain::Vram);
    b.gpu(101, "QKVProj");
    b.gpu(102, "AppendKV");
    b.gpu(103, "FlashAttention");
    b.chain(&[101, 102, 103]);
    b.build()
}

fn main() {
    dry_run(&graph()).expect("example");
}
