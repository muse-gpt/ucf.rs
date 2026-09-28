use ucf::prelude::*;

fn graph() -> Graph {
    let mut b = GraphBuilder::new();
    b.tensor(1, Domain::Vram);
    b.tensor(2, Domain::Vram);
    b.tensor(3, Domain::Vram);
    b.tensor(4, Domain::Vram);
    b.gpu(101, "Im2Col");
    b.gpu_matmul(102, "MatMul");
    b.chain(&[101, 102]);
    b.build()
}

fn main() {
    dry_run(&graph()).expect("example");
}
