use ucf::prelude::*;

fn graph() -> Graph {
    let mut b = GraphBuilder::new();
    for id in 1..=3 { b.tensor(id, Domain::Vram); }
    b.gpu_matmul(101, "MatMul");
    b.gpu(102, "Backward");
    b.executor(103, "AllReduce", "NvLink", TaskKind::Custom("AllReduce".into()), Objective::MaxThroughput, Priority::Batch);
    b.gpu(104, "Adam");
    b.chain(&[101, 102, 103, 104]);
    b.build()
}

fn main() {
    dry_run(&graph()).expect("example");
}
