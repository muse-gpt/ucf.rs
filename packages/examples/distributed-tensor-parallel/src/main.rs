use ucf::prelude::*;

fn graph() -> Graph {
    let mut b = GraphBuilder::new();
    b.tensor(1, Domain::Vram);
    b.tensor(2, Domain::Vram);
    b.tensor(3, Domain::Vram);
    b.executor(101, "AllGather", "NvLink", TaskKind::Custom("AllGather".into()), Objective::MaxThroughput, Priority::Batch);
    b.gpu_matmul(102, "MatMul");
    b.executor(103, "ReduceScatter", "NvLink", TaskKind::Custom("ReduceScatter".into()), Objective::MaxThroughput, Priority::Batch);
    b.chain(&[101, 102, 103]);
    b.build()
}

fn main() {
    dry_run(&graph()).expect("example");
}
