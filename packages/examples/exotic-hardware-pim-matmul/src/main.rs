use ucf::prelude::*;

fn graph() -> Graph {
    let mut b = GraphBuilder::new();
    b.tensor(1, Domain::Hbm);
    b.tensor(2, Domain::Vram);
    b.tensor(3, Domain::Vram);
    b.executor(101, "Transfer", "Link", TaskKind::Copy, Objective::MaxThroughput, Priority::Batch);
    b.executor(102, "MatVec", "PimArray", TaskKind::MatMul, Objective::MaxThroughput, Priority::Batch);
    b.executor(103, "Transfer", "Link", TaskKind::Copy, Objective::MaxThroughput, Priority::Batch);
    b.chain(&[101, 102, 103]);
    b.build()
}

fn main() {
    dry_run(&graph()).expect("example");
}
