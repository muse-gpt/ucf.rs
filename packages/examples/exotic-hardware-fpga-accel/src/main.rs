use ucf::prelude::*;

fn graph() -> Graph {
    let mut b = GraphBuilder::new();
    b.buffer(1, Domain::Host);
    b.buffer(2, Domain::Host);
    b.buffer(3, Domain::Host);
    b.executor(101, "Synthesize", "Host", TaskKind::Custom("Synthesize".into()), Objective::MaxUtilization, Priority::Background);
    b.executor(102, "Configure", "Fpga", TaskKind::Custom("Configure".into()), Objective::MaxUtilization, Priority::Background);
    b.executor(103, "Run", "Fpga", TaskKind::Dispatch, Objective::MaxThroughput, Priority::Batch);
    b.chain(&[101, 102, 103]);
    b.build()
}

fn main() {
    dry_run(&graph()).expect("example");
}
