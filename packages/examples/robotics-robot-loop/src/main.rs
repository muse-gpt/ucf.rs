use ucf::prelude::*;

fn graph() -> Graph {
    let mut b = GraphBuilder::new();
    b.buffer(1, Domain::Host);
    b.buffer(2, Domain::Vram);
    b.buffer(3, Domain::Host);
    b.buffer(4, Domain::Host);
    b.executor(101, "LidarDma", "Dpu", TaskKind::Copy, Objective::MinLatency { deadline_micros: 1_000 }, Priority::Streaming);
    b.gpu(102, "SlamKernel");
    b.executor(103, "Planner", "Cpu", TaskKind::Custom("Planner".into()), Objective::MinLatency { deadline_micros: 5_000 }, Priority::Interactive);
    b.executor(104, "Controller", "Mcu", TaskKind::Custom("Controller".into()), Objective::MinLatency { deadline_micros: 500 }, Priority::Interactive);
    b.chain(&[101, 102, 103, 104]);
    b.build()
}

fn main() {
    dry_run(&graph()).expect("example");
}
