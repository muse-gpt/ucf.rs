use ucf::prelude::*;

fn graph() -> Graph {
    let mut b = GraphBuilder::new();
    b.buffer(1, Domain::Vram);
    b.buffer(2, Domain::Vram);
    b.texture(3, Domain::Vram);
    b.texture(4, Domain::Vram);
    b.gpu_raster(101, "CullShader");
    b.gpu_raster(102, "VertexShader");
    b.gpu_raster(103, "Rasterize");
    b.gpu_raster(104, "PixelShader");
    b.chain(&[101, 102, 103, 104]);
    b.build()
}

fn main() {
    dry_run(&graph()).expect("example");
}
