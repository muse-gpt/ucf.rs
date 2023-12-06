// 源自 TON blackhole 渲染器（MIT, © TON Team），按 MPL-2.0 分发。

//! 并行 ACES 色调映射程序。

use hardware_program::{BinOp, Expr, Program, ProgramBuilder, StructuredFunctionBuilder, Ty};

/// 构建并行线性 HDR 到显示 RGB 的 ACES tone-map 程序。
///
/// 输入 buffer `1` 和输出 buffer `2` 均按每实例三个 `f32` 通道布局。
#[must_use]
pub fn tone_map_aces_batch() -> Program {
    let mut builder = StructuredFunctionBuilder::new("tone_map_aces_batch");
    let global = builder.global_id();
    let input_base = builder.binary(BinOp::Mul, global.clone(), builder.const_u32(3));
    let output_base = builder.binary(BinOp::Mul, global, builder.const_u32(3));
    let zero = builder.const_f32(0.0);
    let one = builder.const_f32(1.0);
    let channel = |builder: &mut StructuredFunctionBuilder, input: Expr| {
        let x = builder.binary(BinOp::Max, input, zero.clone());
        let numerator = builder.binary(
            BinOp::Mul,
            x.clone(),
            builder.binary(BinOp::Add, builder.binary(BinOp::Mul, builder.const_f32(2.51), x.clone()), builder.const_f32(0.03)),
        );
        let denominator = builder.binary(
            BinOp::Add,
            builder.binary(BinOp::Mul, x.clone(), builder.binary(BinOp::Add, builder.binary(BinOp::Mul, builder.const_f32(2.43), x), builder.const_f32(0.59))),
            builder.const_f32(0.14),
        );
        builder.binary(BinOp::Min, one.clone(), builder.binary(BinOp::Max, zero.clone(), builder.binary(BinOp::Div, numerator, denominator)))
    };
    for component in 0..3u32 {
        let input = builder.load_buffer(1, builder.binary(BinOp::Add, input_base.clone(), builder.const_u32(component)));
        let mapped = channel(&mut builder, input);
        builder.store_buffer(2, builder.binary(BinOp::Add, output_base.clone(), builder.const_u32(component)), mapped);
    }
    ProgramBuilder::new().structured_entry(builder, Ty::f32(), zero).expect("valid batch tone map builder").finish().expect("valid batch tone map program")
}
