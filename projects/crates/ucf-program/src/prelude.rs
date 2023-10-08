//! 通用程序构造器。

use crate::expr::{BinOp, Expr, UnOp};
use crate::program::{Function, Program};
use crate::stmt::Stmt;
use crate::types::Ty;

fn c(v: f64) -> Expr {
    Expr::ConstF64(v)
}

fn bin(op: BinOp, lhs: Expr, rhs: Expr) -> Expr {
    Expr::BinOp { op, lhs: Box::new(lhs), rhs: Box::new(rhs) }
}

/// 构建向量归一化程序，输入为 uniform 0。
pub fn vector_math_normalize() -> Program {
    Program {
        version: 1,
        functions: vec![Function {
            name: "vector_math_normalize".into(),
            params: vec![],
            locals: vec![],
            body: vec![Stmt::Return(Expr::UnOp { op: UnOp::Normalize, arg: Box::new(Expr::LoadUniform(0)) })],
            ret: Ty::vec3(),
        }],
        entry: 0,
    }
}

/// 构建单步标量更新程序，输入为 uniform 0、1、2。
pub fn scalar_step_once() -> Program {
    Program {
        version: 1,
        functions: vec![Function {
            name: "scalar_step_once".into(),
            params: vec![],
            locals: vec![],
            body: vec![Stmt::Return(bin(
                BinOp::Add,
                Expr::LoadUniform(0),
                bin(BinOp::Mul, Expr::LoadUniform(1), Expr::LoadUniform(2)),
            ))],
            ret: Ty::f64(),
        }],
        entry: 0,
    }
}

/// 构建固定上限累加程序，输入为 uniform 0 的每步增量。
pub fn bounded_accumulate() -> Program {
    Program {
        version: 1,
        functions: vec![Function {
            name: "bounded_accumulate".into(),
            params: vec![],
            locals: vec![(0, Ty::f64())],
            body: vec![
                Stmt::Let { id: 0, ty: Ty::f64(), init: c(0.0) },
                Stmt::For {
                    iter: 1,
                    limit: 64,
                    body: vec![Stmt::StoreLocal { id: 0, value: bin(BinOp::Add, Expr::LoadLocal(0), Expr::LoadUniform(0)) }],
                },
                Stmt::Return(Expr::LoadLocal(0)),
            ],
            ret: Ty::f64(),
        }],
        entry: 0,
    }
}

/// 构建并行实例路径长度程序。
pub fn parallel_path_steps() -> Program {
    Program {
        version: 1,
        functions: vec![Function {
            name: "parallel_path_steps".into(),
            params: vec![],
            locals: vec![(0, Ty::f64()), (1, Ty::f64())],
            body: vec![
                Stmt::Let { id: 0, ty: Ty::f64(), init: c(0.0) },
                Stmt::Let { id: 1, ty: Ty::f64(), init: bin(BinOp::Add, bin(BinOp::Mod, Expr::GlobalId, c(16.0)), c(1.0)) },
                Stmt::For {
                    iter: 2,
                    limit: 32,
                    body: vec![Stmt::If {
                        cond: bin(BinOp::Lt, Expr::LoadLocal(2), Expr::LoadLocal(1)),
                        then_body: vec![Stmt::StoreLocal { id: 0, value: bin(BinOp::Add, Expr::LoadLocal(0), c(1.0)) }],
                        else_body: vec![],
                    }],
                },
                Stmt::Return(Expr::LoadLocal(0)),
            ],
            ret: Ty::f64(),
        }],
        entry: 0,
    }
}

/// 构建 ACES 拟合色调映射程序，输入为 uniform 0 的线性 HDR `Vec3`。
pub fn tone_map_aces() -> Program {
    fn channel(x: Expr) -> Expr {
        let x = bin(BinOp::Max, x, c(0.0));
        let num = bin(BinOp::Mul, x.clone(), bin(BinOp::Add, bin(BinOp::Mul, c(2.51), x.clone()), c(0.03)));
        let den =
            bin(BinOp::Add, bin(BinOp::Mul, x.clone(), bin(BinOp::Add, bin(BinOp::Mul, c(2.43), x.clone()), c(0.59))), c(0.14));
        bin(BinOp::Min, c(1.0), bin(BinOp::Max, c(0.0), bin(BinOp::Div, num, den)))
    }
    let color = Expr::LoadUniform(0);
    let comp = |index| Expr::UnOp { op: UnOp::Vec3Comp(index), arg: Box::new(color.clone()) };
    Program {
        version: 1,
        functions: vec![Function {
            name: "tone_map_aces".into(),
            params: vec![],
            locals: vec![],
            body: vec![Stmt::Return(Expr::Vec3Compose {
                x: Box::new(channel(comp(0))),
                y: Box::new(channel(comp(1))),
                z: Box::new(channel(comp(2))),
            })],
            ret: Ty::vec3(),
        }],
        entry: 0,
    }
}

/// 构建泛光亮部提取程序，输入为 uniform 0 的颜色、uniform 1 的阈值。
pub fn post_bloom_extract() -> Program {
    let color = Expr::LoadUniform(0);
    let comp = |index| Expr::UnOp { op: UnOp::Vec3Comp(index), arg: Box::new(color.clone()) };
    let lum = bin(
        BinOp::Add,
        bin(BinOp::Add, bin(BinOp::Mul, c(0.2126), comp(0)), bin(BinOp::Mul, c(0.7152), comp(1))),
        bin(BinOp::Mul, c(0.0722), comp(2)),
    );
    let scale = bin(BinOp::Div, bin(BinOp::Sub, lum.clone(), Expr::LoadUniform(1)), bin(BinOp::Max, lum.clone(), c(1.0e-9)));
    let bright = Expr::Vec3Compose {
        x: Box::new(bin(BinOp::Mul, comp(0), scale.clone())),
        y: Box::new(bin(BinOp::Mul, comp(1), scale.clone())),
        z: Box::new(bin(BinOp::Mul, comp(2), scale)),
    };
    Program {
        version: 1,
        functions: vec![Function {
            name: "post_bloom_extract".into(),
            params: vec![],
            locals: vec![],
            body: vec![Stmt::Return(Expr::Select {
                cond: Box::new(bin(BinOp::Gt, lum, Expr::LoadUniform(1))),
                then_: Box::new(bright),
                else_: Box::new(Expr::Vec3Compose { x: Box::new(c(0.0)), y: Box::new(c(0.0)), z: Box::new(c(0.0)) }),
            })],
            ret: Ty::vec3(),
        }],
        entry: 0,
    }
}
