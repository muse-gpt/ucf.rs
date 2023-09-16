//! 黑洞验证切片用的内建程序模板。

use crate::expr::{BinOp, Expr, UnOp};
use crate::program::{Function, Program};
use crate::stmt::Stmt;
use crate::types::Ty;

/// VectorMath：归一化 uniform 槽 0 的 `Vec3`。
pub fn vector_math_normalize() -> Program {
    let body = vec![
        Stmt::Let {
            id: 0,
            ty: Ty::vec3(),
            init: Expr::LoadUniform(0),
        },
        Stmt::Return(Expr::UnOp {
            op: UnOp::Normalize,
            arg: Box::new(Expr::LoadLocal(0)),
        }),
    ];
    Program {
        version: 1,
        functions: vec![Function {
            name: "vector_math_normalize".into(),
            params: vec![],
            locals: vec![(0, Ty::vec3())],
            body,
            ret: Ty::vec3(),
        }],
        entry: 0,
    }
}

/// OrbitStep：单步半径更新 `r_next = r + dr * step_size`（uniform 0=r, 1=dr, 2=step）。
pub fn orbit_step_once() -> Program {
    let body = vec![
        Stmt::Let {
            id: 0,
            ty: Ty::f64(),
            init: Expr::BinOp {
                op: BinOp::Add,
                lhs: Box::new(Expr::LoadUniform(0)),
                rhs: Box::new(Expr::BinOp {
                    op: BinOp::Mul,
                    lhs: Box::new(Expr::LoadUniform(1)),
                    rhs: Box::new(Expr::LoadUniform(2)),
                }),
            },
        },
        Stmt::Return(Expr::LoadLocal(0)),
    ];
    Program {
        version: 1,
        functions: vec![Function {
            name: "orbit_step_once".into(),
            params: vec![],
            locals: vec![(0, Ty::f64())],
            body,
            ret: Ty::f64(),
        }],
        entry: 0,
    }
}

/// OrbitLoop：有上限累加 `sum += i`，`i` 从 0 到 `limit-1`（uniform 0=limit）。
pub fn orbit_loop_accumulate() -> Program {
    let body = vec![
        Stmt::Let {
            id: 0,
            ty: Ty::f64(),
            init: Expr::ConstF64(0.0),
        },
        Stmt::Let {
            id: 1,
            ty: Ty::f64(),
            init: Expr::LoadUniform(0),
        },
        Stmt::For {
            iter: 2,
            limit: 64,
            body: vec![
                Stmt::StoreLocal {
                    id: 0,
                    value: Expr::BinOp {
                        op: BinOp::Add,
                        lhs: Box::new(Expr::LoadLocal(0)),
                        rhs: Box::new(Expr::LoadLocal(2)),
                    },
                },
            ],
        },
        Stmt::Return(Expr::LoadLocal(0)),
    ];
    Program {
        version: 1,
        functions: vec![Function {
            name: "orbit_loop_accumulate".into(),
            params: vec![],
            locals: vec![(0, Ty::f64()), (1, Ty::f64())],
            body,
            ret: Ty::f64(),
        }],
        entry: 0,
    }
}

/// RayBatch：每实例步数 = `(global_id % 16) + 1`，验证并行实例不同路径长度。
pub fn ray_batch_steps() -> Program {
    let body = vec![
        Stmt::Let {
            id: 0,
            ty: Ty::f64(),
            init: Expr::ConstF64(0.0),
        },
        Stmt::Let {
            id: 1,
            ty: Ty::f64(),
            init: Expr::BinOp {
                op: BinOp::Add,
                lhs: Box::new(Expr::BinOp {
                    op: BinOp::Mod,
                    lhs: Box::new(Expr::GlobalId),
                    rhs: Box::new(Expr::ConstF64(16.0)),
                }),
                rhs: Box::new(Expr::ConstF64(1.0)),
            },
        },
        Stmt::For {
            iter: 2,
            limit: 32,
            body: vec![Stmt::If {
                cond: Expr::BinOp {
                    op: BinOp::Lt,
                    lhs: Box::new(Expr::LoadLocal(2)),
                    rhs: Box::new(Expr::LoadLocal(1)),
                },
                then_body: vec![Stmt::StoreLocal {
                    id: 0,
                    value: Expr::BinOp {
                        op: BinOp::Add,
                        lhs: Box::new(Expr::LoadLocal(0)),
                        rhs: Box::new(Expr::ConstF64(1.0)),
                    },
                }],
                else_body: vec![],
            }],
        },
        Stmt::Return(Expr::LoadLocal(0)),
    ];
    Program {
        version: 1,
        functions: vec![Function {
            name: "ray_batch_steps".into(),
            params: vec![],
            locals: vec![(0, Ty::f64()), (1, Ty::f64())],
            body,
            ret: Ty::f64(),
        }],
        entry: 0,
    }
}
