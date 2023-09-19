//! 黑洞验证切片用的内建程序模板。

use crate::expr::{BinOp, Expr, UnOp};
use crate::program::{Function, Program};
use crate::stmt::Stmt;
use crate::types::Ty;

mod disk_expr {
    use super::*;

    pub fn uni(slot: u32) -> Expr {
        Expr::LoadUniform(slot)
    }
    pub fn loc(id: u32) -> Expr {
        Expr::LoadLocal(id)
    }
    pub fn cnst(v: f64) -> Expr {
        Expr::ConstF64(v)
    }
    pub fn bin(op: BinOp, lhs: Expr, rhs: Expr) -> Expr {
        Expr::BinOp {
            op,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        }
    }
    pub fn un(op: UnOp, arg: Expr) -> Expr {
        Expr::UnOp {
            op,
            arg: Box::new(arg),
        }
    }
    pub fn sel(cond: Expr, then_: Expr, else_: Expr) -> Expr {
        Expr::Select {
            cond: Box::new(cond),
            then_: Box::new(then_),
            else_: Box::new(else_),
        }
    }
    pub fn v3(x: Expr, y: Expr, z: Expr) -> Expr {
        Expr::Vec3Compose {
            x: Box::new(x),
            y: Box::new(y),
            z: Box::new(z),
        }
    }
    pub fn call(func: u32, args: Vec<Expr>) -> Expr {
        Expr::Call { func, args }
    }
    pub fn add(a: Expr, b: Expr) -> Expr {
        bin(BinOp::Add, a, b)
    }
    pub fn sub(a: Expr, b: Expr) -> Expr {
        bin(BinOp::Sub, a, b)
    }
    pub fn mul(a: Expr, b: Expr) -> Expr {
        bin(BinOp::Mul, a, b)
    }
    pub fn div(a: Expr, b: Expr) -> Expr {
        bin(BinOp::Div, a, b)
    }
    pub fn min(a: Expr, b: Expr) -> Expr {
        bin(BinOp::Min, a, b)
    }
    pub fn max(a: Expr, b: Expr) -> Expr {
        bin(BinOp::Max, a, b)
    }
    pub fn pow(a: Expr, b: Expr) -> Expr {
        bin(BinOp::Pow, a, b)
    }
    pub fn lt(a: Expr, b: Expr) -> Expr {
        bin(BinOp::Lt, a, b)
    }
    pub fn le(a: Expr, b: Expr) -> Expr {
        bin(BinOp::Le, a, b)
    }
    pub fn dot(a: Expr, b: Expr) -> Expr {
        bin(BinOp::Dot, a, b)
    }
    pub fn neg(a: Expr) -> Expr {
        un(UnOp::Neg, a)
    }
    pub fn sqrt(a: Expr) -> Expr {
        un(UnOp::Sqrt, a)
    }
    pub fn ln(a: Expr) -> Expr {
        un(UnOp::Ln, a)
    }
    pub fn normalize(a: Expr) -> Expr {
        un(UnOp::Normalize, a)
    }
    pub fn v3x(v: Expr) -> Expr {
        un(UnOp::Vec3Comp(0), v)
    }
    pub fn v3y(v: Expr) -> Expr {
        un(UnOp::Vec3Comp(1), v)
    }
    pub fn v3z(v: Expr) -> Expr {
        un(UnOp::Vec3Comp(2), v)
    }
    pub fn ge(a: Expr, b: Expr) -> Expr {
        bin(BinOp::Ge, a, b)
    }
    pub fn clamp01(x: Expr) -> Expr {
        min(cnst(1.0), max(cnst(0.0), x))
    }
}

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

/// GeodesicCapture：uniform 0=`b`，1=`b_c`；返回 `1.0` 当 `b < b_c`（史瓦西解析捕获）。
pub fn geodesic_capture_classify() -> Program {
    let body = vec![Stmt::Return(Expr::Select {
        cond: Box::new(Expr::BinOp {
            op: BinOp::Lt,
            lhs: Box::new(Expr::LoadUniform(0)),
            rhs: Box::new(Expr::LoadUniform(1)),
        }),
        then_: Box::new(Expr::ConstF64(1.0)),
        else_: Box::new(Expr::ConstF64(0.0)),
    })];
    Program {
        version: 1,
        functions: vec![Function {
            name: "geodesic_capture_classify".into(),
            params: vec![],
            locals: vec![],
            body,
            ret: Ty::f64(),
        }],
        entry: 0,
    }
}

/// GeodesicRadial：径向落向视界，uniform 0=`r`，1=`r_s`，2=`step_size`；捕获时返回步数，否则 `-1`。
pub fn geodesic_radial_infall_steps() -> Program {
    let body = vec![
        Stmt::Let {
            id: 0,
            ty: Ty::f64(),
            init: Expr::LoadUniform(0),
        },
        Stmt::Let {
            id: 1,
            ty: Ty::f64(),
            init: Expr::ConstF64(0.0),
        },
        Stmt::While {
            cond: Expr::BinOp {
                op: BinOp::Gt,
                lhs: Box::new(Expr::LoadLocal(0)),
                rhs: Box::new(Expr::LoadUniform(1)),
            },
            limit: 512,
            body: vec![
                Stmt::StoreLocal {
                    id: 0,
                    value: Expr::BinOp {
                        op: BinOp::Sub,
                        lhs: Box::new(Expr::LoadLocal(0)),
                        rhs: Box::new(Expr::LoadUniform(2)),
                    },
                },
                Stmt::StoreLocal {
                    id: 1,
                    value: Expr::BinOp {
                        op: BinOp::Add,
                        lhs: Box::new(Expr::LoadLocal(1)),
                        rhs: Box::new(Expr::ConstF64(1.0)),
                    },
                },
            ],
        },
        Stmt::If {
            cond: Expr::BinOp {
                op: BinOp::Le,
                lhs: Box::new(Expr::LoadLocal(0)),
                rhs: Box::new(Expr::LoadUniform(1)),
            },
            then_body: vec![Stmt::Return(Expr::LoadLocal(1))],
            else_body: vec![Stmt::Return(Expr::ConstF64(-1.0))],
        },
    ];
    Program {
        version: 1,
        functions: vec![Function {
            name: "geodesic_radial_infall_steps".into(),
            params: vec![],
            locals: vec![(0, Ty::f64()), (1, Ty::f64())],
            body,
            ret: Ty::f64(),
        }],
        entry: 0,
    }
}

/// DiskShade：Novikov-Thorne 薄盘 HDR 辐射；uniform 0=`r`，1=`doppler`，2=`r_in`，3=`r_out`，4=`M`，5=`mdot`。
pub fn disk_shade_novikov() -> Program {
    use disk_expr::*;

    let radius = uni(0);
    let doppler = uni(1);
    let r_in = uni(2);
    let r_out = uni(3);
    let mass = uni(4);
    let mdot = uni(5);
    let zero_v3 = v3(cnst(0.0), cnst(0.0), cnst(0.0));

    let temp_expr = mul(
        mul(
            div(mdot, mul(mass.clone(), mass)),
            pow(div(r_in.clone(), radius.clone()), cnst(0.75)),
        ),
        cnst(5000.0),
    );
    let brightness = pow(
        max(cnst(0.0), min(doppler, cnst(8.0))),
        cnst(3.0),
    );

    let t_le_66 = le(loc(1), cnst(66.0));
    let t_le_19 = le(loc(1), cnst(19.0));
    let cool_r = cnst(255.0);
    let cool_g = sub(
        mul(cnst(99.4708025861), ln(loc(1))),
        cnst(161.1195681661),
    );
    let cool_b = sel(
        t_le_19,
        cnst(0.0),
        sub(
            mul(cnst(138.5177312231), ln(sub(loc(1), cnst(10.0)))),
            cnst(305.0447927307),
        ),
    );
    let warm_r = mul(
        cnst(329.698727446),
        pow(sub(loc(1), cnst(60.0)), cnst(-0.1332047592)),
    );
    let warm_g = mul(
        cnst(288.1221695283),
        pow(sub(loc(1), cnst(60.0)), cnst(-0.0755148492)),
    );
    let warm_b = cnst(255.0);
    let ch_r = sel(t_le_66.clone(), cool_r, warm_r);
    let ch_g = sel(t_le_66.clone(), cool_g, warm_g);
    let ch_b = sel(t_le_66, cool_b, warm_b);
    let radiance = mul(
        v3(
            clamp01(div(ch_r, cnst(255.0))),
            clamp01(div(ch_g, cnst(255.0))),
            clamp01(div(ch_b, cnst(255.0))),
        ),
        loc(5),
    );

    let shade_body = vec![
        Stmt::Let {
            id: 0,
            ty: Ty::f64(),
            init: temp_expr,
        },
        Stmt::If {
            cond: le(loc(0), cnst(0.0)),
            then_body: vec![Stmt::Return(zero_v3.clone())],
            else_body: vec![],
        },
        Stmt::Let {
            id: 1,
            ty: Ty::f64(),
            init: mul(loc(0), cnst(0.01)),
        },
        Stmt::Let {
            id: 5,
            ty: Ty::f64(),
            init: brightness,
        },
        Stmt::Return(radiance),
    ];

    let body = vec![Stmt::If {
        cond: ge(radius.clone(), r_in.clone()),
        then_body: vec![Stmt::If {
            cond: le(radius, r_out),
            then_body: shade_body,
            else_body: vec![Stmt::Return(zero_v3.clone())],
        }],
        else_body: vec![Stmt::Return(zero_v3)],
    }];

    Program {
        version: 1,
        functions: vec![Function {
            name: "disk_shade_novikov".into(),
            params: vec![],
            locals: vec![
                (0, Ty::f64()),
                (1, Ty::f64()),
                (5, Ty::f64()),
            ],
            body,
            ret: Ty::vec3(),
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
