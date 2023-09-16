use serde::{Deserialize, Serialize};

/// 二元运算（L0）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Min,
    Max,
    Mod,
    Dot,
    Lt,
    Le,
    Eq,
    Ne,
    Ge,
    Gt,
}

/// 一元运算（L0）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum UnOp {
    Neg,
    Sqrt,
    Sin,
    Cos,
    Atan,
    Length,
    Normalize,
    Abs,
}

/// 表达式节点。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Expr {
    ConstF32(f32),
    ConstF64(f64),
    ConstBool(bool),
    ConstU32(u32),
    /// 局部槽位（函数内 `Let` 绑定）。
    LoadLocal(u32),
    /// Uniform 槽位（内核绑定期填入）。
    LoadUniform(u32),
    /// 并行实例全局索引（L2）。
    GlobalId,
    BinOp {
        op: BinOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    UnOp {
        op: UnOp,
        arg: Box<Expr>,
    },
    Select {
        cond: Box<Expr>,
        then_: Box<Expr>,
        else_: Box<Expr>,
    },
    Call {
        func: u32,
        args: Vec<Expr>,
    },
}
