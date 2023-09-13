use serde::{Deserialize, Serialize};

use crate::expr::Expr;
use crate::types::Ty;

/// 结构化语句（L1）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Stmt {
    Let {
        id: u32,
        ty: Ty,
        init: Expr,
    },
    StoreLocal {
        id: u32,
        value: Expr,
    },
    If {
        cond: Expr,
        then_body: Vec<Stmt>,
        else_body: Vec<Stmt>,
    },
    /// 有上限 `for i in 0..limit`。
    For {
        iter: u32,
        limit: u32,
        body: Vec<Stmt>,
    },
    /// 有上限 `while cond && steps < limit`。
    While {
        cond: Expr,
        limit: u32,
        body: Vec<Stmt>,
    },
    Break,
    Continue,
    Return(Expr),
    Expr(Expr),
}
