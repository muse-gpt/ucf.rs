use serde::{Deserialize, Serialize};

use crate::expr::Expr;
use crate::types::Ty;

/// SSA 指令。操作数使用函数内稳定的值编号，不携带领域语义。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CfgInstruction {
    ConstF32 { result: u32, value: f32 },
    ConstF64 { result: u32, value: f64 },
    ConstU32 { result: u32, value: u32 },
    LoadUniform { result: u32, slot: u32, ty: Ty },
    LoadBuffer { result: u32, resource: u32, index: u32, ty: Ty },
    GlobalId { result: u32 },
    Binary { result: u32, op: crate::BinOp, lhs: u32, rhs: u32 },
    Unary { result: u32, op: crate::UnOp, arg: u32 },
    Vec3Compose { result: u32, x: u32, y: u32, z: u32 },
    Select { result: u32, cond: u32, then_: u32, else_: u32 },
    StoreBuffer { resource: u32, index: u32, value: u32 },
}

/// SSA-CFG 基本块终结器。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CfgTerminator {
    Return(u32),
    Branch { target: u32, args: Vec<u32> },
    ConditionalBranch { condition: u32, then_block: u32, then_args: Vec<u32>, else_block: u32, else_args: Vec<u32> },
}

/// 基本块参数。跳转时由前驱提供对应实参，形成显式 phi 语义。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CfgParameter {
    pub value: u32,
    pub ty: Ty,
}

/// UCF 函数的基本块。它是通用程序 IR，不是领域 DSL。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CfgBlock {
    pub id: u32,
    pub parameters: Vec<CfgParameter>,
    pub instructions: Vec<CfgInstruction>,
    pub terminator: CfgTerminator,
}

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
    /// 向逻辑资源 buffer 写入一个元素。
    StoreBuffer {
        resource: u32,
        index: Expr,
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
    /// SSA-CFG 函数体。旧结构化语句和 CFG 程序可以共存，但不能互相伪装。
    Cfg {
        entry: u32,
        blocks: Vec<CfgBlock>,
    },
}
