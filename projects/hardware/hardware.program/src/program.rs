use serde::{Deserialize, Serialize};

use crate::stmt::Stmt;
use crate::types::Ty;

/// 单函数定义。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Function {
    pub name: String,
    pub params: Vec<(u32, Ty)>,
    pub locals: Vec<(u32, Ty)>,
    pub body: Vec<Stmt>,
    pub ret: Ty,
}

/// 完整 UCF 程序（可序列化、可哈希）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Program {
    pub version: u32,
    pub functions: Vec<Function>,
    pub entry: usize,
}

/// 最小并行绑定（L2）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ParallelDim {
    Scalar,
    Global1D { count: u32 },
}

/// 程序 + 并行范围 + uniform 布局占位。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KernelBinding {
    pub program: Program,
    pub parallel: ParallelDim,
}

impl Program {
    pub fn entry_fn(&self) -> Option<&Function> {
        self.functions.get(self.entry)
    }
}
