//! UCF 通用程序 IR 与 CPU 参考解释器。

mod builder;
mod cpu;
mod error;
mod expr;
mod hash;
mod prelude;
mod program;
mod stmt;
mod types;
mod validate;

pub use builder::{BlockId, BuildError, FunctionBuilder, ProgramBuilder, StructuredFunctionBuilder, Terminator, ValueId};
pub use cpu::{CpuExecResult, CpuInterpreter, ParallelInstance, Value};
pub use error::{ProgramError, ValidateError};
pub use expr::{BinOp, Expr, UnOp};
pub use hash::program_hash;
pub use prelude::{
    bounded_accumulate, parallel_path_steps, post_bloom_extract, scalar_step_once, tone_map_aces, vector_math_normalize,
};
pub use program::{Function, KernelBinding, ParallelDim, Program};
pub use stmt::{CfgBlock, CfgInstruction, CfgParameter, CfgTerminator, Stmt};
pub use types::{ScalarTy, Ty};
pub use validate::validate_program;
