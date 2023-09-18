//! UCF 通用程序 IR 与 CPU 参考解释器。

mod cpu;
mod error;
mod expr;
mod hash;
mod prelude;
mod program;
mod stmt;
mod types;
mod validate;

pub use cpu::{CpuExecResult, CpuInterpreter, ParallelInstance, Value};
pub use error::{ProgramError, ValidateError};
pub use expr::{BinOp, Expr, UnOp};
pub use hash::program_hash;
pub use prelude::{
    geodesic_capture_classify, geodesic_radial_infall_steps, orbit_loop_accumulate,
    orbit_step_once, ray_batch_steps, vector_math_normalize,
};
pub use program::{Function, KernelBinding, ParallelDim, Program};
pub use stmt::Stmt;
pub use types::{ScalarTy, Ty};
pub use validate::validate_program;
