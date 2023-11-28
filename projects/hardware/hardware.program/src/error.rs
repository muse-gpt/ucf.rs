use thiserror::Error;

/// 程序验证错误。
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ValidateError {
    #[error("empty program")]
    EmptyProgram,
    #[error("entry index {0} out of range")]
    BadEntry(usize),
    #[error("function `{name}` loop limit {limit} exceeds max {max}")]
    LoopLimitExceeded { name: String, limit: u32, max: u32 },
    #[error("function `{name}` references unknown local {id}")]
    UnknownLocal { name: String, id: u32 },
    #[error("function `{name}` references unknown function {index}")]
    UnknownFunction { name: String, index: u32 },
    #[error("function `{name}` call to {index} expects {expected} arguments, got {actual}")]
    CallArity { name: String, index: u32, expected: usize, actual: usize },
    #[error("function `{name}` uses break or continue outside a loop")]
    BadControlFlow { name: String },
    #[error("recursive function call detected at function {index}")]
    RecursiveCall { index: u32 },
}

/// CPU 解释执行错误。
#[derive(Debug, Clone, PartialEq, Error)]
pub enum ProgramError {
    #[error("validation: {0}")]
    Validate(#[from] ValidateError),
    #[error("type mismatch: expected {expected}, got {actual}")]
    TypeMismatch { expected: String, actual: String },
    #[error("divide by zero")]
    DivideByZero,
    #[error("normalize zero vector")]
    NormalizeZero,
    #[error("unsupported op `{0}` for type")]
    UnsupportedOp(String),
    #[error("unknown function index {0}")]
    UnknownFunction(u32),
    #[error("break/continue outside loop")]
    BadControlFlow,
    #[error("while limit {0} exceeded")]
    WhileLimitExceeded(u32),
    #[error("uniform slot {0} missing")]
    MissingUniform(u32),
    #[error("resource buffer {0} is missing")]
    MissingBuffer(u32),
    #[error("resource buffer {resource} index {index} is out of bounds for length {length}")]
    BufferOutOfBounds { resource: u32, index: usize, length: usize },
}
