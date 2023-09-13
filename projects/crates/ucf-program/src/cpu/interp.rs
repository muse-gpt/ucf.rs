use std::collections::HashMap;

use crate::error::ProgramError;
use crate::expr::{BinOp, Expr, UnOp};
use crate::program::{Function, Program};
use crate::stmt::Stmt;
use crate::validate::validate_program;

use super::value::Value;

/// 单实例并行上下文（L2）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParallelInstance {
    pub global_id: u32,
}

/// 单次解释执行结果。
#[derive(Debug, Clone, PartialEq)]
pub struct CpuExecResult {
    pub value: Value,
    pub loop_steps: u32,
}

/// CPU 语义参考解释器（F32 主路径）。
pub struct CpuInterpreter {
    uniforms: HashMap<u32, Value>,
    max_while: u32,
}

impl CpuInterpreter {
    pub fn new() -> Self {
        Self {
            uniforms: HashMap::new(),
            max_while: 1_000_000,
        }
    }

    pub fn with_uniform(&mut self, slot: u32, value: Value) -> &mut Self {
        self.uniforms.insert(slot, value);
        self
    }

    pub fn run(&mut self, program: &Program) -> Result<CpuExecResult, ProgramError> {
        validate_program(program)?;
        let func = program
            .entry_fn()
            .ok_or(ValidateError::EmptyProgram)?;
        let mut locals = HashMap::new();
        let mut loop_steps = 0u32;
        let value = self.exec_function(func, program, &mut locals, &mut loop_steps, None)?;
        Ok(CpuExecResult { value, loop_steps })
    }

    pub fn run_parallel(
        &mut self,
        program: &Program,
        count: u32,
    ) -> Result<Vec<CpuExecResult>, ProgramError> {
        let mut out = Vec::with_capacity(count as usize);
        for global_id in 0..count {
            let func = program.entry_fn().ok_or(ValidateError::EmptyProgram)?;
            let mut locals = HashMap::new();
            let mut loop_steps = 0u32;
            let inst = ParallelInstance { global_id };
            let value =
                self.exec_function(func, program, &mut locals, &mut loop_steps, Some(inst))?;
            out.push(CpuExecResult { value, loop_steps });
        }
        Ok(out)
    }

    fn exec_function(
        &mut self,
        func: &Function,
        program: &Program,
        locals: &mut HashMap<u32, Value>,
        loop_steps: &mut u32,
        parallel: Option<ParallelInstance>,
    ) -> Result<Value, ProgramError> {
        match self.exec_stmts(&func.body, program, locals, loop_steps, parallel, false)? {
            Some(v) => Ok(v),
            None => Ok(Value::F32(0.0)),
        }
    }

    fn exec_stmts(
        &mut self,
        body: &[Stmt],
        program: &Program,
        locals: &mut HashMap<u32, Value>,
        loop_steps: &mut u32,
        parallel: Option<ParallelInstance>,
        in_loop: bool,
    ) -> Result<Option<Value>, ProgramError> {
        let mut idx = 0;
        while idx < body.len() {
            match &body[idx] {
                Stmt::Let { id, init, .. } => {
                    let v = self.eval_expr(init, program, locals, parallel)?;
                    locals.insert(*id, v);
                }
                Stmt::StoreLocal { id, value } => {
                    let v = self.eval_expr(value, program, locals, parallel)?;
                    locals.insert(*id, v);
                }
                Stmt::If { cond, then_body, else_body } => {
                    let c = self.eval_expr(cond, program, locals, parallel)?;
                    let branch = if c.as_bool().map_err(|_| type_mismatch("bool", &c))? {
                        then_body
                    } else {
                        else_body
                    };
                    if let Some(v) =
                        self.exec_stmts(branch, program, locals, loop_steps, parallel, in_loop)?
                    {
                        return Ok(Some(v));
                    }
                }
                Stmt::For { iter, limit, body } => {
                    for step in 0..*limit {
                        *loop_steps += 1;
                        locals.insert(*iter, Value::U32(step));
                        if let Some(v) =
                            self.exec_stmts(body, program, locals, loop_steps, parallel, true)?
                        {
                            return Ok(Some(v));
                        }
                    }
                }
                Stmt::While { cond, limit, body } => {
                    for step in 0..*limit {
                        let c = self.eval_expr(cond, program, locals, parallel)?;
                        if !c.as_bool().map_err(|_| type_mismatch("bool", &c))? {
                            break;
                        }
                        *loop_steps += 1;
                        if step + 1 > self.max_while {
                            return Err(ProgramError::WhileLimitExceeded(self.max_while));
                        }
                        if let Some(v) =
                            self.exec_stmts(body, program, locals, loop_steps, parallel, true)?
                        {
                            return Ok(Some(v));
                        }
                    }
                }
                Stmt::Break => {
                    if !in_loop {
                        return Err(ProgramError::BadControlFlow);
                    }
                    return Ok(None);
                }
                Stmt::Continue => {
                    if !in_loop {
                        return Err(ProgramError::BadControlFlow);
                    }
                    idx += 1;
                    continue;
                }
                Stmt::Return(expr) => {
                    return Ok(Some(self.eval_expr(expr, program, locals, parallel)?));
                }
                Stmt::Expr(expr) => {
                    self.eval_expr(expr, program, locals, parallel)?;
                }
            }
            idx += 1;
        }
        Ok(None)
    }

    fn eval_expr(
        &mut self,
        expr: &Expr,
        program: &Program,
        locals: &HashMap<u32, Value>,
        parallel: Option<ParallelInstance>,
    ) -> Result<Value, ProgramError> {
        match expr {
            Expr::ConstF32(v) => Ok(Value::F32(*v)),
            Expr::ConstF64(v) => Ok(Value::F64(*v)),
            Expr::ConstBool(v) => Ok(Value::Bool(*v)),
            Expr::ConstU32(v) => Ok(Value::U32(*v)),
            Expr::LoadLocal(id) => locals
                .get(id)
                .cloned()
                .ok_or(ProgramError::TypeMismatch {
                    expected: format!("local {}", id),
                    actual: "missing".into(),
                }),
            Expr::LoadUniform(slot) => self
                .uniforms
                .get(slot)
                .cloned()
                .ok_or(ProgramError::MissingUniform(*slot)),
            Expr::GlobalId => Ok(Value::U32(parallel.map(|p| p.global_id).unwrap_or(0))),
            Expr::BinOp { op, lhs, rhs } => {
                let l = self.eval_expr(lhs, program, locals, parallel)?;
                let r = self.eval_expr(rhs, program, locals, parallel)?;
                self.eval_binop(*op, l, r)
            }
            Expr::UnOp { op, arg } => {
                let v = self.eval_expr(arg, program, locals, parallel)?;
                self.eval_unop(*op, v)
            }
            Expr::Select { cond, then_, else_ } => {
                let c = self.eval_expr(cond, program, locals, parallel)?;
                if c.as_bool().map_err(|_| type_mismatch("bool", &c))? {
                    self.eval_expr(then_, program, locals, parallel)
                } else {
                    self.eval_expr(else_, program, locals, parallel)
                }
            }
            Expr::Call { func, args } => {
                let fn_idx = *func as usize;
                let callee = program
                    .functions
                    .get(fn_idx)
                    .ok_or(ProgramError::UnknownFunction(*func))?;
                let mut call_locals = HashMap::new();
                for ((param_id, _), arg_expr) in callee.params.iter().zip(args.iter()) {
                    let v = self.eval_expr(arg_expr, program, locals, parallel)?;
                    call_locals.insert(*param_id, v);
                }
                let mut steps = 0u32;
                self.exec_function(callee, program, &mut call_locals, &mut steps, parallel)
            }
        }
    }

    fn eval_binop(&self, op: BinOp, l: Value, r: Value) -> Result<Value, ProgramError> {
        match op {
            BinOp::Dot => {
                let a = l.as_vec3().map_err(|_| ProgramError::UnsupportedOp("dot".into()))?;
                let b = r.as_vec3().map_err(|_| ProgramError::UnsupportedOp("dot".into()))?;
                Ok(Value::F64(a[0] * b[0] + a[1] * b[1] + a[2] * b[2]))
            }
            BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div => {
                if let (Ok(la), Ok(ra)) = (l.as_vec3(), r.as_vec3()) {
                    let out = match op {
                        BinOp::Add => [la[0] + ra[0], la[1] + ra[1], la[2] + ra[2]],
                        BinOp::Sub => [la[0] - ra[0], la[1] - ra[1], la[2] - ra[2]],
                        BinOp::Mul => [la[0] * ra[0], la[1] * ra[1], la[2] * ra[2]],
                        BinOp::Div => {
                            if ra[0] == 0.0 || ra[1] == 0.0 || ra[2] == 0.0 {
                                return Err(ProgramError::DivideByZero);
                            }
                            [la[0] / ra[0], la[1] / ra[1], la[2] / ra[2]]
                        }
                        _ => unreachable!(),
                    };
                    return Ok(Value::Vec3(out));
                }
                let la = l.as_f64().map_err(|_| ProgramError::UnsupportedOp("binop".into()))?;
                let ra = r.as_f64().map_err(|_| ProgramError::UnsupportedOp("binop".into()))?;
                let out = match op {
                    BinOp::Add => la + ra,
                    BinOp::Sub => la - ra,
                    BinOp::Mul => la * ra,
                    BinOp::Div => {
                        if ra == 0.0 {
                            return Err(ProgramError::DivideByZero);
                        }
                        la / ra
                    }
                    BinOp::Min => la.min(ra),
                    BinOp::Max => la.max(ra),
                    _ => return Err(ProgramError::UnsupportedOp(format!("{:?}", op))),
                };
                Ok(Value::F64(out))
            }
            BinOp::Min | BinOp::Max => {
                let la = l.as_f64().map_err(|_| ProgramError::UnsupportedOp("minmax".into()))?;
                let ra = r.as_f64().map_err(|_| ProgramError::UnsupportedOp("minmax".into()))?;
                Ok(Value::F64(match op {
                    BinOp::Min => la.min(ra),
                    BinOp::Max => la.max(ra),
                    _ => unreachable!(),
                }))
            }
            BinOp::Lt | BinOp::Le | BinOp::Eq | BinOp::Ne | BinOp::Ge | BinOp::Gt => {
                let la = l.as_f64().map_err(|_| ProgramError::UnsupportedOp("cmp".into()))?;
                let ra = r.as_f64().map_err(|_| ProgramError::UnsupportedOp("cmp".into()))?;
                let b = match op {
                    BinOp::Lt => la < ra,
                    BinOp::Le => la <= ra,
                    BinOp::Eq => la == ra,
                    BinOp::Ne => la != ra,
                    BinOp::Ge => la >= ra,
                    BinOp::Gt => la > ra,
                    _ => unreachable!(),
                };
                Ok(Value::Bool(b))
            }
        }
    }

    fn eval_unop(&self, op: UnOp, v: Value) -> Result<Value, ProgramError> {
        match op {
            UnOp::Neg => {
                if let Ok(x) = v.as_f64() {
                    Ok(Value::F64(-x))
                } else {
                    let a = v.as_vec3().map_err(|_| ProgramError::UnsupportedOp("neg".into()))?;
                    Ok(Value::Vec3([-a[0], -a[1], -a[2]]))
                }
            }
            UnOp::Sqrt => Ok(Value::F64(v.as_f64().map_err(|_| ProgramError::UnsupportedOp("sqrt".into()))?.sqrt())),
            UnOp::Sin => Ok(Value::F64(v.as_f64().map_err(|_| ProgramError::UnsupportedOp("sin".into()))?.sin())),
            UnOp::Cos => Ok(Value::F64(v.as_f64().map_err(|_| ProgramError::UnsupportedOp("cos".into()))?.cos())),
            UnOp::Atan => Ok(Value::F64(v.as_f64().map_err(|_| ProgramError::UnsupportedOp("atan".into()))?.atan())),
            UnOp::Abs => Ok(Value::F64(v.as_f64().map_err(|_| ProgramError::UnsupportedOp("abs".into()))?.abs())),
            UnOp::Length => {
                let a = v.as_vec3().map_err(|_| ProgramError::UnsupportedOp("length".into()))?;
                Ok(Value::F64((a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()))
            }
            UnOp::Normalize => {
                let a = v.as_vec3().map_err(|_| ProgramError::UnsupportedOp("normalize".into()))?;
                let len = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
                if len == 0.0 {
                    return Err(ProgramError::NormalizeZero);
                }
                Ok(Value::Vec3([a[0] / len, a[1] / len, a[2] / len]))
            }
        }
    }
}

use crate::error::ValidateError;

fn type_mismatch(expected: &str, actual: &Value) -> ProgramError {
    ProgramError::TypeMismatch {
        expected: expected.into(),
        actual: actual.type_name().into(),
    }
}
