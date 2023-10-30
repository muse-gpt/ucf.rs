use std::collections::HashMap;

use crate::error::ProgramError;
use crate::expr::{BinOp, Expr, UnOp};
use crate::program::{Function, Program};
use crate::stmt::{CfgInstruction, CfgTerminator, Stmt};
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

enum Flow {
    Next,
    Break,
    Continue,
    Return(Value),
}

/// CPU 语义参考解释器（F32 主路径）。
pub struct CpuInterpreter {
    uniforms: HashMap<u32, Value>,
    buffers: HashMap<u32, Vec<Value>>,
    max_while: u32,
}

impl CpuInterpreter {
    pub fn new() -> Self {
        Self { uniforms: HashMap::new(), buffers: HashMap::new(), max_while: 1_000_000 }
    }

    pub fn with_uniform(&mut self, slot: u32, value: Value) -> &mut Self {
        self.uniforms.insert(slot, value);
        self
    }

    pub fn with_buffer(&mut self, resource: u32, values: Vec<Value>) -> &mut Self {
        self.buffers.insert(resource, values);
        self
    }

    pub fn buffer(&self, resource: u32) -> Option<&[Value]> {
        self.buffers.get(&resource).map(Vec::as_slice)
    }

    pub fn resource_ids(&self) -> impl Iterator<Item = u32> + '_ {
        self.buffers.keys().copied()
    }

    pub fn run(&mut self, program: &Program) -> Result<CpuExecResult, ProgramError> {
        validate_program(program)?;
        let func = program.entry_fn().ok_or(ValidateError::EmptyProgram)?;
        let mut locals = HashMap::new();
        let mut loop_steps = 0u32;
        let value = self.exec_function(func, program, &mut locals, &mut loop_steps, None)?;
        Ok(CpuExecResult { value, loop_steps })
    }

    pub fn run_parallel(&mut self, program: &Program, count: u32) -> Result<Vec<CpuExecResult>, ProgramError> {
        validate_program(program)?;
        let mut out = Vec::with_capacity(count as usize);
        for global_id in 0..count {
            let func = program.entry_fn().ok_or(ValidateError::EmptyProgram)?;
            let mut locals = HashMap::new();
            let mut loop_steps = 0u32;
            let inst = ParallelInstance { global_id };
            let value = self.exec_function(func, program, &mut locals, &mut loop_steps, Some(inst))?;
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
        if func.body.len() == 1 {
            if let Stmt::Cfg { entry, blocks } = &func.body[0] {
                return self.exec_cfg(*entry, blocks, program, locals, loop_steps, parallel);
            }
        }
        match self.exec_stmts(&func.body, program, locals, loop_steps, parallel, false)? {
            Flow::Return(v) => Ok(v),
            Flow::Next => Ok(Value::F32(0.0)),
            Flow::Break | Flow::Continue => Err(ProgramError::BadControlFlow),
        }
    }

    fn exec_cfg(
        &mut self,
        entry: u32,
        blocks: &[crate::CfgBlock],
        _program: &Program,
        _locals: &mut HashMap<u32, Value>,
        loop_steps: &mut u32,
        parallel: Option<ParallelInstance>,
    ) -> Result<Value, ProgramError> {
        let mut values = HashMap::<u32, Value>::new();
        let mut block_id = entry;
        let mut transitions = 0u32;
        loop {
            transitions = transitions.saturating_add(1);
            if transitions > self.max_while {
                return Err(ProgramError::WhileLimitExceeded(self.max_while));
            }
            let block = blocks.iter().find(|block| block.id == block_id).ok_or(ProgramError::BadControlFlow)?;
            for instruction in &block.instructions {
                match instruction {
                    CfgInstruction::ConstF32 { result, value } => {
                        values.insert(*result, Value::F32(*value));
                    }
                    CfgInstruction::ConstF64 { result, value } => {
                        values.insert(*result, Value::F64(*value));
                    }
                    CfgInstruction::ConstU32 { result, value } => {
                        values.insert(*result, Value::U32(*value));
                    }
                    CfgInstruction::LoadUniform { result, slot, ty } => {
                        let value = self.uniforms.get(slot).cloned().ok_or(ProgramError::MissingUniform(*slot))?;
                        values.insert(*result, coerce_binding_value(value, ty)?);
                    }
                    CfgInstruction::LoadBuffer { result, resource, index, .. } => {
                        let index = value_index(values.get(index).cloned().ok_or_else(|| missing_ssa(*index))?)?;
                        let buffer = self.buffers.get(resource).ok_or(ProgramError::MissingBuffer(*resource))?;
                        let length = buffer.len();
                        let value = buffer.get(index).cloned().ok_or(ProgramError::BufferOutOfBounds {
                            resource: *resource,
                            index,
                            length,
                        })?;
                        values.insert(*result, value);
                    }
                    CfgInstruction::GlobalId { result } => {
                        values.insert(*result, Value::U32(parallel.map(|instance| instance.global_id).unwrap_or(0)));
                    }
                    CfgInstruction::Binary { result, op, lhs, rhs } => {
                        let lhs = values.get(lhs).cloned().ok_or_else(|| missing_ssa(*lhs))?;
                        let rhs = values.get(rhs).cloned().ok_or_else(|| missing_ssa(*rhs))?;
                        values.insert(*result, self.eval_binop(*op, lhs, rhs)?);
                    }
                    CfgInstruction::Unary { result, op, arg } => {
                        let arg = values.get(arg).cloned().ok_or_else(|| missing_ssa(*arg))?;
                        values.insert(*result, self.eval_unop(*op, arg)?);
                    }
                    CfgInstruction::Vec3Compose { result, x, y, z } => {
                        let x = values.get(x).cloned().ok_or_else(|| missing_ssa(*x))?;
                        let y = values.get(y).cloned().ok_or_else(|| missing_ssa(*y))?;
                        let z = values.get(z).cloned().ok_or_else(|| missing_ssa(*z))?;
                        values.insert(
                            *result,
                            Value::Vec3([
                                x.as_f64().map_err(|_| type_mismatch("f64", &x))?,
                                y.as_f64().map_err(|_| type_mismatch("f64", &y))?,
                                z.as_f64().map_err(|_| type_mismatch("f64", &z))?,
                            ]),
                        );
                    }
                    CfgInstruction::Select { result, cond, then_, else_ } => {
                        let condition = values.get(cond).cloned().ok_or_else(|| missing_ssa(*cond))?;
                        let selected =
                            if condition.as_bool().map_err(|_| type_mismatch("bool", &condition))? { then_ } else { else_ };
                        values.insert(*result, values.get(selected).cloned().ok_or_else(|| missing_ssa(*selected))?);
                    }
                    CfgInstruction::StoreBuffer { resource, index, value } => {
                        let index = value_index(values.get(index).cloned().ok_or_else(|| missing_ssa(*index))?)?;
                        let value = values.get(value).cloned().ok_or_else(|| missing_ssa(*value))?;
                        let buffer = self.buffers.get_mut(resource).ok_or(ProgramError::MissingBuffer(*resource))?;
                        let length = buffer.len();
                        *buffer.get_mut(index).ok_or(ProgramError::BufferOutOfBounds {
                            resource: *resource,
                            index,
                            length,
                        })? = value;
                    }
                }
            }
            match &block.terminator {
                CfgTerminator::Return(value) => return values.get(value).cloned().ok_or_else(|| missing_ssa(*value)),
                CfgTerminator::Branch { target, args } => {
                    let target_block = blocks.iter().find(|candidate| candidate.id == *target).ok_or(ProgramError::BadControlFlow)?;
                    bind_cfg_parameters(target_block, args, &mut values)?;
                    block_id = *target;
                }
                CfgTerminator::ConditionalBranch { condition, then_block, then_args, else_block, else_args } => {
                    let condition = values.get(condition).cloned().ok_or_else(|| missing_ssa(*condition))?;
                    let (target, args) = if condition.as_bool().map_err(|_| type_mismatch("bool", &condition))? {
                        (*then_block, then_args)
                    } else {
                        (*else_block, else_args)
                    };
                    let target_block = blocks.iter().find(|candidate| candidate.id == target).ok_or(ProgramError::BadControlFlow)?;
                    bind_cfg_parameters(target_block, args, &mut values)?;
                    block_id = target;
                }
            }
            *loop_steps = loop_steps.saturating_add(1);
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
    ) -> Result<Flow, ProgramError> {
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
                Stmt::StoreBuffer { resource, index, value } => {
                    let index = value_index(self.eval_expr(index, program, locals, parallel)?)?;
                    let value = self.eval_expr(value, program, locals, parallel)?;
                    let buffer = self.buffers.get_mut(resource).ok_or(ProgramError::MissingBuffer(*resource))?;
                    let length = buffer.len();
                    let slot =
                        buffer.get_mut(index).ok_or(ProgramError::BufferOutOfBounds { resource: *resource, index, length })?;
                    *slot = value;
                }
                Stmt::If { cond, then_body, else_body } => {
                    let c = self.eval_expr(cond, program, locals, parallel)?;
                    let branch = if c.as_bool().map_err(|_| type_mismatch("bool", &c))? { then_body } else { else_body };
                    match self.exec_stmts(branch, program, locals, loop_steps, parallel, in_loop)? {
                        Flow::Next => {}
                        flow => return Ok(flow),
                    }
                }
                Stmt::For { iter, limit, body } => {
                    for step in 0..*limit {
                        *loop_steps += 1;
                        locals.insert(*iter, Value::U32(step));
                        match self.exec_stmts(body, program, locals, loop_steps, parallel, true)? {
                            Flow::Next | Flow::Continue => {}
                            Flow::Break => break,
                            Flow::Return(v) => return Ok(Flow::Return(v)),
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
                        match self.exec_stmts(body, program, locals, loop_steps, parallel, true)? {
                            Flow::Next | Flow::Continue => {}
                            Flow::Break => break,
                            Flow::Return(v) => return Ok(Flow::Return(v)),
                        }
                    }
                }
                Stmt::Break => {
                    if !in_loop {
                        return Err(ProgramError::BadControlFlow);
                    }
                    return Ok(Flow::Break);
                }
                Stmt::Continue => {
                    if !in_loop {
                        return Err(ProgramError::BadControlFlow);
                    }
                    return Ok(Flow::Continue);
                }
                Stmt::Return(expr) => {
                    return Ok(Flow::Return(self.eval_expr(expr, program, locals, parallel)?));
                }
                Stmt::Expr(expr) => {
                    self.eval_expr(expr, program, locals, parallel)?;
                }
                Stmt::Cfg { .. } => {
                    return Err(ProgramError::UnsupportedOp("nested CFG statement".into()));
                }
            }
            idx += 1;
        }
        Ok(Flow::Next)
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
                .ok_or(ProgramError::TypeMismatch { expected: format!("local {}", id), actual: "missing".into() }),
            Expr::LoadUniform(slot) => self.uniforms.get(slot).cloned().ok_or(ProgramError::MissingUniform(*slot)),
            Expr::LoadBuffer { resource, index } => {
                let index = value_index(self.eval_expr(index, program, locals, parallel)?)?;
                let buffer = self.buffers.get(resource).ok_or(ProgramError::MissingBuffer(*resource))?;
                let length = buffer.len();
                buffer.get(index).cloned().ok_or(ProgramError::BufferOutOfBounds { resource: *resource, index, length })
            }
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
            Expr::Vec3Compose { x, y, z } => {
                let x = self.eval_expr(x, program, locals, parallel)?;
                let y = self.eval_expr(y, program, locals, parallel)?;
                let z = self.eval_expr(z, program, locals, parallel)?;
                Ok(Value::Vec3([
                    x.as_f64().map_err(|_| type_mismatch("f64", &x))?,
                    y.as_f64().map_err(|_| type_mismatch("f64", &y))?,
                    z.as_f64().map_err(|_| type_mismatch("f64", &z))?,
                ]))
            }
            Expr::Call { func, args } => {
                let fn_idx = *func as usize;
                let callee = program.functions.get(fn_idx).ok_or(ProgramError::UnknownFunction(*func))?;
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
        if matches!((&l, &r), (Value::U32(_), Value::U32(0))) && matches!(op, BinOp::Div | BinOp::Mod) {
            return Err(ProgramError::DivideByZero);
        }
        if let (Value::F32(left), Value::F32(right)) = (&l, &r) {
            let result = match op {
                BinOp::Add => Some(*left + *right),
                BinOp::Sub => Some(*left - *right),
                BinOp::Mul => Some(*left * *right),
                BinOp::Div => Some(*left / *right),
                BinOp::Min => Some(left.min(*right)),
                BinOp::Max => Some(left.max(*right)),
                BinOp::Mod => Some(*left % *right),
                BinOp::Pow => Some(left.powf(*right)),
                _ => None,
            };
            if let Some(result) = result { return Ok(Value::F32(result)); }
        }
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
                if let (Ok(la), Ok(rs)) = (l.as_vec3(), r.as_f64()) {
                    let out = match op {
                        BinOp::Mul => [la[0] * rs, la[1] * rs, la[2] * rs],
                        BinOp::Div => {
                            if rs == 0.0 {
                                return Err(ProgramError::DivideByZero);
                            }
                            [la[0] / rs, la[1] / rs, la[2] / rs]
                        }
                        _ => return Err(ProgramError::UnsupportedOp(format!("{:?}", op))),
                    };
                    return Ok(Value::Vec3(out));
                }
                if let (Ok(ls), Ok(ra)) = (l.as_f64(), r.as_vec3()) {
                    let out = match op {
                        BinOp::Mul => [ls * ra[0], ls * ra[1], ls * ra[2]],
                        _ => return Err(ProgramError::UnsupportedOp(format!("{:?}", op))),
                    };
                    return Ok(Value::Vec3(out));
                }
                let la = l.as_f64().map_err(|_| ProgramError::UnsupportedOp("binop".into()))?;
                let ra = r.as_f64().map_err(|_| ProgramError::UnsupportedOp("binop".into()))?;
                let out = match op {
                    BinOp::Add => la + ra,
                    BinOp::Sub => la - ra,
                    BinOp::Mul => la * ra,
                    BinOp::Div => la / ra,
                    BinOp::Min => la.min(ra),
                    BinOp::Max => la.max(ra),
                    _ => return Err(ProgramError::UnsupportedOp(format!("{:?}", op))),
                };
                Ok(Value::F64(out))
            }
            BinOp::Min | BinOp::Max | BinOp::Mod | BinOp::Pow => {
                let la = l.as_f64().map_err(|_| ProgramError::UnsupportedOp("minmax".into()))?;
                let ra = r.as_f64().map_err(|_| ProgramError::UnsupportedOp("minmax".into()))?;
                Ok(Value::F64(match op {
                    BinOp::Min => la.min(ra),
                    BinOp::Max => la.max(ra),
                    BinOp::Mod => la % ra,
                    BinOp::Pow => la.powf(ra),
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
        if let Value::F32(value) = &v {
            let result = match op {
                UnOp::Neg => Some(-*value),
                UnOp::Sqrt => Some(value.sqrt()),
                UnOp::Sin => Some(value.sin()),
                UnOp::Cos => Some(value.cos()),
                UnOp::Atan => Some(value.atan()),
                UnOp::Abs => Some(value.abs()),
                UnOp::Ln => Some(value.ln()),
                _ => None,
            };
            if let Some(result) = result { return Ok(Value::F32(result)); }
        }
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
            UnOp::Ln => Ok(Value::F64(v.as_f64().map_err(|_| ProgramError::UnsupportedOp("ln".into()))?.ln())),
            UnOp::IsFinite => Ok(Value::Bool(v.as_f64().map_err(|_| ProgramError::UnsupportedOp("is_finite".into()))?.is_finite())),
            UnOp::Vec3Comp(axis) => {
                let a = v.as_vec3().map_err(|_| ProgramError::UnsupportedOp("vec3comp".into()))?;
                let idx = axis as usize;
                if idx > 2 {
                    return Err(ProgramError::UnsupportedOp("vec3comp axis".into()));
                }
                Ok(Value::F64(a[idx]))
            }
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
    ProgramError::TypeMismatch { expected: expected.into(), actual: actual.type_name().into() }
}

fn coerce_binding_value(value: Value, ty: &crate::Ty) -> Result<Value, ProgramError> {
    use crate::{ScalarTy, Ty};
    match (ty, &value) {
        (Ty::Scalar(ScalarTy::F32), Value::F32(_) | Value::F64(_) | Value::U32(_)) => Ok(Value::F32(value.as_f64().unwrap() as f32)),
        (Ty::Scalar(ScalarTy::F64), Value::F32(_) | Value::F64(_) | Value::U32(_)) => Ok(Value::F64(value.as_f64().unwrap())),
        (Ty::Scalar(ScalarTy::U32), Value::U32(_)) | (Ty::Scalar(ScalarTy::Bool), Value::Bool(_)) | (Ty::Vec3, Value::Vec3(_)) => Ok(value),
        (Ty::Scalar(ScalarTy::U32), Value::F32(_) | Value::F64(_)) => {
            let scalar = value.as_f64().unwrap();
            if scalar.is_finite() && (0.0..=u32::MAX as f64).contains(&scalar) && scalar.fract() == 0.0 {
                Ok(Value::U32(scalar as u32))
            } else { Err(type_mismatch("u32 binding", &value)) }
        }
        _ => Err(type_mismatch(&format!("{ty:?} binding"), &value)),
    }
}

fn missing_ssa(id: u32) -> ProgramError {
    ProgramError::TypeMismatch { expected: format!("ssa value {id}"), actual: "missing".into() }
}

fn bind_cfg_parameters(
    block: &crate::CfgBlock,
    args: &[u32],
    values: &mut HashMap<u32, Value>,
) -> Result<(), ProgramError> {
    if block.parameters.len() != args.len() {
        return Err(ProgramError::BadControlFlow);
    }
    let incoming = args
        .iter()
        .map(|argument| values.get(argument).cloned().ok_or_else(|| missing_ssa(*argument)))
        .collect::<Result<Vec<_>, _>>()?;
    for (parameter, value) in block.parameters.iter().zip(incoming) {
        values.insert(parameter.value, value);
    }
    Ok(())
}

fn value_index(value: Value) -> Result<usize, ProgramError> {
    match value {
        Value::U32(index) => Ok(index as usize),
        Value::F64(index) if index.is_finite() && index >= 0.0 && index.fract() == 0.0 => Ok(index as usize),
        Value::F32(index) if index.is_finite() && index >= 0.0 && index.fract() == 0.0 => Ok(index as usize),
        other => Err(type_mismatch("non-negative integer index", &other)),
    }
}
