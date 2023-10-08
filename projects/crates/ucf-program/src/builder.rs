//! UCF 程序构造器。
//!
//! 构造器管理函数、指令、值编号和终结器。builder 的输出始终是原生
//! SSA-CFG，不把 `Expr`/`Stmt` 包装成领域 DSL。

use crate::{
    BinOp, CfgBlock, CfgInstruction, CfgParameter, CfgTerminator, Expr, Function, Program, Stmt, Ty, UnOp, ValidateError, validate_program,
};

/// 函数内 SSA 值的稳定编号。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ValueId(u32);

/// 基本块的稳定编号，供后续结构化控制流 lowering 使用。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(u32);

/// 基本块终结器。终结器不属于指令流，必须是每个可达基本块的最后一个构造动作。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Terminator {
    /// 返回一个 SSA 值。
    Return(ValueId),
    /// 无条件跳转到目标基本块。
    Branch { target: BlockId, args: Vec<ValueId> },
    /// 根据布尔 SSA 值选择目标基本块。
    ConditionalBranch { condition: ValueId, then_block: BlockId, then_args: Vec<ValueId>, else_block: BlockId, else_args: Vec<ValueId> },
}

/// 程序构造错误。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BuildError {
    /// 程序没有入口函数。
    MissingEntry,
    /// 函数没有返回终结器。
    MissingTerminator { function: String },
    /// 当前块已经终止。
    BlockTerminated,
    /// 引用不存在的值。
    UnknownValue { value: u32 },
    /// 引用不存在的基本块。
    UnknownBlock { block: u32 },
    /// 指令输入类型不匹配。
    TypeMismatch { expected: Ty, actual: Ty },
    /// 当前兼容 lowering 尚不支持该控制流形态。
    UnsupportedControlFlow { function: String },
    /// builder 生成的程序没有通过 UCF 验证。
    InvalidProgram(ValidateError),
}

impl std::fmt::Display for BuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingEntry => f.write_str("program builder has no entry function"),
            Self::MissingTerminator { function } => write!(f, "function `{function}` has no return"),
            Self::BlockTerminated => f.write_str("current block is already terminated"),
            Self::UnknownValue { value } => write!(f, "unknown value {value}"),
            Self::UnknownBlock { block } => write!(f, "unknown block {block}"),
            Self::TypeMismatch { expected, actual } => write!(f, "type mismatch: expected {expected:?}, got {actual:?}"),
            Self::UnsupportedControlFlow { function } => write!(f, "unsupported control flow in `{function}`"),
            Self::InvalidProgram(error) => write!(f, "invalid UCF program: {error}"),
        }
    }
}

impl std::error::Error for BuildError {}

#[derive(Debug, Clone)]
enum Instruction {
    ConstF32 { result: ValueId, value: f32 },
    ConstF64 { result: ValueId, value: f64 },
    ConstU32 { result: ValueId, value: u32 },
    LoadUniform { result: ValueId, slot: u32, ty: Ty },
    LoadBuffer { result: ValueId, resource: u32, index: ValueId, ty: Ty },
    GlobalId { result: ValueId },
    Binary { result: ValueId, op: BinOp, lhs: ValueId, rhs: ValueId },
    Unary { result: ValueId, op: UnOp, arg: ValueId },
    Vec3Compose { result: ValueId, x: ValueId, y: ValueId, z: ValueId },
    Select { result: ValueId, cond: ValueId, then_: ValueId, else_: ValueId },
    StoreBuffer { resource: u32, index: ValueId, value: ValueId },
}

#[derive(Debug, Clone)]
struct BasicBlock {
    parameters: Vec<(ValueId, Ty)>,
    instructions: Vec<Instruction>,
    terminator: Option<Terminator>,
}

/// 构造单个 UCF 函数。值只能通过 builder 产生，不能携带领域对象。
pub struct FunctionBuilder {
    name: String,
    blocks: Vec<BasicBlock>,
    current: BlockId,
    values: Vec<Ty>,
    return_value_id: Option<ValueId>,
}

/// 构造结构化 UCF 函数。它直接生成 `Stmt` 结构化控制流，不引入领域 DSL。
///
/// 表达式仍由通用 [`Expr`] 构造，builder 只负责局部变量、循环和返回终结器。
pub struct StructuredFunctionBuilder {
    name: String,
    body: Vec<Stmt>,
    locals: Vec<(u32, Ty)>,
    next_local: u32,
}

impl StructuredFunctionBuilder {
    /// 创建结构化函数 builder。
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into(), body: Vec::new(), locals: Vec::new(), next_local: 0 }
    }

    /// 声明局部变量并写入初值，返回局部槽位。
    pub fn let_value(&mut self, ty: Ty, init: Expr) -> u32 {
        let id = self.next_local;
        self.next_local = self.next_local.saturating_add(1);
        self.locals.push((id, ty.clone()));
        self.body.push(Stmt::Let { id, ty, init });
        id
    }

    /// 写入已有局部变量。
    pub fn store_local(&mut self, id: u32, value: Expr) {
        self.body.push(Stmt::StoreLocal { id, value });
    }

    /// 添加有界 `for` 循环。
    pub fn bounded_for(&mut self, limit: u32, build: impl FnOnce(&mut Self, u32)) {
        let iter = self.next_local;
        self.next_local += 1;
        let parent = std::mem::take(&mut self.body);
        build(self, iter);
        let body = std::mem::replace(&mut self.body, parent);
        self.body.push(Stmt::For { iter, limit, body });
    }

    /// 添加有界 `while` 循环。
    pub fn bounded_while(&mut self, cond: Expr, limit: u32, build: impl FnOnce(&mut Self)) {
        let parent = std::mem::take(&mut self.body);
        build(self);
        let body = std::mem::replace(&mut self.body, parent);
        self.body.push(Stmt::While { cond, limit, body });
    }

    /// 添加通用 buffer 写入。
    pub fn store_buffer(&mut self, resource: u32, index: Expr, value: Expr) {
        self.body.push(Stmt::StoreBuffer { resource, index, value });
    }

    /// 构造常量表达式。
    pub fn const_f32(&self, value: f32) -> Expr { Expr::ConstF32(value) }

    /// 构造无符号整数常量表达式。
    pub fn const_u32(&self, value: u32) -> Expr { Expr::ConstU32(value) }

    /// 构造 uniform 读取表达式。
    pub fn load_uniform(&self, slot: u32) -> Expr { Expr::LoadUniform(slot) }

    /// 构造并行实例索引表达式。
    pub fn global_id(&self) -> Expr { Expr::GlobalId }

    /// 构造 buffer 读取表达式。
    pub fn load_buffer(&self, resource: u32, index: Expr) -> Expr {
        Expr::LoadBuffer { resource, index: Box::new(index) }
    }

    /// 构造二元表达式。
    pub fn binary(&self, op: BinOp, lhs: Expr, rhs: Expr) -> Expr {
        Expr::BinOp { op, lhs: Box::new(lhs), rhs: Box::new(rhs) }
    }

    /// 构造一元表达式。
    pub fn unary(&self, op: UnOp, arg: Expr) -> Expr {
        Expr::UnOp { op, arg: Box::new(arg) }
    }

    /// 构造条件选择表达式。
    pub fn select(&self, cond: Expr, then_: Expr, else_: Expr) -> Expr {
        Expr::Select { cond: Box::new(cond), then_: Box::new(then_), else_: Box::new(else_) }
    }

    /// 添加返回表达式并完成函数。
    pub fn finish(mut self, ret: Ty, value: Expr) -> Function {
        self.body.push(Stmt::Return(value));
        Function { name: self.name, params: Vec::new(), locals: self.locals, body: self.body, ret }
    }
}

impl FunctionBuilder {
    /// 创建带有入口基本块的函数。
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            blocks: vec![BasicBlock { parameters: Vec::new(), instructions: Vec::new(), terminator: None }],
            current: BlockId(0),
            values: Vec::new(),
            return_value_id: None,
        }
    }

    /// 返回入口基本块。
    pub fn entry_block(&self) -> BlockId {
        BlockId(0)
    }

    /// 创建基本块。
    pub fn create_block(&mut self) -> BlockId {
        let id = BlockId(self.blocks.len() as u32);
        self.blocks.push(BasicBlock { parameters: Vec::new(), instructions: Vec::new(), terminator: None });
        id
    }

    /// 切换当前基本块。
    pub fn switch_to(&mut self, block: BlockId) -> Result<(), BuildError> {
        self.blocks.get(block.0 as usize).ok_or(BuildError::UnknownBlock { block: block.0 })?;
        self.current = block;
        Ok(())
    }

    /// 读取当前基本块的终结器，供构造阶段和诊断使用。
    pub fn current_terminator(&self) -> Option<&Terminator> {
        self.blocks[self.current.0 as usize].terminator.as_ref()
    }

    /// 为基本块声明一个 phi/block argument。
    pub fn block_parameter(&mut self, block: BlockId, ty: Ty) -> Result<ValueId, BuildError> {
        self.ensure_block_exists(block)?;
        let value = ValueId(self.values.len() as u32);
        self.values.push(ty.clone());
        self.blocks[block.0 as usize].parameters.push((value, ty));
        Ok(value)
    }

    /// 发出无条件跳转终结器。
    pub fn branch(&mut self, target: BlockId) -> Result<(), BuildError> {
        self.branch_with_args(target, Vec::new())
    }

    /// 发出携带 block arguments 的无条件跳转。
    pub fn branch_with_args(&mut self, target: BlockId, args: Vec<ValueId>) -> Result<(), BuildError> {
        self.ensure_block_exists(target)?;
        self.check_block_args(target, &args)?;
        self.set_terminator(Terminator::Branch { target, args })
    }

    /// 发出条件跳转终结器。
    pub fn conditional_branch(
        &mut self,
        condition: ValueId,
        then_block: BlockId,
        else_block: BlockId,
    ) -> Result<(), BuildError> {
        self.conditional_branch_with_args(condition, then_block, Vec::new(), else_block, Vec::new())
    }

    /// 发出携带 block arguments 的条件跳转。
    pub fn conditional_branch_with_args(
        &mut self,
        condition: ValueId,
        then_block: BlockId,
        then_args: Vec<ValueId>,
        else_block: BlockId,
        else_args: Vec<ValueId>,
    ) -> Result<(), BuildError> {
        self.ensure_block_exists(then_block)?;
        self.ensure_block_exists(else_block)?;
        let actual = self.value_type(condition)?.clone();
        let expected = Ty::Scalar(crate::ScalarTy::Bool);
        if actual != expected {
            return Err(BuildError::TypeMismatch { expected, actual });
        }
        self.check_block_args(then_block, &then_args)?;
        self.check_block_args(else_block, &else_args)?;
        self.set_terminator(Terminator::ConditionalBranch { condition, then_block, then_args, else_block, else_args })
    }

    /// 发出 `f32` 常量。
    pub fn emit_const_f32(&mut self, value: f32) -> Result<ValueId, BuildError> {
        self.emit_value(Ty::f32(), Instruction::ConstF32 { result: ValueId(0), value })
    }

    /// 发出 `f64` 常量。
    pub fn emit_const_f64(&mut self, value: f64) -> Result<ValueId, BuildError> {
        self.emit_value(Ty::f64(), Instruction::ConstF64 { result: ValueId(0), value })
    }

    /// 发出 `u32` 常量。
    pub fn emit_const_u32(&mut self, value: u32) -> Result<ValueId, BuildError> {
        self.emit_value(Ty::Scalar(crate::ScalarTy::U32), Instruction::ConstU32 { result: ValueId(0), value })
    }

    /// 发出 uniform 读取。
    pub fn emit_load_uniform(&mut self, slot: u32, ty: Ty) -> Result<ValueId, BuildError> {
        self.emit_value(ty.clone(), Instruction::LoadUniform { result: ValueId(0), slot, ty })
    }

    /// 发出并行实例索引读取。
    pub fn emit_global_id(&mut self) -> Result<ValueId, BuildError> {
        self.emit_value(Ty::Scalar(crate::ScalarTy::U32), Instruction::GlobalId { result: ValueId(0) })
    }

    /// 发出 `u32` 加法。
    pub fn emit_add_u32(&mut self, lhs: ValueId, rhs: ValueId) -> Result<ValueId, BuildError> {
        self.emit_binary(BinOp::Add, lhs, rhs)
    }

    /// 发出 `u32` 乘法。
    pub fn emit_mul_u32(&mut self, lhs: ValueId, rhs: ValueId) -> Result<ValueId, BuildError> {
        self.emit_binary(BinOp::Mul, lhs, rhs)
    }

    /// 发出逻辑资源 buffer 读取。
    pub fn emit_load_buffer(&mut self, resource: u32, index: ValueId, ty: Ty) -> Result<ValueId, BuildError> {
        let index_ty = self.value_type(index)?.clone();
        let expected = Ty::Scalar(crate::ScalarTy::U32);
        if index_ty != expected {
            return Err(BuildError::TypeMismatch { expected, actual: index_ty });
        }
        self.emit_value(ty.clone(), Instruction::LoadBuffer { result: ValueId(0), resource, index, ty })
    }

    /// 发出逻辑资源 buffer 写入。
    pub fn emit_store_buffer(&mut self, resource: u32, index: ValueId, value: ValueId) -> Result<(), BuildError> {
        let index_ty = self.value_type(index)?.clone();
        let expected = Ty::Scalar(crate::ScalarTy::U32);
        if index_ty != expected {
            return Err(BuildError::TypeMismatch { expected, actual: index_ty });
        }
        self.value_type(value)?;
        if self.blocks[self.current.0 as usize].terminator.is_some() {
            return Err(BuildError::BlockTerminated);
        }
        self.blocks[self.current.0 as usize].instructions.push(Instruction::StoreBuffer { resource, index, value });
        Ok(())
    }

    /// 发出二元指令并检查输入类型。
    pub fn emit_binary(&mut self, op: BinOp, lhs: ValueId, rhs: ValueId) -> Result<ValueId, BuildError> {
        let lhs_ty = self.value_type(lhs)?.clone();
        let rhs_ty = self.value_type(rhs)?.clone();
        let vector_scalar = matches!(
            (&lhs_ty, &rhs_ty),
            (Ty::Vec2 | Ty::Vec3 | Ty::Vec4, Ty::Scalar(crate::ScalarTy::F32 | crate::ScalarTy::F64))
                | (Ty::Scalar(crate::ScalarTy::F32 | crate::ScalarTy::F64), Ty::Vec2 | Ty::Vec3 | Ty::Vec4)
        );
        if lhs_ty != rhs_ty && !(vector_scalar && matches!(op, BinOp::Mul | BinOp::Div)) {
            return Err(BuildError::TypeMismatch { expected: lhs_ty, actual: rhs_ty });
        }
        let result_ty = match op {
            BinOp::Lt | BinOp::Le | BinOp::Eq | BinOp::Ne | BinOp::Ge | BinOp::Gt => Ty::Scalar(crate::ScalarTy::Bool),
            BinOp::Mul if matches!(lhs_ty, Ty::Vec2 | Ty::Vec3 | Ty::Vec4) => lhs_ty.clone(),
            BinOp::Mul if matches!(rhs_ty, Ty::Vec2 | Ty::Vec3 | Ty::Vec4) => rhs_ty.clone(),
            BinOp::Div if matches!(lhs_ty, Ty::Vec2 | Ty::Vec3 | Ty::Vec4) => lhs_ty.clone(),
            _ => lhs_ty,
        };
        self.emit_value(result_ty, Instruction::Binary { result: ValueId(0), op, lhs, rhs })
    }

    /// 发出一元指令。
    pub fn emit_unary(&mut self, op: UnOp, arg: ValueId, result_ty: Ty) -> Result<ValueId, BuildError> {
        self.value_type(arg)?;
        self.emit_value(result_ty, Instruction::Unary { result: ValueId(0), op, arg })
    }

    /// 由三个同类型标量构造 `Vec3`。
    pub fn emit_vec3(&mut self, x: ValueId, y: ValueId, z: ValueId) -> Result<ValueId, BuildError> {
        let x_ty = self.value_type(x)?.clone();
        let y_ty = self.value_type(y)?.clone();
        let z_ty = self.value_type(z)?.clone();
        if x_ty != y_ty {
            return Err(BuildError::TypeMismatch { expected: x_ty, actual: y_ty });
        }
        if x_ty != z_ty {
            return Err(BuildError::TypeMismatch { expected: x_ty, actual: z_ty });
        }
        self.emit_value(Ty::vec3(), Instruction::Vec3Compose { result: ValueId(0), x, y, z })
    }

    /// 读取 `Vec3` 的一个标量分量。
    pub fn emit_vec3_component(&mut self, value: ValueId, axis: u8, result_ty: Ty) -> Result<ValueId, BuildError> {
        if axis > 2 {
            return Err(BuildError::UnknownValue { value: axis as u32 });
        }
        if self.value_type(value)? != &Ty::vec3() {
            return Err(BuildError::TypeMismatch { expected: Ty::vec3(), actual: self.value_type(value)?.clone() });
        }
        self.emit_value(result_ty, Instruction::Unary { result: ValueId(0), op: UnOp::Vec3Comp(axis), arg: value })
    }

    /// 发出条件选择指令。
    pub fn emit_select(&mut self, cond: ValueId, then_: ValueId, else_: ValueId) -> Result<ValueId, BuildError> {
        let cond_ty = self.value_type(cond)?.clone();
        let then_ty = self.value_type(then_)?.clone();
        let else_ty = self.value_type(else_)?.clone();
        if cond_ty != Ty::Scalar(crate::ScalarTy::Bool) {
            return Err(BuildError::TypeMismatch { expected: Ty::Scalar(crate::ScalarTy::Bool), actual: cond_ty });
        }
        if then_ty != else_ty {
            return Err(BuildError::TypeMismatch { expected: then_ty, actual: else_ty });
        }
        self.emit_value(then_ty, Instruction::Select { result: ValueId(0), cond, then_, else_ })
    }

    /// 发出返回终结器。
    pub fn return_value(&mut self, value: ValueId) -> Result<(), BuildError> {
        self.value_type(value)?;
        self.set_terminator(Terminator::Return(value))
    }

    /// 完成函数并写入 UCF 的 SSA-CFG 程序表示。
    pub fn finish(self) -> Result<Function, BuildError> {
        if self.blocks.iter().any(|block| block.terminator.is_none()) {
            return Err(BuildError::MissingTerminator { function: self.name });
        }
        let value = self.return_value_id.ok_or_else(|| BuildError::MissingTerminator { function: self.name.clone() })?;
        let ret = self.values[value.0 as usize].clone();
        let blocks = self
            .blocks
            .iter()
            .enumerate()
            .map(|(id, block)| CfgBlock {
                id: id as u32,
                parameters: block.parameters.iter().map(|(value, ty)| CfgParameter { value: value.0, ty: ty.clone() }).collect(),
                instructions: block.instructions.iter().map(Self::lower_cfg_instruction).collect(),
                terminator: Self::lower_terminator(block.terminator.as_ref().expect("checked above")),
            })
            .collect();
        let body = vec![Stmt::Cfg { entry: 0, blocks }];
        let locals = self.values.iter().enumerate().map(|(id, ty)| (id as u32, ty.clone())).collect();
        Ok(Function { name: self.name, params: Vec::new(), locals, body, ret })
    }

    fn lower_cfg_instruction(instruction: &Instruction) -> CfgInstruction {
        match instruction {
            Instruction::ConstF32 { result, value } => CfgInstruction::ConstF32 { result: result.0, value: *value },
            Instruction::ConstF64 { result, value } => CfgInstruction::ConstF64 { result: result.0, value: *value },
            Instruction::ConstU32 { result, value } => CfgInstruction::ConstU32 { result: result.0, value: *value },
            Instruction::LoadUniform { result, slot, ty } => {
                CfgInstruction::LoadUniform { result: result.0, slot: *slot, ty: ty.clone() }
            }
            Instruction::LoadBuffer { result, resource, index, ty } => {
                CfgInstruction::LoadBuffer { result: result.0, resource: *resource, index: index.0, ty: ty.clone() }
            }
            Instruction::GlobalId { result } => CfgInstruction::GlobalId { result: result.0 },
            Instruction::Binary { result, op, lhs, rhs } => {
                CfgInstruction::Binary { result: result.0, op: *op, lhs: lhs.0, rhs: rhs.0 }
            }
            Instruction::Unary { result, op, arg } => CfgInstruction::Unary { result: result.0, op: *op, arg: arg.0 },
            Instruction::Vec3Compose { result, x, y, z } => {
                CfgInstruction::Vec3Compose { result: result.0, x: x.0, y: y.0, z: z.0 }
            }
            Instruction::Select { result, cond, then_, else_ } => {
                CfgInstruction::Select { result: result.0, cond: cond.0, then_: then_.0, else_: else_.0 }
            }
            Instruction::StoreBuffer { resource, index, value } => {
                CfgInstruction::StoreBuffer { resource: *resource, index: index.0, value: value.0 }
            }
        }
    }

    fn lower_terminator(terminator: &Terminator) -> CfgTerminator {
        match terminator {
            Terminator::Return(value) => CfgTerminator::Return(value.0),
            Terminator::Branch { target, args } => CfgTerminator::Branch { target: target.0, args: args.iter().map(|value| value.0).collect() },
            Terminator::ConditionalBranch { condition, then_block, then_args, else_block, else_args } => {
                CfgTerminator::ConditionalBranch { condition: condition.0, then_block: then_block.0, then_args: then_args.iter().map(|value| value.0).collect(), else_block: else_block.0, else_args: else_args.iter().map(|value| value.0).collect() }
            }
        }
    }

    fn emit_value(&mut self, ty: Ty, mut instruction: Instruction) -> Result<ValueId, BuildError> {
        if self.blocks[self.current.0 as usize].terminator.is_some() {
            return Err(BuildError::BlockTerminated);
        }
        let value = ValueId(self.values.len() as u32);
        match &mut instruction {
            Instruction::ConstF32 { result, .. }
            | Instruction::ConstF64 { result, .. }
            | Instruction::ConstU32 { result, .. }
            | Instruction::LoadUniform { result, .. }
            | Instruction::LoadBuffer { result, .. }
            | Instruction::GlobalId { result }
            | Instruction::Binary { result, .. }
            | Instruction::Unary { result, .. }
            | Instruction::Vec3Compose { result, .. }
            | Instruction::Select { result, .. } => *result = value,
            Instruction::StoreBuffer { .. } => unreachable!("store has no SSA result"),
        }
        self.values.push(ty);
        self.blocks[self.current.0 as usize].instructions.push(instruction);
        Ok(value)
    }

    fn value_type(&self, value: ValueId) -> Result<&Ty, BuildError> {
        self.values.get(value.0 as usize).ok_or(BuildError::UnknownValue { value: value.0 })
    }

    fn ensure_block_exists(&self, block: BlockId) -> Result<(), BuildError> {
        self.blocks.get(block.0 as usize).map(|_| ()).ok_or(BuildError::UnknownBlock { block: block.0 })
    }

    fn check_block_args(&self, block: BlockId, args: &[ValueId]) -> Result<(), BuildError> {
        let parameters = &self.blocks[block.0 as usize].parameters;
        if parameters.len() != args.len() {
            return Err(BuildError::UnsupportedControlFlow { function: self.name.clone() });
        }
        for ((_, expected), actual) in parameters.iter().zip(args) {
            let actual = self.value_type(*actual)?.clone();
            if *expected != actual {
                return Err(BuildError::TypeMismatch { expected: expected.clone(), actual });
            }
        }
        Ok(())
    }

    fn set_terminator(&mut self, terminator: Terminator) -> Result<(), BuildError> {
        let block = &mut self.blocks[self.current.0 as usize];
        if block.terminator.is_some() {
            return Err(BuildError::BlockTerminated);
        }
        if let Terminator::Return(value) = terminator {
            self.return_value_id = Some(value);
            block.terminator = Some(Terminator::Return(value));
        } else {
            block.terminator = Some(terminator);
        }
        Ok(())
    }

}

/// 构造完整 UCF 程序。
pub struct ProgramBuilder {
    functions: Vec<Function>,
    entry: Option<usize>,
}

impl ProgramBuilder {
    /// 创建程序构造器。
    pub fn new() -> Self {
        Self { functions: Vec::new(), entry: None }
    }

    /// 添加入口函数。
    pub fn entry(mut self, function: FunctionBuilder) -> Result<Self, BuildError> {
        let function = function.finish()?;
        self.entry = Some(self.functions.len());
        self.functions.push(function);
        Ok(self)
    }

    /// 添加结构化入口函数。
    pub fn structured_entry(mut self, function: StructuredFunctionBuilder, ret: Ty, value: Expr) -> Result<Self, BuildError> {
        self.entry = Some(self.functions.len());
        self.functions.push(function.finish(ret, value));
        Ok(self)
    }

    /// 添加非入口函数。
    pub fn function(mut self, function: FunctionBuilder) -> Result<Self, BuildError> {
        self.functions.push(function.finish()?);
        Ok(self)
    }

    /// 完成并验证程序。
    pub fn finish(self) -> Result<Program, BuildError> {
        let entry = self.entry.ok_or(BuildError::MissingEntry)?;
        let program = Program { version: 1, functions: self.functions, entry };
        validate_program(&program).map_err(BuildError::InvalidProgram)?;
        Ok(program)
    }
}

impl Default for ProgramBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CpuInterpreter, Value};

    #[test]
    fn builder_constructs_value_graph() {
        let mut function = FunctionBuilder::new("add_uniform");
        let lhs = function.emit_load_uniform(0, Ty::f32()).expect("open block");
        let rhs = function.emit_const_f32(1.0).expect("open block");
        let value = function.emit_binary(BinOp::Add, lhs, rhs).expect("matching types");
        function.return_value(value).expect("open block");
        let program = ProgramBuilder::new().entry(function).expect("entry").finish().expect("valid");
        assert_eq!(program.functions[0].locals.len(), 3);
    }

    #[test]
    fn builder_rejects_mixed_types() {
        let mut function = FunctionBuilder::new("invalid");
        let lhs = function.emit_load_uniform(0, Ty::f32()).expect("open block");
        let rhs = function.emit_const_u32(1).expect("open block");
        assert!(matches!(function.emit_binary(BinOp::Add, lhs, rhs), Err(BuildError::TypeMismatch { .. })));
    }

    #[test]
    fn builder_lowers_buffer_io_for_cpu_execution() {
        let mut function = FunctionBuilder::new("write_instance_value");
        let index = function.emit_global_id().expect("open block");
        let value = function.emit_const_f32(3.5).expect("open block");
        function.emit_store_buffer(0, index, value).expect("u32 index");
        function.return_value(value).expect("open block");
        let program = ProgramBuilder::new().entry(function).expect("entry").finish().expect("valid");
        let mut interpreter = CpuInterpreter::new();
        interpreter.with_buffer(0, vec![Value::F32(0.0); 3]);
        let results = interpreter.run_parallel(&program, 3).expect("CPU execution");
        assert_eq!(results.len(), 3);
        assert_eq!(interpreter.buffer(0), Some(vec![Value::F32(3.5); 3].as_slice()));
    }

    #[test]
    fn builder_rejects_values_from_other_functions_and_after_return() {
        let mut first = FunctionBuilder::new("first");
        let foreign = first.emit_const_f32(1.0).expect("open block");
        let mut second = FunctionBuilder::new("second");
        assert!(matches!(second.emit_unary(UnOp::Neg, foreign, Ty::f32()), Err(BuildError::UnknownValue { .. })));
        let own = second.emit_const_f32(2.0).expect("open block");
        second.return_value(own).expect("return");
        assert!(matches!(second.emit_const_f32(3.0), Err(BuildError::BlockTerminated)));
    }

    #[test]
    fn builder_records_explicit_control_flow_terminators() {
        let mut function = FunctionBuilder::new("branching");
        let condition = function.emit_load_uniform(0, Ty::Scalar(crate::ScalarTy::Bool)).expect("condition");
        let then_block = function.create_block();
        let else_block = function.create_block();
        function.conditional_branch(condition, then_block, else_block).expect("conditional branch");
        assert!(matches!(function.current_terminator(), Some(Terminator::ConditionalBranch { .. })));

        function.switch_to(then_block).expect("then block");
        let then_value = function.emit_const_f32(1.0).expect("then value");
        function.return_value(then_value).expect("then return");

        function.switch_to(else_block).expect("else block");
        let else_value = function.emit_const_f32(0.0).expect("else value");
        function.return_value(else_value).expect("else return");
        let program = ProgramBuilder::new().entry(function).expect("entry").finish().expect("CFG program");
        let true_result = CpuInterpreter::new().with_uniform(0, Value::Bool(true)).run(&program).expect("true branch");
        assert_eq!(true_result.value, Value::F32(1.0));
        let false_result = CpuInterpreter::new().with_uniform(0, Value::Bool(false)).run(&program).expect("false branch");
        assert_eq!(false_result.value, Value::F32(0.0));
    }

    #[test]
    fn builder_rejects_unterminated_blocks_before_lowering() {
        let function = FunctionBuilder::new("unterminated");
        assert!(matches!(function.finish(), Err(BuildError::MissingTerminator { .. })));
    }

    #[test]
    fn cfg_program_rejects_use_before_definition() {
        let function = Function {
            name: "invalid_cfg".into(),
            params: Vec::new(),
            locals: Vec::new(),
            body: vec![Stmt::Cfg {
                entry: 0,
                blocks: vec![CfgBlock {
                    id: 0,
                    parameters: Vec::new(),
                    instructions: vec![CfgInstruction::Binary { result: 0, op: BinOp::Add, lhs: 0, rhs: 0 }],
                    terminator: CfgTerminator::Return(0),
                }],
            }],
            ret: Ty::f32(),
        };
        let program = Program { version: 1, functions: vec![function], entry: 0 };
        assert!(matches!(validate_program(&program), Err(ValidateError::UnknownLocal { id: 0, .. })));
    }

    #[test]
    fn cfg_block_arguments_carry_loop_state_in_cpu() {
        let mut function = FunctionBuilder::new("cfg_loop_argument");
        let initial = function.emit_const_f32(0.0).expect("initial");
        let one = function.emit_const_f32(1.0).expect("one");
        let limit = function.emit_const_f32(4.0).expect("limit");
        let loop_block = function.create_block();
        let exit_block = function.create_block();
        let loop_value = function.block_parameter(loop_block, Ty::f32()).expect("loop parameter");
        let exit_value = function.block_parameter(exit_block, Ty::f32()).expect("exit parameter");
        function.branch_with_args(loop_block, vec![initial]).expect("enter loop");
        function.switch_to(loop_block).expect("loop block");
        let next = function.emit_binary(BinOp::Add, loop_value, one).expect("increment");
        let condition = function.emit_binary(BinOp::Lt, next, limit).expect("loop condition");
        function
            .conditional_branch_with_args(condition, loop_block, vec![next], exit_block, vec![next])
            .expect("loop branch");
        function.switch_to(exit_block).expect("exit block");
        function.return_value(exit_value).expect("exit return");
        let program = ProgramBuilder::new().entry(function).expect("entry").finish().expect("valid CFG loop");
        let result = CpuInterpreter::new().run(&program).expect("CPU CFG loop");
        assert_eq!(result.value, Value::F64(4.0));
        assert!(result.loop_steps >= 4);
    }

    #[test]
    fn builder_constructs_f64_and_vec3_values() {
        let mut function = FunctionBuilder::new("vec3_value_graph");
        let x = function.emit_const_f64(1.0).expect("open block");
        let y = function.emit_const_f64(2.0).expect("open block");
        let z = function.emit_const_f64(3.0).expect("open block");
        let vector = function.emit_vec3(x, y, z).expect("matching scalar types");
        let component = function.emit_vec3_component(vector, 1, Ty::f64()).expect("valid component");
        function.return_value(component).expect("open block");
        let program = ProgramBuilder::new().entry(function).expect("entry").finish().expect("valid");
        let result = CpuInterpreter::new().run(&program).expect("CPU execution");
        assert_eq!(result.value, Value::F64(2.0));
    }

    #[test]
    fn structured_builder_constructs_bounded_for() {
        let mut function = StructuredFunctionBuilder::new("structured_sum");
        let accumulator = function.let_value(Ty::f32(), Expr::ConstF32(0.0));
        function.bounded_for(4, |function, iter| {
            function.store_local(accumulator, Expr::BinOp {
                op: BinOp::Add,
                lhs: Box::new(Expr::LoadLocal(accumulator)),
                rhs: Box::new(Expr::LoadLocal(iter)),
            });
        });
        let program = ProgramBuilder::new()
            .structured_entry(function, Ty::f32(), Expr::LoadLocal(accumulator))
            .expect("structured entry")
            .finish()
            .expect("valid structured program");
        let result = CpuInterpreter::new().run(&program).expect("CPU structured loop");
        assert_eq!(result.value, Value::F64(6.0));
    }

    #[test]
    fn cpu_is_finite_distinguishes_nan_infinity_and_regular_values() {
        let mut function = StructuredFunctionBuilder::new("is_finite");
        let input = function.let_value(Ty::f32(), Expr::LoadUniform(0));
        let finite = Expr::UnOp { op: UnOp::IsFinite, arg: Box::new(Expr::LoadLocal(input)) };
        let result = Expr::Select { cond: Box::new(finite), then_: Box::new(Expr::ConstF32(1.0)), else_: Box::new(Expr::ConstF32(0.0)) };
        let program = ProgramBuilder::new().structured_entry(function, Ty::f32(), result).expect("is finite entry").finish().expect("valid is finite program");
        for (value, expected) in [(Value::F32(1.0), 1.0), (Value::F32(f32::NAN), 0.0), (Value::F32(f32::INFINITY), 0.0)] {
            let result = CpuInterpreter::new().with_uniform(0, value).run(&program).expect("CPU is finite execution");
            assert_eq!(result.value, Value::F32(expected));
        }
    }
}
