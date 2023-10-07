use naga::{
    AddressSpace, ArraySize, BinaryOperator, Binding, BuiltIn, EntryPoint, Expression,
    Function, FunctionArgument, FunctionResult, GlobalVariable, Literal, Module, ResourceBinding,
    MathFunction, Scalar, ScalarKind, ShaderStage, Statement, StorageAccess, StructMember, Type, TypeInner,
    VectorSize,
};
use ucf_program::{validate_program, BinOp, CfgInstruction, CfgTerminator, Expr, Function as UcfFunction, Program, Stmt, Ty, UnOp};

/// Errors returned when a generic UCF program is outside the first native GPU subset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProgramSpirvError {
    /// The program has no valid entry function.
    MissingEntry,
    /// The program uses a type that this lowering slice cannot represent.
    UnsupportedType(String),
    /// The program uses an expression that this lowering slice cannot represent.
    UnsupportedExpression(String),
    /// The program uses a statement that this lowering slice cannot represent.
    UnsupportedStatement(String),
    /// The generated backend IR failed validation or SPIR-V writing.
    Backend(String),
}

impl std::fmt::Display for ProgramSpirvError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingEntry => f.write_str("generic UCF program has no entry function"),
            Self::UnsupportedType(v) => write!(f, "unsupported UCF GPU type: {v}"),
            Self::UnsupportedExpression(v) => write!(f, "unsupported UCF GPU expression: {v}"),
            Self::UnsupportedStatement(v) => write!(f, "unsupported UCF GPU statement: {v}"),
            Self::Backend(v) => write!(f, "generic UCF SPIR-V backend error: {v}"),
        }
    }
}

impl std::error::Error for ProgramSpirvError {}

/// Lower a generic scalar UCF program directly to SPIR-V.
///
/// This path deliberately constructs backend IR instead of accepting shader source.
/// The first slice writes the entry result to storage binding zero. It supports
/// scalar constants, uniforms, global instance IDs, arithmetic, comparisons,
/// local variables, selection, and bounded `for` loops are added incrementally.
pub fn emit_program_spirv(program: &Program, entry: &str) -> Result<Vec<u8>, ProgramSpirvError> {
    validate_program(program).map_err(|e| ProgramSpirvError::UnsupportedStatement(format!("invalid UCF program: {e}")))?;
    let function = program.entry_fn().ok_or(ProgramSpirvError::MissingEntry)?;
    if function.ret != Ty::f32() {
        return Err(ProgramSpirvError::UnsupportedType(format!("return type {:?}", function.ret)));
    }

    let mut module = Module::default();
    let f32_ty = module.types.insert(Type {
        name: Some("ucf_f32".into()),
        inner: TypeInner::Scalar(Scalar { kind: ScalarKind::Float, width: 4 }),
    }, naga::Span::UNDEFINED);
    let gid_ty = module.types.insert(Type {
        name: Some("ucf_gid".into()),
        inner: TypeInner::Vector { size: VectorSize::Tri, scalar: Scalar { kind: ScalarKind::Uint, width: 4 } },
    }, naga::Span::UNDEFINED);
    let uniform_count = function
        .body
        .iter()
        .flat_map(|stmt| uniform_slots_stmt(stmt))
        .max()
        .map(|slot| slot + 1)
        .unwrap_or(0);
    if uniform_count > 16 {
        return Err(ProgramSpirvError::UnsupportedType(format!(
            "more than sixteen push-constant uniforms ({uniform_count})"
        )));
    }
    let push_u32_ty = module.types.insert(Type {
        name: Some("ucf_u32".into()),
        inner: TypeInner::Scalar(Scalar { kind: ScalarKind::Uint, width: 4 }),
    }, naga::Span::UNDEFINED);
    let bool_ty = module.types.insert(Type {
        name: Some("ucf_bool".into()),
        inner: TypeInner::Scalar(Scalar { kind: ScalarKind::Bool, width: 1 }),
    }, naga::Span::UNDEFINED);
    let uniform_global = {
        let mut members = vec![StructMember {
            name: Some("dispatch_count".into()), ty: push_u32_ty, binding: None, offset: 0,
        }];
        members.extend((0..uniform_count).map(|slot| StructMember {
            name: Some(format!("u{slot}")), ty: f32_ty, binding: None, offset: 4 + slot * 4,
        }));
        let uniform_struct = module.types.insert(Type {
            name: Some("ucf_push_constants".into()),
            inner: TypeInner::Struct {
                members,
                span: 4 + uniform_count * 4,
            },
        }, naga::Span::UNDEFINED);
        module.global_variables.append(GlobalVariable {
            name: Some("ucf_push_constants".into()),
            space: AddressSpace::PushConstant,
            binding: None,
            ty: uniform_struct,
            init: None,
        }, naga::Span::UNDEFINED)
    };
    let output_array = module.types.insert(Type {
        name: Some("ucf_output_array".into()),
        inner: TypeInner::Array { base: f32_ty, size: ArraySize::Dynamic, stride: 4 },
    }, naga::Span::UNDEFINED);
    let output_struct = module.types.insert(Type {
        name: Some("ucf_output".into()),
        inner: TypeInner::Struct {
            members: vec![StructMember { name: Some("data".into()), ty: output_array, binding: None, offset: 0 }],
            span: 4,
        },
    }, naga::Span::UNDEFINED);
    let output_global = module.global_variables.append(GlobalVariable {
        name: Some("ucf_output".into()),
        space: AddressSpace::Storage { access: StorageAccess::LOAD | StorageAccess::STORE },
        binding: Some(ResourceBinding { group: 0, binding: 0 }),
        ty: output_struct,
        init: None,
    }, naga::Span::UNDEFINED);
    let resource_slots = program
        .functions
        .iter()
        .flat_map(|function| function.body.iter().flat_map(resource_slots_stmt))
        .collect::<std::collections::BTreeSet<_>>();
    let mut buffer_globals = std::collections::BTreeMap::new();
    for resource in resource_slots {
        let input_array = module.types.insert(Type {
            name: Some(format!("ucf_input_{resource}_array")),
            inner: TypeInner::Array { base: f32_ty, size: ArraySize::Dynamic, stride: 4 },
        }, naga::Span::UNDEFINED);
        let input_struct = module.types.insert(Type {
            name: Some(format!("ucf_input_{resource}")),
            inner: TypeInner::Struct {
                members: vec![StructMember { name: Some("data".into()), ty: input_array, binding: None, offset: 0 }],
                span: 4,
            },
        }, naga::Span::UNDEFINED);
        let global = module.global_variables.append(GlobalVariable {
            name: Some(format!("ucf_input_{resource}")),
            space: AddressSpace::Storage { access: StorageAccess::LOAD | StorageAccess::STORE },
            binding: Some(ResourceBinding { group: 0, binding: resource + 1 }),
            ty: input_struct,
            init: None,
        }, naga::Span::UNDEFINED);
        buffer_globals.insert(resource, global);
    }

    let mut function_ir = Function {
        name: Some(entry.into()),
        arguments: vec![FunctionArgument {
            name: Some("global_id".into()),
            ty: gid_ty,
            binding: Some(Binding::BuiltIn(BuiltIn::GlobalInvocationId)),
        }],
        result: Some(FunctionResult { ty: f32_ty, binding: None }),
        ..Function::default()
    };
    let gid = function_ir.expressions.append(Expression::FunctionArgument(0), naga::Span::UNDEFINED);
    let gid_x = function_ir.expressions.append(Expression::AccessIndex { base: gid, index: 0 }, naga::Span::UNDEFINED);
    function_ir.body.push(Statement::Emit(naga::Range::new_from_bounds(gid_x, gid_x)), naga::Span::UNDEFINED);
    let push_ptr = function_ir.expressions.append(Expression::GlobalVariable(uniform_global), naga::Span::UNDEFINED);
    let count_ptr = function_ir.expressions.append(Expression::AccessIndex { base: push_ptr, index: 0 }, naga::Span::UNDEFINED);
    let count = function_ir.expressions.append(Expression::Load { pointer: count_ptr }, naga::Span::UNDEFINED);
    let in_bounds = function_ir.expressions.append(Expression::Binary { op: BinaryOperator::Less, left: gid_x, right: count }, naga::Span::UNDEFINED);
    function_ir.body.push(Statement::Emit(naga::Range::new_from_bounds(count, in_bounds)), naga::Span::UNDEFINED);
    let mut inactive = naga::Block::new();
    inactive.push(Statement::Return { value: None }, naga::Span::UNDEFINED);
    function_ir.body.push(Statement::If {
        condition: in_bounds,
        accept: naga::Block::new(),
        reject: inactive,
    }, naga::Span::UNDEFINED);
    let result = {
        let locals = std::collections::BTreeMap::new();
        let emitted_count = function_ir.expressions.len();
        let mut lower = Lowerer { function: &mut function_ir, f32_ty, u32_ty: push_u32_ty, bool_ty, gid_x, in_bounds, uniform_global: Some(uniform_global), buffer_globals, locals, emitted_count };
        let result = lower.lower_statements(function)?;
        lower.emit_pending();
        result
    };
    let out_ptr = function_ir.expressions.append(Expression::GlobalVariable(output_global), naga::Span::UNDEFINED);
    let out_data = function_ir.expressions.append(Expression::AccessIndex { base: out_ptr, index: 0 }, naga::Span::UNDEFINED);
    let out_element = function_ir.expressions.append(Expression::Access { base: out_data, index: gid_x }, naga::Span::UNDEFINED);
    function_ir.body.push(Statement::Emit(naga::Range::new_from_bounds(out_data, out_element)), naga::Span::UNDEFINED);
    let mut store_body = naga::Block::new();
    store_body.push(Statement::Store { pointer: out_element, value: result }, naga::Span::UNDEFINED);
    function_ir.body.push(Statement::If { condition: in_bounds, accept: store_body, reject: naga::Block::new() }, naga::Span::UNDEFINED);
    function_ir.body.push(Statement::Return { value: None }, naga::Span::UNDEFINED);
    function_ir.result = None;
    module.entry_points.push(EntryPoint {
        name: entry.into(),
        stage: ShaderStage::Compute,
        early_depth_test: None,
        workgroup_size: [64, 1, 1],
        workgroup_size_overrides: None,
        function: function_ir,
    });

    let info = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
        .validate(&module)
        .map_err(|e| ProgramSpirvError::Backend(format!("validation failed: {e:?}")))?;
    let options = naga::back::spv::Options {
        lang_version: (1, 3),
        flags: naga::back::spv::WriterFlags::empty(),
        binding_map: Default::default(),
        capabilities: None,
        bounds_check_policies: naga::proc::BoundsCheckPolicies::default(),
        zero_initialize_workgroup_memory: naga::back::spv::ZeroInitializeWorkgroupMemoryMode::Native,
        debug_info: None,
    };
    let words = naga::back::spv::write_vec(&module, &info, &options, None)
        .map_err(|e| ProgramSpirvError::Backend(format!("SPIR-V write failed: {e}")))?;
    Ok(words.into_iter().flat_map(u32::to_le_bytes).collect())
}

struct Lowerer<'a> {
    function: &'a mut Function,
    f32_ty: naga::Handle<Type>,
    u32_ty: naga::Handle<Type>,
    bool_ty: naga::Handle<Type>,
    gid_x: naga::Handle<Expression>,
    in_bounds: naga::Handle<Expression>,
    uniform_global: Option<naga::Handle<GlobalVariable>>,
    buffer_globals: std::collections::BTreeMap<u32, naga::Handle<GlobalVariable>>,
    locals: std::collections::BTreeMap<u32, naga::Handle<naga::LocalVariable>>,
    emitted_count: usize,
}

impl<'a> Lowerer<'a> {
    fn emit_pending(&mut self) {
        let mut block = std::mem::take(&mut self.function.body);
        self.emit_pending_into(&mut block);
        self.function.body = block;
    }

    fn emit_pending_into(&mut self, block: &mut naga::Block) {
        let pending: Vec<_> = self.function.expressions.iter()
            .skip(self.emitted_count)
            .filter(|(_, expression)| !expression.needs_pre_emit())
            .map(|(handle, _)| handle)
            .collect();
        for handle in pending {
            block.push(Statement::Emit(naga::Range::new_from_bounds(handle, handle)), naga::Span::UNDEFINED);
        }
        self.emitted_count = self.function.expressions.len();
    }

    fn lower_statements(&mut self, function: &UcfFunction) -> Result<naga::Handle<Expression>, ProgramSpirvError> {
        let mut result = None;
        for stmt in &function.body {
            match stmt {
                Stmt::Cfg { entry, blocks } => {
                    if result.is_some() || function.body.len() != 1 {
                        return Err(ProgramSpirvError::UnsupportedStatement("CFG must be the complete function body".into()));
                    }
                    result = Some(self.lower_pure_cfg(*entry, blocks)?);
                }
                Stmt::Let { id, ty, init } => {
                    let local_ty = match ty {
                        Ty::Scalar(ucf_program::ScalarTy::F32) => self.f32_ty,
                        Ty::Scalar(ucf_program::ScalarTy::U32) => self.u32_ty,
                        Ty::Scalar(ucf_program::ScalarTy::Bool) => self.bool_ty,
                        other => return Err(ProgramSpirvError::UnsupportedType(format!("local {other:?}"))),
                    };
                    let local = self.function.local_variables.append(naga::LocalVariable { name: Some(format!("local_{id}")), ty: local_ty, init: None }, naga::Span::UNDEFINED);
                    let value = self.lower_expr(init)?;
                    let pointer = self.function.expressions.append(Expression::LocalVariable(local), naga::Span::UNDEFINED);
                    self.emit_pending();
                    self.function.body.push(Statement::Store { pointer, value }, naga::Span::UNDEFINED);
                    self.locals.insert(*id, local);
                }
                Stmt::StoreLocal { id, value } => {
                    let local = *self.locals.get(id).ok_or_else(|| ProgramSpirvError::UnsupportedStatement(format!("unknown local {id}")))?;
                    let pointer = self.function.expressions.append(Expression::LocalVariable(local), naga::Span::UNDEFINED);
                    let value = self.lower_expr(value)?;
                    self.emit_pending();
                    self.function.body.push(Statement::Store { pointer, value }, naga::Span::UNDEFINED);
                }
                Stmt::StoreBuffer { resource, index, value } => {
                    let global = *self.buffer_globals.get(resource).ok_or_else(|| {
                        ProgramSpirvError::UnsupportedStatement(format!("unknown buffer resource {resource}"))
                    })?;
                    let index = self.lower_expr(index)?;
                    let value = self.lower_expr(value)?;
                    let base = self.function.expressions.append(
                        Expression::GlobalVariable(global),
                        naga::Span::UNDEFINED,
                    );
                    let data = self.function.expressions.append(
                        Expression::AccessIndex { base, index: 0 },
                        naga::Span::UNDEFINED,
                    );
                    let pointer = self.function.expressions.append(
                        Expression::Access { base: data, index },
                        naga::Span::UNDEFINED,
                    );
                    self.emit_pending();
                    let mut store_body = naga::Block::new();
                    store_body.push(Statement::Store { pointer, value }, naga::Span::UNDEFINED);
                    self.function.body.push(Statement::If {
                        condition: self.in_bounds,
                        accept: store_body,
                        reject: naga::Block::new(),
                    }, naga::Span::UNDEFINED);
                }
                Stmt::For { iter, body, limit } => {
                    let iter_handle = self.function.local_variables.append(
                        naga::LocalVariable { name: Some("ucf_for_iter".into()), ty: self.u32_ty, init: None },
                        naga::Span::UNDEFINED,
                    );
                    let iter_ptr = self.function.expressions.append(Expression::LocalVariable(iter_handle), naga::Span::UNDEFINED);
                    let previous_iter = self.locals.insert(*iter, iter_handle);
                    let zero = self.function.expressions.append(Expression::Literal(Literal::U32(0)), naga::Span::UNDEFINED);
                    self.emit_pending();
                    self.function.body.push(Statement::Store { pointer: iter_ptr, value: zero }, naga::Span::UNDEFINED);

                    let mut loop_body = naga::Block::new();
                    let iter_value = self.function.expressions.append(Expression::Load { pointer: iter_ptr }, naga::Span::UNDEFINED);
                    let limit_value = self.function.expressions.append(Expression::Literal(Literal::U32(*limit)), naga::Span::UNDEFINED);
                    let done = self.function.expressions.append(
                        Expression::Binary { op: BinaryOperator::GreaterEqual, left: iter_value, right: limit_value },
                        naga::Span::UNDEFINED,
                    );
                    // The iterator load must be emitted inside the loop. Emitting it
                    // in the parent block freezes the condition at the initial value.
                    self.emit_pending_into(&mut loop_body);
                    let mut stop = naga::Block::new();
                    stop.push(Statement::Break, naga::Span::UNDEFINED);
                    loop_body.push(Statement::If { condition: done, accept: stop, reject: naga::Block::new() }, naga::Span::UNDEFINED);
                    for nested in body {
                        let Stmt::StoreLocal { id, value } = nested else {
                            return Err(ProgramSpirvError::UnsupportedStatement(
                                "native GPU fixed For currently supports StoreLocal bodies only".into(),
                            ));
                        };
                        let local = *self.locals.get(id).ok_or_else(|| {
                            ProgramSpirvError::UnsupportedStatement(format!("unknown local {id}"))
                        })?;
                        let pointer = self.function.expressions.append(Expression::LocalVariable(local), naga::Span::UNDEFINED);
                        let value = self.lower_expr(value)?;
                        self.emit_pending_into(&mut loop_body);
                        loop_body.push(Statement::Store { pointer, value }, naga::Span::UNDEFINED);
                    }
                    let one = self.function.expressions.append(Expression::Literal(Literal::U32(1)), naga::Span::UNDEFINED);
                    let current = self.function.expressions.append(Expression::Load { pointer: iter_ptr }, naga::Span::UNDEFINED);
                    let next = self.function.expressions.append(Expression::Binary { op: BinaryOperator::Add, left: current, right: one }, naga::Span::UNDEFINED);
                    let mut continuing = naga::Block::new();
                    self.emit_pending_into(&mut continuing);
                    continuing.push(Statement::Store { pointer: iter_ptr, value: next }, naga::Span::UNDEFINED);
                    self.function.body.push(Statement::Loop { body: loop_body, continuing, break_if: None }, naga::Span::UNDEFINED);
                    if let Some(previous) = previous_iter {
                        self.locals.insert(*iter, previous);
                    } else {
                        self.locals.remove(iter);
                    }
                }
                Stmt::While { cond, body, limit } => {
                    let iter = self.function.local_variables.append(
                        naga::LocalVariable { name: Some("ucf_while_iter".into()), ty: self.u32_ty, init: None },
                        naga::Span::UNDEFINED,
                    );
                    let iter_ptr = self.function.expressions.append(Expression::LocalVariable(iter), naga::Span::UNDEFINED);
                    let zero = self.function.expressions.append(Expression::Literal(Literal::U32(0)), naga::Span::UNDEFINED);
                    self.emit_pending();
                    self.function.body.push(Statement::Store { pointer: iter_ptr, value: zero }, naga::Span::UNDEFINED);

                    let mut loop_body = naga::Block::new();
                    let iter_value = self.function.expressions.append(Expression::Load { pointer: iter_ptr }, naga::Span::UNDEFINED);
                    let limit_value = self.function.expressions.append(Expression::Literal(Literal::U32(*limit)), naga::Span::UNDEFINED);
                    let exhausted = self.function.expressions.append(
                        Expression::Binary { op: BinaryOperator::GreaterEqual, left: iter_value, right: limit_value },
                        naga::Span::UNDEFINED,
                    );
                    let condition = self.lower_expr(cond)?;
                    self.emit_pending_into(&mut loop_body);
                    let mut stop = naga::Block::new();
                    stop.push(Statement::Break, naga::Span::UNDEFINED);
                    loop_body.push(Statement::If { condition: exhausted, accept: stop, reject: naga::Block::new() }, naga::Span::UNDEFINED);
                    let mut body_block = naga::Block::new();
                    let mut body_stop = naga::Block::new();
                    body_stop.push(Statement::Break, naga::Span::UNDEFINED);
                    let mut body_accept = naga::Block::new();
                    for nested in body {
                        let Stmt::StoreLocal { id, value } = nested else {
                            return Err(ProgramSpirvError::UnsupportedStatement(
                                "native GPU bounded While currently supports StoreLocal bodies only".into(),
                            ));
                        };
                        let local = *self.locals.get(id).ok_or_else(|| {
                            ProgramSpirvError::UnsupportedStatement(format!("unknown local {id}"))
                        })?;
                        let pointer = self.function.expressions.append(Expression::LocalVariable(local), naga::Span::UNDEFINED);
                        let value = self.lower_expr(value)?;
                        self.emit_pending_into(&mut body_accept);
                        body_accept.push(Statement::Store { pointer, value }, naga::Span::UNDEFINED);
                    }
                    body_block.push(Statement::If { condition, accept: body_accept, reject: body_stop }, naga::Span::UNDEFINED);
                    loop_body.append(&mut body_block);
                    let one = self.function.expressions.append(Expression::Literal(Literal::U32(1)), naga::Span::UNDEFINED);
                    let current = self.function.expressions.append(Expression::Load { pointer: iter_ptr }, naga::Span::UNDEFINED);
                    let next = self.function.expressions.append(Expression::Binary { op: BinaryOperator::Add, left: current, right: one }, naga::Span::UNDEFINED);
                    let mut continuing = naga::Block::new();
                    self.emit_pending_into(&mut continuing);
                    continuing.push(Statement::Store { pointer: iter_ptr, value: next }, naga::Span::UNDEFINED);
                    self.function.body.push(Statement::Loop { body: loop_body, continuing, break_if: None }, naga::Span::UNDEFINED);
                }
                Stmt::Return(value) => result = Some(self.lower_expr(value)?),
                Stmt::Expr(expr) => { let _ = self.lower_expr(expr)?; }
                other => return Err(ProgramSpirvError::UnsupportedStatement(format!("{other:?}"))),
            }
        }
        result.ok_or_else(|| ProgramSpirvError::UnsupportedStatement("entry has no return".into()))
    }

    fn lower_pure_cfg(&mut self, entry: u32, blocks: &[ucf_program::CfgBlock]) -> Result<naga::Handle<Expression>, ProgramSpirvError> {
        let entry_block = blocks.iter().find(|block| block.id == entry).ok_or_else(|| ProgramSpirvError::UnsupportedStatement("CFG entry block is missing".into()))?;
        if blocks.len() == 1 {
            let CfgTerminator::Return(value) = entry_block.terminator else {
                return Err(ProgramSpirvError::UnsupportedStatement("single-block CFG must return".into()));
            };
            let mut values = std::collections::BTreeMap::new();
            for instruction in &entry_block.instructions {
                self.lower_cfg_instruction(instruction, &mut values)?;
            }
            return values.get(&value).copied().ok_or_else(|| {
                ProgramSpirvError::UnsupportedStatement(format!("CFG return value {value} is not defined"))
            });
        }
        let CfgTerminator::ConditionalBranch { condition, then_block, then_args, else_block, else_args } = &entry_block.terminator else {
            return Err(ProgramSpirvError::UnsupportedStatement("GPU CFG subset requires one conditional entry branch".into()));
        };
        let mut values = std::collections::BTreeMap::new();
        for instruction in &entry_block.instructions {
            self.lower_cfg_instruction(instruction, &mut values)?;
        }
        let condition = *values.get(condition).ok_or_else(|| ProgramSpirvError::UnsupportedStatement("CFG condition is not defined in entry block".into()))?;
        let then_value = self.lower_cfg_return_block(*then_block, then_args, blocks, &mut values)?;
        let else_value = self.lower_cfg_return_block(*else_block, else_args, blocks, &mut values)?;
        Ok(self.function.expressions.append(Expression::Select { condition, accept: then_value, reject: else_value }, naga::Span::UNDEFINED))
    }

    fn lower_cfg_return_block(&mut self, block_id: u32, args: &[u32], blocks: &[ucf_program::CfgBlock], values: &mut std::collections::BTreeMap<u32, naga::Handle<Expression>>) -> Result<naga::Handle<Expression>, ProgramSpirvError> {
        let block = blocks.iter().find(|block| block.id == block_id).ok_or_else(|| ProgramSpirvError::UnsupportedStatement(format!("CFG target block {block_id} is missing")))?;
        if block.parameters.len() != args.len() {
            return Err(ProgramSpirvError::UnsupportedStatement("CFG branch argument count does not match target parameters".into()));
        }
        for (parameter, argument) in block.parameters.iter().zip(args) {
            let value = *values.get(argument).ok_or_else(|| ProgramSpirvError::UnsupportedExpression(format!("unknown CFG branch argument {argument}")))?;
            values.insert(parameter.value, value);
        }
        for instruction in &block.instructions {
            self.lower_cfg_instruction(instruction, values)?;
        }
        let CfgTerminator::Return(value) = block.terminator else {
            return Err(ProgramSpirvError::UnsupportedStatement("GPU CFG subset requires branch blocks to return directly".into()));
        };
        values.get(&value).copied().ok_or_else(|| ProgramSpirvError::UnsupportedStatement(format!("CFG return value {value} is not defined")))
    }

    fn lower_cfg_instruction(&mut self, instruction: &CfgInstruction, values: &mut std::collections::BTreeMap<u32, naga::Handle<Expression>>) -> Result<(), ProgramSpirvError> {
        let span = naga::Span::UNDEFINED;
        if let CfgInstruction::StoreBuffer { resource, index, value } = instruction {
            let global = *self.buffer_globals.get(resource).ok_or_else(|| {
                ProgramSpirvError::UnsupportedStatement(format!("unknown buffer resource {resource}"))
            })?;
            let index = *values.get(index).ok_or_else(|| ProgramSpirvError::UnsupportedExpression(format!("unknown CFG value {index}")))?;
            let value = *values.get(value).ok_or_else(|| ProgramSpirvError::UnsupportedExpression(format!("unknown CFG value {value}")))?;
            let base = self.function.expressions.append(Expression::GlobalVariable(global), span);
            let data = self.function.expressions.append(Expression::AccessIndex { base, index: 0 }, span);
            let pointer = self.function.expressions.append(Expression::Access { base: data, index }, span);
            self.emit_pending();
            self.function.body.push(Statement::Store { pointer, value }, span);
            return Ok(());
        }
        let handle = match instruction {
            CfgInstruction::ConstF32 { value, .. } => self.function.expressions.append(Expression::Literal(Literal::F32(*value)), span),
            CfgInstruction::ConstU32 { value, .. } => self.function.expressions.append(Expression::Literal(Literal::U32(*value)), span),
            CfgInstruction::LoadUniform { slot, .. } => {
                let global = self.uniform_global.ok_or_else(|| ProgramSpirvError::UnsupportedExpression("uniform without push-constant layout".into()))?;
                let base = self.function.expressions.append(Expression::GlobalVariable(global), span);
                let field = self.function.expressions.append(Expression::AccessIndex { base, index: *slot + 1 }, span);
                self.function.expressions.append(Expression::Load { pointer: field }, span)
            }
            CfgInstruction::GlobalId { .. } => self.gid_x,
            CfgInstruction::LoadBuffer { resource, index, .. } => {
                let global = *self.buffer_globals.get(resource).ok_or_else(|| {
                    ProgramSpirvError::UnsupportedStatement(format!("unknown buffer resource {resource}"))
                })?;
                let index = *values.get(index).ok_or_else(|| ProgramSpirvError::UnsupportedExpression(format!("unknown CFG value {index}")))?;
                let base = self.function.expressions.append(Expression::GlobalVariable(global), span);
                let data = self.function.expressions.append(Expression::AccessIndex { base, index: 0 }, span);
                let element = self.function.expressions.append(Expression::Access { base: data, index }, span);
                self.function.expressions.append(Expression::Load { pointer: element }, span)
            }
            CfgInstruction::Binary { op, lhs, rhs, .. } => {
                let left = *values.get(lhs).ok_or_else(|| ProgramSpirvError::UnsupportedExpression(format!("unknown CFG value {lhs}")))?;
                let right = *values.get(rhs).ok_or_else(|| ProgramSpirvError::UnsupportedExpression(format!("unknown CFG value {rhs}")))?;
                let expression = match op {
                    BinOp::Min => Expression::Math { fun: MathFunction::Min, arg: left, arg1: Some(right), arg2: None, arg3: None },
                    BinOp::Max => Expression::Math { fun: MathFunction::Max, arg: left, arg1: Some(right), arg2: None, arg3: None },
                    BinOp::Add => Expression::Binary { op: BinaryOperator::Add, left, right },
                    BinOp::Sub => Expression::Binary { op: BinaryOperator::Subtract, left, right },
                    BinOp::Mul => Expression::Binary { op: BinaryOperator::Multiply, left, right },
                    BinOp::Div => Expression::Binary { op: BinaryOperator::Divide, left, right },
                    BinOp::Lt => Expression::Binary { op: BinaryOperator::Less, left, right },
                    BinOp::Le => Expression::Binary { op: BinaryOperator::LessEqual, left, right },
                    BinOp::Eq => Expression::Binary { op: BinaryOperator::Equal, left, right },
                    BinOp::Ne => Expression::Binary { op: BinaryOperator::NotEqual, left, right },
                    BinOp::Ge => Expression::Binary { op: BinaryOperator::GreaterEqual, left, right },
                    BinOp::Gt => Expression::Binary { op: BinaryOperator::Greater, left, right },
                    other => return Err(ProgramSpirvError::UnsupportedExpression(format!("CFG binary {other:?}"))),
                };
                self.function.expressions.append(expression, span)
            }
            CfgInstruction::Select { cond, then_, else_, .. } => {
                let condition = *values.get(cond).ok_or_else(|| ProgramSpirvError::UnsupportedExpression(format!("unknown CFG value {cond}")))?;
                let accept = *values.get(then_).ok_or_else(|| ProgramSpirvError::UnsupportedExpression(format!("unknown CFG value {then_}")))?;
                let reject = *values.get(else_).ok_or_else(|| ProgramSpirvError::UnsupportedExpression(format!("unknown CFG value {else_}")))?;
                self.function.expressions.append(Expression::Select { condition, accept, reject }, span)
            }
            CfgInstruction::Unary { op, arg, .. } => {
                let arg = *values.get(arg).ok_or_else(|| ProgramSpirvError::UnsupportedExpression(format!("unknown CFG value {arg}")))?;
                match op {
                    UnOp::Neg => self.function.expressions.append(Expression::Unary { op: naga::UnaryOperator::Negate, expr: arg }, span),
                    UnOp::Sqrt => self.function.expressions.append(Expression::Math { fun: MathFunction::Sqrt, arg, arg1: None, arg2: None, arg3: None }, span),
                    UnOp::Sin => self.function.expressions.append(Expression::Math { fun: MathFunction::Sin, arg, arg1: None, arg2: None, arg3: None }, span),
                    UnOp::Cos => self.function.expressions.append(Expression::Math { fun: MathFunction::Cos, arg, arg1: None, arg2: None, arg3: None }, span),
                    UnOp::Atan => self.function.expressions.append(Expression::Math { fun: MathFunction::Atan, arg, arg1: None, arg2: None, arg3: None }, span),
                    UnOp::Abs => self.function.expressions.append(Expression::Math { fun: MathFunction::Abs, arg, arg1: None, arg2: None, arg3: None }, span),
                    UnOp::Ln => self.function.expressions.append(Expression::Math { fun: MathFunction::Log, arg, arg1: None, arg2: None, arg3: None }, span),
                    UnOp::IsFinite => {
                        let same = self.function.expressions.append(Expression::Binary { op: BinaryOperator::Equal, left: arg, right: arg }, span);
                        let abs = self.function.expressions.append(Expression::Math { fun: MathFunction::Abs, arg, arg1: None, arg2: None, arg3: None }, span);
                        let limit = self.function.expressions.append(Expression::Literal(Literal::F32(f32::MAX)), span);
                        let below_limit = self.function.expressions.append(Expression::Binary { op: BinaryOperator::Less, left: abs, right: limit }, span);
                        let false_value = self.function.expressions.append(Expression::Literal(Literal::Bool(false)), span);
                        self.function.expressions.append(Expression::Select { condition: same, accept: below_limit, reject: false_value }, span)
                    }
                    other => return Err(ProgramSpirvError::UnsupportedExpression(format!("CFG unary {other:?}"))),
                }
            }
            _ => return Err(ProgramSpirvError::UnsupportedStatement("CFG instruction is outside the pure F32 GPU subset".into())),
        };
        let result = match instruction {
            CfgInstruction::ConstF32 { result, .. } | CfgInstruction::ConstU32 { result, .. }
            | CfgInstruction::LoadUniform { result, .. } | CfgInstruction::LoadBuffer { result, .. }
            | CfgInstruction::GlobalId { result } | CfgInstruction::Binary { result, .. }
            | CfgInstruction::Unary { result, .. } | CfgInstruction::Select { result, .. } => *result,
            CfgInstruction::StoreBuffer { .. } => unreachable!(),
            _ => unreachable!("unsupported CFG instruction did not produce a handle"),
        };
        values.insert(result, handle);
        Ok(())
    }

    fn lower_expr(&mut self, expr: &Expr) -> Result<naga::Handle<Expression>, ProgramSpirvError> {
        let span = naga::Span::UNDEFINED;
        match expr {
            Expr::ConstF32(v) => Ok(self.function.expressions.append(Expression::Literal(Literal::F32(*v)), span)),
            Expr::ConstBool(v) => Ok(self.function.expressions.append(Expression::Literal(Literal::Bool(*v)), span)),
            Expr::ConstU32(v) => Ok(self.function.expressions.append(Expression::Literal(Literal::U32(*v)), span)),
            Expr::GlobalId => Ok(self.gid_x),
            Expr::LoadUniform(slot) => {
                let global = self.uniform_global.ok_or_else(|| ProgramSpirvError::UnsupportedExpression("uniform without push-constant layout".into()))?;
                let base = self.function.expressions.append(Expression::GlobalVariable(global), span);
                let field = self.function.expressions.append(Expression::AccessIndex { base, index: *slot + 1 }, span);
                let value = self.function.expressions.append(Expression::Load { pointer: field }, span);
                Ok(value)
            }
            Expr::LoadLocal(id) => {
                let local = *self.locals.get(id).ok_or_else(|| ProgramSpirvError::UnsupportedExpression(format!("unknown local {id}")))?;
                let pointer = self.function.expressions.append(Expression::LocalVariable(local), span);
                Ok(self.function.expressions.append(Expression::Load { pointer }, span))
            }
            Expr::LoadBuffer { resource, index } => {
                let global = *self.buffer_globals.get(resource).ok_or_else(|| {
                    ProgramSpirvError::UnsupportedExpression(format!("unknown buffer resource {resource}"))
                })?;
                let index = self.lower_expr(index)?;
                let base = self.function.expressions.append(Expression::GlobalVariable(global), span);
                let data = self.function.expressions.append(Expression::AccessIndex { base, index: 0 }, span);
                let element = self.function.expressions.append(Expression::Access { base: data, index }, span);
                let value = self.function.expressions.append(Expression::Load { pointer: element }, span);
                Ok(value)
            }
            Expr::BinOp { op, lhs, rhs } => {
                let left = self.lower_expr(lhs)?; let right = self.lower_expr(rhs)?;
                let expression = match op {
                    BinOp::Min => Expression::Math { fun: MathFunction::Min, arg: left, arg1: Some(right), arg2: None, arg3: None },
                    BinOp::Max => Expression::Math { fun: MathFunction::Max, arg: left, arg1: Some(right), arg2: None, arg3: None },
                    BinOp::Add => Expression::Binary { op: BinaryOperator::Add, left, right },
                    BinOp::Sub => Expression::Binary { op: BinaryOperator::Subtract, left, right },
                    BinOp::Mul => Expression::Binary { op: BinaryOperator::Multiply, left, right },
                    BinOp::Div => Expression::Binary { op: BinaryOperator::Divide, left, right },
                    BinOp::Lt => Expression::Binary { op: BinaryOperator::Less, left, right },
                    BinOp::Le => Expression::Binary { op: BinaryOperator::LessEqual, left, right },
                    BinOp::Eq => Expression::Binary { op: BinaryOperator::Equal, left, right },
                    BinOp::Ne => Expression::Binary { op: BinaryOperator::NotEqual, left, right },
                    BinOp::Ge => Expression::Binary { op: BinaryOperator::GreaterEqual, left, right },
                    BinOp::Gt => Expression::Binary { op: BinaryOperator::Greater, left, right },
                    other => return Err(ProgramSpirvError::UnsupportedExpression(format!("binary {other:?}"))),
                };
                Ok(self.function.expressions.append(expression, span))
            }
            Expr::UnOp { op, arg } => {
                let arg = self.lower_expr(arg)?;
                let expression = match op {
                    UnOp::Neg => Expression::Unary { op: naga::UnaryOperator::Negate, expr: arg },
                    UnOp::Sqrt => Expression::Math { fun: MathFunction::Sqrt, arg, arg1: None, arg2: None, arg3: None },
                    UnOp::Sin => Expression::Math { fun: MathFunction::Sin, arg, arg1: None, arg2: None, arg3: None },
                    UnOp::Cos => Expression::Math { fun: MathFunction::Cos, arg, arg1: None, arg2: None, arg3: None },
                    UnOp::Atan => Expression::Math { fun: MathFunction::Atan, arg, arg1: None, arg2: None, arg3: None },
                    UnOp::Abs => Expression::Math { fun: MathFunction::Abs, arg, arg1: None, arg2: None, arg3: None },
                    UnOp::Ln => Expression::Math { fun: MathFunction::Log, arg, arg1: None, arg2: None, arg3: None },
                    UnOp::IsFinite => {
                        let same = self.function.expressions.append(Expression::Binary { op: BinaryOperator::Equal, left: arg, right: arg }, span);
                        let abs = self.function.expressions.append(Expression::Math { fun: MathFunction::Abs, arg, arg1: None, arg2: None, arg3: None }, span);
                        let limit = self.function.expressions.append(Expression::Literal(Literal::F32(f32::MAX)), span);
                        let below_limit = self.function.expressions.append(Expression::Binary { op: BinaryOperator::Less, left: abs, right: limit }, span);
                        let false_value = self.function.expressions.append(Expression::Literal(Literal::Bool(false)), span);
                        Expression::Select { condition: same, accept: below_limit, reject: false_value }
                    }
                    other => return Err(ProgramSpirvError::UnsupportedExpression(format!("unary {other:?}"))),
                };
                Ok(self.function.expressions.append(expression, span))
            }
            Expr::Select { cond, then_, else_ } => {
                let condition = self.lower_expr(cond)?;
                let accept = self.lower_expr(then_)?;
                let reject = self.lower_expr(else_)?;
                Ok(self.function.expressions.append(Expression::Select { condition, accept, reject }, span))
            }
            other => Err(ProgramSpirvError::UnsupportedExpression(format!("{other:?}"))),
        }
    }
}

fn uniform_slots_stmt(stmt: &Stmt) -> Vec<u32> {
    let mut slots = Vec::new();
    fn visit_expr(value: &Expr, slots: &mut Vec<u32>) {
        match value {
            Expr::LoadUniform(slot) => slots.push(*slot),
            Expr::BinOp { lhs, rhs, .. } => { visit_expr(lhs, slots); visit_expr(rhs, slots); }
            Expr::UnOp { arg, .. } => visit_expr(arg, slots),
            Expr::Select { cond, then_, else_ } => { visit_expr(cond, slots); visit_expr(then_, slots); visit_expr(else_, slots); }
            _ => {}
        }
    }
    match stmt {
        Stmt::Let { init, .. } | Stmt::StoreLocal { value: init, .. } | Stmt::Return(init) | Stmt::Expr(init) => visit_expr(init, &mut slots),
        Stmt::If { cond, then_body, else_body } => { visit_expr(cond, &mut slots); for stmt in then_body.iter().chain(else_body) { slots.extend(uniform_slots_stmt(stmt)); } }
        Stmt::For { body, .. } => for stmt in body { slots.extend(uniform_slots_stmt(stmt)); },
        Stmt::While { cond, body, .. } => {
            visit_expr(cond, &mut slots);
            for stmt in body { slots.extend(uniform_slots_stmt(stmt)); }
        }
        Stmt::StoreBuffer { index, value, .. } => { visit_expr(index, &mut slots); visit_expr(value, &mut slots); }
        Stmt::Break | Stmt::Continue => {}
        Stmt::Cfg { blocks, .. } => {
            for block in blocks {
                for instruction in &block.instructions {
                    match instruction {
                        CfgInstruction::LoadUniform { slot, .. } => slots.push(*slot),
                        CfgInstruction::Binary { .. } | CfgInstruction::Unary { .. } | CfgInstruction::Select { .. }
                        | CfgInstruction::Vec3Compose { .. } | CfgInstruction::LoadBuffer { .. }
                        | CfgInstruction::StoreBuffer { .. } | CfgInstruction::ConstF32 { .. }
                        | CfgInstruction::ConstF64 { .. } | CfgInstruction::ConstU32 { .. }
                        | CfgInstruction::GlobalId { .. } => {}
                    }
                }
            }
        }
    }
    slots
}

fn resource_slots_stmt(stmt: &Stmt) -> Vec<u32> {
    let mut slots = Vec::new();
    fn visit_expr(value: &Expr, slots: &mut Vec<u32>) {
        match value {
            Expr::LoadBuffer { resource, index } => {
                slots.push(*resource);
                visit_expr(index, slots);
            }
            Expr::BinOp { lhs, rhs, .. } => { visit_expr(lhs, slots); visit_expr(rhs, slots); }
            Expr::UnOp { arg, .. } => visit_expr(arg, slots),
            Expr::Select { cond, then_, else_ } => {
                visit_expr(cond, slots); visit_expr(then_, slots); visit_expr(else_, slots);
            }
            Expr::Vec3Compose { x, y, z } => {
                visit_expr(x, slots); visit_expr(y, slots); visit_expr(z, slots);
            }
            Expr::Call { args, .. } => for arg in args { visit_expr(arg, slots); },
            _ => {}
        }
    }
    match stmt {
        Stmt::Let { init, .. } | Stmt::StoreLocal { value: init, .. }
        | Stmt::Return(init) | Stmt::Expr(init) => visit_expr(init, &mut slots),
        Stmt::StoreBuffer { resource, index, value } => {
            slots.push(*resource); visit_expr(index, &mut slots); visit_expr(value, &mut slots);
        }
        Stmt::If { cond, then_body, else_body } => {
            visit_expr(cond, &mut slots);
            for nested in then_body.iter().chain(else_body) { slots.extend(resource_slots_stmt(nested)); }
        }
        Stmt::For { body, .. } | Stmt::While { body, .. } => {
            for nested in body { slots.extend(resource_slots_stmt(nested)); }
        }
        Stmt::Break | Stmt::Continue => {}
        Stmt::Cfg { blocks, .. } => {
            for block in blocks {
                for instruction in &block.instructions {
                    match instruction {
                        CfgInstruction::LoadBuffer { resource, .. }
                        | CfgInstruction::StoreBuffer { resource, .. } => slots.push(*resource),
                        _ => {}
                    }
                }
            }
        }
    }
    slots
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lowers_scalar_program_without_shader_source() {
        let program = Program {
            version: 1,
            functions: vec![UcfFunction {
                name: "main".into(),
                params: Vec::new(),
                locals: Vec::new(),
                body: vec![Stmt::Return(Expr::BinOp {
                    op: BinOp::Add,
                    lhs: Box::new(Expr::ConstF32(1.0)),
                    rhs: Box::new(Expr::ConstF32(2.0)),
                })],
                ret: Ty::f32(),
            }],
            entry: 0,
        };
        let bytes = emit_program_spirv(&program, "ucf_program_main").expect("valid SPIR-V");
        assert_eq!(&bytes[..4], &[0x03, 0x02, 0x23, 0x07]);
        assert!(bytes.len() > 20);
    }

    #[test]
    fn lowers_push_constant_uniform() {
        let program = Program {
            version: 1,
            functions: vec![UcfFunction {
                name: "main".into(), params: Vec::new(), locals: Vec::new(),
                body: vec![Stmt::Return(Expr::LoadUniform(0))], ret: Ty::f32(),
            }], entry: 0,
        };
        let bytes = emit_program_spirv(&program, "ucf_uniform_main").expect("valid SPIR-V");
        assert_eq!(&bytes[..4], &[0x03, 0x02, 0x23, 0x07]);
    }
}
