use std::collections::{HashMap, HashSet, VecDeque};

use crate::error::ValidateError;
use crate::expr::Expr;
use crate::program::Program;
use crate::stmt::{CfgInstruction, CfgTerminator, Stmt};

/// 默认循环上限（与 CPU 解释器一致）。
pub const DEFAULT_MAX_LOOP: u32 = 1_000_000;

/// 验证程序结构、循环上限与局部槽位声明。
pub fn validate_program(program: &Program) -> Result<(), ValidateError> {
    if program.functions.is_empty() {
        return Err(ValidateError::EmptyProgram);
    }
    if program.entry >= program.functions.len() {
        return Err(ValidateError::BadEntry(program.entry));
    }
    for func in &program.functions {
        let mut known: HashSet<u32> = func.params.iter().map(|(id, _)| *id).collect();
        for (id, _) in &func.locals {
            known.insert(*id);
        }
        validate_stmts(program, &func.name, &func.body, &known, DEFAULT_MAX_LOOP, false, &func.ret)?;
    }
    for index in 0..program.functions.len() {
        let mut visiting = HashSet::new();
        validate_call_graph(program, index, &mut visiting)?;
    }
    Ok(())
}

fn validate_call_graph(program: &Program, index: usize, visiting: &mut HashSet<usize>) -> Result<(), ValidateError> {
    if !visiting.insert(index) {
        return Err(ValidateError::RecursiveCall { index: index as u32 });
    }
    let mut callees = Vec::new();
    collect_calls(&program.functions[index].body, &mut callees);
    for callee in callees {
        validate_call_graph(program, callee as usize, visiting)?;
    }
    visiting.remove(&index);
    Ok(())
}

fn collect_calls(body: &[Stmt], out: &mut Vec<u32>) {
    for stmt in body {
        match stmt {
            Stmt::Let { init, .. } | Stmt::StoreLocal { value: init, .. } | Stmt::Return(init) | Stmt::Expr(init) => {
                collect_expr_calls(init, out)
            }
            Stmt::StoreBuffer { index, value, .. } => {
                collect_expr_calls(index, out);
                collect_expr_calls(value, out);
            }
            Stmt::If { cond, then_body, else_body } => {
                collect_expr_calls(cond, out);
                collect_calls(then_body, out);
                collect_calls(else_body, out);
            }
            Stmt::For { body, .. } | Stmt::While { body, .. } => collect_calls(body, out),
            Stmt::Break | Stmt::Continue => {}
            Stmt::Cfg { .. } => {}
        }
    }
}

fn validate_cfg(name: &str, entry: u32, blocks: &[crate::CfgBlock], known_locals: &HashSet<u32>, return_ty: &crate::Ty) -> Result<(), ValidateError> {
    let ids: HashSet<u32> = blocks.iter().map(|block| block.id).collect();
    if blocks.is_empty() || !ids.contains(&entry) || ids.len() != blocks.len() {
        return Err(ValidateError::BadControlFlow { name: name.to_string() });
    }
    let mut definitions = HashMap::<u32, (crate::Ty, u32)>::new();
    let mut declared = HashSet::new();
    let mut pending = Vec::new();
    for block in blocks {
        if block.id == entry && !block.parameters.is_empty() {
            return Err(ValidateError::BadControlFlow { name: name.to_string() });
        }
        for parameter in &block.parameters {
            if known_locals.contains(&parameter.value) || !declared.insert(parameter.value) {
                return Err(ValidateError::BadControlFlow { name: name.to_string() });
            }
            definitions.insert(parameter.value, (parameter.ty.clone(), block.id));
        }
        for instruction in &block.instructions {
            if let Some(value) = cfg_instruction_result(instruction) {
                if known_locals.contains(&value) || !declared.insert(value) {
                    return Err(ValidateError::BadControlFlow { name: name.to_string() });
                }
                pending.push((value, block.id, instruction));
            }
        }
    }
    while !pending.is_empty() {
        let before = pending.len();
        pending.retain(|(value, block, instruction)| {
            if let Ok(ty) = cfg_instruction_type(instruction, &definitions) {
                definitions.insert(*value, (ty, *block));
                false
            } else { true }
        });
        if pending.len() == before {
            let id = pending[0].0;
            return Err(ValidateError::UnknownLocal { name: name.to_string(), id });
        }
    }
    let mut predecessors = HashMap::<u32, Vec<u32>>::new();
    for block in blocks {
        let targets = match &block.terminator {
            crate::CfgTerminator::Return(_) => Vec::new(),
            crate::CfgTerminator::Branch { target, .. } => vec![*target],
            crate::CfgTerminator::ConditionalBranch { then_block, else_block, .. } => vec![*then_block, *else_block],
        };
        for target in targets {
            if !ids.contains(&target) { return Err(ValidateError::BadControlFlow { name: name.to_string() }); }
            predecessors.entry(target).or_default().push(block.id);
        }
    }
    let mut reachable = HashSet::new();
    let mut queue = VecDeque::from([entry]);
    while let Some(id) = queue.pop_front() {
        if !reachable.insert(id) { continue; }
        if let Some(block) = blocks.iter().find(|block| block.id == id) {
            for next in match &block.terminator {
                crate::CfgTerminator::Return(_) => Vec::new(),
                crate::CfgTerminator::Branch { target, .. } => vec![*target],
                crate::CfgTerminator::ConditionalBranch { then_block, else_block, .. } => vec![*then_block, *else_block],
            } { queue.push_back(next); }
        }
    }
    if reachable.len() != blocks.len() { return Err(ValidateError::BadControlFlow { name: name.to_string() }); }
    let all: HashSet<u32> = ids.clone();
    let mut dominators = HashMap::<u32, HashSet<u32>>::new();
    for block in blocks {
        dominators.insert(block.id, if block.id == entry { HashSet::from([entry]) } else { all.clone() });
    }
    let mut changed = true;
    while changed {
        changed = false;
        for block in blocks.iter().filter(|block| block.id != entry) {
            let mut next = all.clone();
            for predecessor in predecessors.get(&block.id).into_iter().flatten() {
                next = next.intersection(dominators.get(predecessor).unwrap()).copied().collect();
            }
            next.insert(block.id);
            if next != dominators[&block.id] { dominators.insert(block.id, next); changed = true; }
        }
    }
    for block in blocks {
        let mut values = known_locals.clone();
        values.extend(definitions.iter().filter_map(|(value, (_, defining_block))| {
            (*defining_block != block.id && dominators[&block.id].contains(defining_block)).then_some(*value)
        }));
        for parameter in &block.parameters {
            values.insert(parameter.value);
        }
        for instruction in &block.instructions {
            let operands = match instruction {
                CfgInstruction::LoadBuffer { index, .. } => vec![*index],
                CfgInstruction::Binary { lhs, rhs, .. } => vec![*lhs, *rhs],
                CfgInstruction::Unary { arg, .. } => vec![*arg],
                CfgInstruction::Vec3Compose { x, y, z, .. } => vec![*x, *y, *z],
                CfgInstruction::Select { cond, then_, else_, .. } => vec![*cond, *then_, *else_],
                CfgInstruction::StoreBuffer { index, value, .. } => vec![*index, *value],
                _ => Vec::new(),
            };
            if operands.iter().any(|operand| !values.contains(operand)) {
                return Err(ValidateError::UnknownLocal {
                    name: name.to_string(),
                    id: operands.iter().find(|operand| !values.contains(operand)).copied().unwrap_or(0),
                });
            }
            for operand in operands.iter().copied() {
                if let Some((_, defining_block)) = definitions.get(&operand) {
                    if !dominators[&block.id].contains(defining_block) {
                        return Err(ValidateError::BadControlFlow { name: name.to_string() });
                    }
                }
            }
            if let Some(result) = cfg_instruction_result(instruction) {
                values.insert(result);
            }
        }
        match &block.terminator {
            CfgTerminator::Return(value) => {
                if !values.contains(value) {
                    return Err(ValidateError::UnknownLocal { name: name.to_string(), id: *value });
                }
                if definitions.get(value).map(|(ty, _)| ty) != Some(return_ty) {
                    return Err(ValidateError::BadControlFlow { name: name.to_string() });
                }
            }
            CfgTerminator::Branch { target, args } => {
                let target_block = blocks.iter().find(|candidate| candidate.id == *target);
                if !validate_edge_args(target_block, args, &values, &definitions) {
                    return Err(ValidateError::BadControlFlow { name: name.to_string() });
                }
            }
            CfgTerminator::ConditionalBranch { condition, then_block, then_args, else_block, else_args } => {
                let then_target = blocks.iter().find(|candidate| candidate.id == *then_block);
                let else_target = blocks.iter().find(|candidate| candidate.id == *else_block);
                if definitions.get(condition).map(|(ty, _)| ty) != Some(&crate::Ty::Scalar(crate::ScalarTy::Bool))
                    || !values.contains(condition)
                    || !validate_edge_args(then_target, then_args, &values, &definitions)
                    || !validate_edge_args(else_target, else_args, &values, &definitions) {
                    return Err(ValidateError::BadControlFlow { name: name.to_string() });
                }
            }
        }
    }
    Ok(())
}

fn cfg_instruction_result(instruction: &CfgInstruction) -> Option<u32> {
    match instruction {
        CfgInstruction::ConstF32 { result, .. } | CfgInstruction::ConstF64 { result, .. } | CfgInstruction::ConstU32 { result, .. }
        | CfgInstruction::LoadUniform { result, .. } | CfgInstruction::LoadBuffer { result, .. } | CfgInstruction::GlobalId { result }
        | CfgInstruction::Binary { result, .. } | CfgInstruction::Unary { result, .. } | CfgInstruction::Vec3Compose { result, .. }
        | CfgInstruction::Select { result, .. } => Some(*result),
        CfgInstruction::StoreBuffer { .. } => None,
    }
}

fn cfg_instruction_type(instruction: &CfgInstruction, definitions: &HashMap<u32, (crate::Ty, u32)>) -> Result<crate::Ty, ()> {
    use crate::{ScalarTy, Ty};
    match instruction {
        CfgInstruction::ConstF32 { .. } => Ok(Ty::f32()), CfgInstruction::ConstF64 { .. } => Ok(Ty::f64()),
        CfgInstruction::ConstU32 { .. } | CfgInstruction::GlobalId { .. } => Ok(Ty::Scalar(ScalarTy::U32)),
        CfgInstruction::LoadUniform { ty, .. } | CfgInstruction::LoadBuffer { ty, .. } => Ok(ty.clone()),
        CfgInstruction::Binary { op, lhs, rhs, .. } => {
            let left = definitions.get(lhs).ok_or(())?.0.clone(); let right = definitions.get(rhs).ok_or(())?.0.clone();
            let vector_scalar = matches!((&left, &right),
                (Ty::Vec2 | Ty::Vec3 | Ty::Vec4, Ty::Scalar(ScalarTy::F32 | ScalarTy::F64))
                    | (Ty::Scalar(ScalarTy::F32 | ScalarTy::F64), Ty::Vec2 | Ty::Vec3 | Ty::Vec4));
            if left != right && !(vector_scalar && matches!(op, crate::BinOp::Mul | crate::BinOp::Div)) { return Err(()); }
            if matches!(op, crate::BinOp::Lt | crate::BinOp::Le | crate::BinOp::Eq | crate::BinOp::Ne | crate::BinOp::Ge | crate::BinOp::Gt) { Ok(Ty::Scalar(ScalarTy::Bool)) }
            else if *op == crate::BinOp::Dot && left == Ty::Vec3 { Ok(Ty::f64()) }
            else if vector_scalar && matches!(right, Ty::Vec2 | Ty::Vec3 | Ty::Vec4) { Ok(right) }
            else { Ok(left) }
        }
        CfgInstruction::Unary { op, arg, .. } => {
            let ty = definitions.get(arg).ok_or(())?.0.clone();
            match op { crate::UnOp::IsFinite => Ok(Ty::Scalar(ScalarTy::Bool)), crate::UnOp::Length | crate::UnOp::Vec3Comp(_) if ty == Ty::Vec3 => Ok(Ty::f64()), crate::UnOp::Normalize if ty == Ty::Vec3 => Ok(Ty::Vec3), _ => Ok(ty) }
        }
        CfgInstruction::Vec3Compose { x, y, z, .. } => { let a = definitions.get(x).ok_or(())?.0.clone(); if definitions.get(y).ok_or(())?.0 != a || definitions.get(z).ok_or(())?.0 != a { Err(()) } else { Ok(Ty::Vec3) } }
        CfgInstruction::Select { cond, then_, else_, .. } => { if definitions.get(cond).ok_or(())?.0 != Ty::Scalar(ScalarTy::Bool) { return Err(()); } let a = definitions.get(then_).ok_or(())?.0.clone(); if definitions.get(else_).ok_or(())?.0 != a { Err(()) } else { Ok(a) } }
        CfgInstruction::StoreBuffer { .. } => Err(()),
    }
}

fn validate_edge_args(
    target: Option<&crate::CfgBlock>,
    args: &[u32],
    available: &HashSet<u32>,
    definitions: &HashMap<u32, (crate::Ty, u32)>,
) -> bool {
    let Some(target) = target else { return false; };
    target.parameters.len() == args.len()
        && target.parameters.iter().zip(args).all(|(parameter, argument)| {
            available.contains(argument) && definitions.get(argument).map(|(ty, _)| ty) == Some(&parameter.ty)
        })
}

fn collect_expr_calls(expr: &Expr, out: &mut Vec<u32>) {
    match expr {
        Expr::Call { func, args } => {
            out.push(*func);
            for arg in args {
                collect_expr_calls(arg, out);
            }
        }
        Expr::BinOp { lhs, rhs, .. } => {
            collect_expr_calls(lhs, out);
            collect_expr_calls(rhs, out);
        }
        Expr::UnOp { arg, .. } => collect_expr_calls(arg, out),
        Expr::Select { cond, then_, else_ } => {
            collect_expr_calls(cond, out);
            collect_expr_calls(then_, out);
            collect_expr_calls(else_, out);
        }
        Expr::Vec3Compose { x, y, z } => {
            collect_expr_calls(x, out);
            collect_expr_calls(y, out);
            collect_expr_calls(z, out);
        }
        Expr::ConstF32(_)
        | Expr::ConstF64(_)
        | Expr::ConstBool(_)
        | Expr::ConstU32(_)
        | Expr::LoadLocal(_)
        | Expr::LoadUniform(_)
        | Expr::LoadBuffer { .. }
        | Expr::GlobalId => {}
    }
}

fn validate_stmts(
    program: &Program,
    name: &str,
    body: &[Stmt],
    known: &HashSet<u32>,
    max_loop: u32,
    in_loop: bool,
    return_ty: &crate::Ty,
) -> Result<(), ValidateError> {
    let mut scope = known.clone();
    for stmt in body {
        match stmt {
            Stmt::Let { id, init, .. } => {
                validate_expr(program, name, init, &scope)?;
                scope.insert(*id);
            }
            Stmt::StoreLocal { id, value } => {
                if !scope.contains(id) {
                    return Err(ValidateError::UnknownLocal { name: name.to_string(), id: *id });
                }
                validate_expr(program, name, value, &scope)?;
            }
            Stmt::StoreBuffer { index, value, .. } => {
                validate_expr(program, name, index, &scope)?;
                validate_expr(program, name, value, &scope)?;
            }
            Stmt::For { iter, limit, body } => {
                if *limit > max_loop {
                    return Err(ValidateError::LoopLimitExceeded { name: name.to_string(), limit: *limit, max: max_loop });
                }
                scope.insert(*iter);
                validate_stmts(program, name, body, &scope, max_loop, true, return_ty)?;
            }
            Stmt::While { cond, limit, body } => {
                if *limit > max_loop {
                    return Err(ValidateError::LoopLimitExceeded { name: name.to_string(), limit: *limit, max: max_loop });
                }
                validate_expr(program, name, cond, &scope)?;
                validate_stmts(program, name, body, &scope, max_loop, true, return_ty)?;
            }
            Stmt::If { cond, then_body, else_body } => {
                validate_expr(program, name, cond, &scope)?;
                validate_stmts(program, name, then_body, &scope, max_loop, in_loop, return_ty)?;
                validate_stmts(program, name, else_body, &scope, max_loop, in_loop, return_ty)?;
            }
            Stmt::Break | Stmt::Continue if !in_loop => {
                return Err(ValidateError::BadControlFlow { name: name.to_string() });
            }
            Stmt::Return(expr) | Stmt::Expr(expr) => {
                validate_expr(program, name, expr, &scope)?;
            }
            Stmt::Break | Stmt::Continue => {}
            Stmt::Cfg { entry, blocks } => validate_cfg(name, *entry, blocks, &HashSet::new(), return_ty)?,
        }
    }
    Ok(())
}

fn validate_expr(program: &Program, name: &str, expr: &Expr, known: &HashSet<u32>) -> Result<(), ValidateError> {
    match expr {
        Expr::LoadLocal(id) if !known.contains(id) => Err(ValidateError::UnknownLocal { name: name.to_string(), id: *id }),
        Expr::Call { func, args } => {
            let callee = program
                .functions
                .get(*func as usize)
                .ok_or(ValidateError::UnknownFunction { name: name.to_string(), index: *func })?;
            if args.len() != callee.params.len() {
                return Err(ValidateError::CallArity {
                    name: name.to_string(),
                    index: *func,
                    expected: callee.params.len(),
                    actual: args.len(),
                });
            }
            for arg in args {
                validate_expr(program, name, arg, known)?;
            }
            Ok(())
        }
        Expr::BinOp { lhs, rhs, .. } => {
            validate_expr(program, name, lhs, known)?;
            validate_expr(program, name, rhs, known)
        }
        Expr::UnOp { arg, .. } => validate_expr(program, name, arg, known),
        Expr::LoadBuffer { index, .. } => validate_expr(program, name, index, known),
        Expr::Select { cond, then_, else_ } => {
            validate_expr(program, name, cond, known)?;
            validate_expr(program, name, then_, known)?;
            validate_expr(program, name, else_, known)
        }
        Expr::Vec3Compose { x, y, z } => {
            validate_expr(program, name, x, known)?;
            validate_expr(program, name, y, known)?;
            validate_expr(program, name, z, known)
        }
        Expr::ConstF32(_)
        | Expr::ConstF64(_)
        | Expr::ConstBool(_)
        | Expr::ConstU32(_)
        | Expr::LoadUniform(_)
        | Expr::GlobalId
        | Expr::LoadLocal(_) => Ok(()),
    }
}
