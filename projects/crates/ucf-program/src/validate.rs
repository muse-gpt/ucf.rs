use std::collections::HashSet;

use crate::error::ValidateError;
use crate::program::Program;
use crate::stmt::Stmt;

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
        validate_stmts(&func.name, &func.body, &known, DEFAULT_MAX_LOOP)?;
    }
    Ok(())
}

fn validate_stmts(
    name: &str,
    body: &[Stmt],
    known: &HashSet<u32>,
    max_loop: u32,
) -> Result<(), ValidateError> {
    let mut scope = known.clone();
    for stmt in body {
        match stmt {
            Stmt::Let { id, .. } => {
                scope.insert(*id);
            }
            Stmt::StoreLocal { id, .. } => {
                if !scope.contains(id) {
                    return Err(ValidateError::UnknownLocal {
                        name: name.to_string(),
                        id: *id,
                    });
                }
            }
            Stmt::For { iter, limit, body } => {
                if *limit > max_loop {
                    return Err(ValidateError::LoopLimitExceeded {
                        name: name.to_string(),
                        limit: *limit,
                        max: max_loop,
                    });
                }
                scope.insert(*iter);
                validate_stmts(name, body, &scope, max_loop)?;
            }
            Stmt::While { limit, body, .. } => {
                if *limit > max_loop {
                    return Err(ValidateError::LoopLimitExceeded {
                        name: name.to_string(),
                        limit: *limit,
                        max: max_loop,
                    });
                }
                validate_stmts(name, body, &scope, max_loop)?;
            }
            Stmt::If { then_body, else_body, .. } => {
                validate_stmts(name, then_body, &scope, max_loop)?;
                validate_stmts(name, else_body, &scope, max_loop)?;
            }
            Stmt::Break | Stmt::Continue | Stmt::Return(_) | Stmt::Expr(_) => {}
        }
    }
    Ok(())
}
