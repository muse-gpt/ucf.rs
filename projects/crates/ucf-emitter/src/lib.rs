//! Built-in shader emitter: UCF `ShaderProgram` → PTX / DXIL bytes.
#![warn(missing_docs)]

mod dxil;
mod ptx;
mod task;

pub use dxil::emit_dxil;
pub use ptx::emit_ptx;
pub use task::{needs_kernel, program_from_task};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShaderIsa {
    Ptx,
    Dxil,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmittedShader {
    pub isa: ShaderIsa,
    pub bytes: Vec<u8>,
}

impl EmittedShader {
    pub fn emit(program: &ucf_types::ShaderProgram, isa: ShaderIsa) -> Result<Self, String> {
        let bytes = match isa {
            ShaderIsa::Ptx => emit_ptx(program)?,
            ShaderIsa::Dxil => emit_dxil(program)?,
        };
        Ok(Self { isa, bytes })
    }
}
