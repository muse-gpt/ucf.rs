//! Built-in shader emitter: UCF `ShaderProgram` → PTX / DXBC / SPIR-V bytes.
#![warn(missing_docs)]

mod dxil;
mod ptx;
mod spirv;
mod task;

pub use dxil::emit_dxil;
pub use ptx::emit_ptx;
pub use spirv::emit_spirv;
pub use task::{needs_kernel, program_from_task};

/// Target ISA for [`EmittedShader::emit`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShaderIsa {
    /// CUDA PTX text (NUL-terminated).
    Ptx,
    /// Direct3D compute bytecode (`cs_5_1` DXBC on Windows).
    Dxil,
    /// Vulkan SPIR-V module bytes.
    Spirv,
}

/// Bytes produced by the built-in emitter for one ISA.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmittedShader {
    /// Selected ISA.
    pub isa: ShaderIsa,
    /// Opaque bytecode / text for the driver.
    pub bytes: Vec<u8>,
}

impl EmittedShader {
    /// Emit `program` for `isa`.
    pub fn emit(program: &ucf_types::ShaderProgram, isa: ShaderIsa) -> Result<Self, String> {
        let bytes = match isa {
            ShaderIsa::Ptx => emit_ptx(program)?,
            ShaderIsa::Dxil => emit_dxil(program)?,
            ShaderIsa::Spirv => emit_spirv(program)?,
        };
        Ok(Self { isa, bytes })
    }
}
