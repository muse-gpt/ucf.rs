//! UCF program and legacy shader operation emitters.
#![warn(missing_docs)]

mod dxil;
mod hip;
mod program_spirv;
mod ptx;
mod spirv;
mod task;

pub use dxil::{emit_dxil, emit_raster_tri_dxbc, RasterTriDxbc};
pub use hip::{emit_hip, emit_hsaco};
pub use program_spirv::{emit_program_spirv, ProgramSpirvError};
pub use ptx::emit_ptx;
pub use spirv::{emit_raster_tri_spirv, emit_spirv, RasterTriSpirv};
pub use task::{needs_kernel, program_from_task, try_program_from_task, ProgramLowerError};

/// Target ISA for [`EmittedShader::emit`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShaderIsa {
    /// CUDA PTX text (NUL-terminated).
    Ptx,
    /// Direct3D compute bytecode (`cs_5_1` DXBC on Windows).
    Dxil,
    /// Vulkan SPIR-V module bytes.
    Spirv,
    /// AMD HSACO ELF code object for `hipModuleLoadData`.
    Hip,
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
    pub fn emit(program: &hardware_types::ShaderProgram, isa: ShaderIsa) -> Result<Self, String> {
        let bytes = match isa {
            ShaderIsa::Ptx => emit_ptx(program)?,
            ShaderIsa::Dxil => emit_dxil(program)?,
            ShaderIsa::Spirv => emit_spirv(program)?,
            ShaderIsa::Hip => emit_hsaco(program)?,
        };
        Ok(Self { isa, bytes })
    }
}
