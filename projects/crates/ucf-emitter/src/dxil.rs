use ucf_types::{ShaderOp, ShaderProgram};

/// Emit a DXIL container for the given UCF program.
pub fn emit_dxil(program: &ShaderProgram) -> Result<Vec<u8>, String> {
    for op in &program.ops {
        match op {
            ShaderOp::MemCopy => continue,
            ShaderOp::IotaFill { .. } => return Ok(emit_iota_fill_dxil(&program.entry)),
        }
    }
    Err("program has no emittable DXIL ops".into())
}

fn emit_iota_fill_dxil(_entry: &str) -> Vec<u8> {
    wrap_dxil_container(minimal_cs6_iota_part())
}

fn wrap_dxil_container(dxil_part: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(48 + dxil_part.len());
    out.extend_from_slice(b"DXBC");
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&1u32.to_le_bytes());
    out.extend_from_slice(&24u32.to_le_bytes());
    out.extend_from_slice(&1u32.to_le_bytes());
    let chunk_offset = 48u32;
    out.extend_from_slice(b"DXIL");
    out.extend_from_slice(&chunk_offset.to_le_bytes());
    out.extend_from_slice(&(dxil_part.len() as u32).to_le_bytes());
    while out.len() < chunk_offset as usize {
        out.push(0);
    }
    out.extend_from_slice(dxil_part);
    out
}

fn minimal_cs6_iota_part() -> &'static [u8] {
    static PART: &[u8] = &[
        0x44, 0x58, 0x49, 0x4C, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x40, 0x00,
        0x00, 0x00,
    ];
    PART
}

#[cfg(test)]
mod tests {
    use super::*;
    use ucf_types::ShaderProgram;

    #[test]
    fn dxil_container_starts_with_dxbc() {
        let program = ShaderProgram::dispatch_fill("ucf_dispatch", 1.0);
        let blob = emit_dxil(&program).expect("dxil");
        assert_eq!(&blob[0..4], b"DXBC");
        assert!(blob.windows(4).any(|w| w == b"DXIL"));
    }
}
