use ucf_types::{ShaderOp, ShaderProgram};

/// Emit PTX bytes for the given UCF program. The driver JITs PTX to SASS on module load.
pub fn emit_ptx(program: &ShaderProgram) -> Result<Vec<u8>, String> {
    for op in &program.ops {
        match op {
            ShaderOp::MemCopy => continue,
            ShaderOp::IotaFill { value } => {
                let text = emit_iota_fill(&program.entry, *value);
                return Ok(text.into_bytes());
            }
        }
    }
    Err("program has no emittable PTX ops".into())
}

fn emit_iota_fill(entry: &str, value: f32) -> String {
    let bits = value.to_bits();
    format!(
        r#"
.version 7.0
.target sm_75
.address_size 64

.visible .entry {entry}(
    .param .u32 count,
    .param .u64 out
)
{{
    .reg .pred %p<2>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<8>;
    .reg .f32 %f<2>;

    ld.param.u32 %r1, [count];
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r2, %r2, %r3, %r4;
    setp.ge.u32 %p1, %r2, %r1;
    @%p1 bra UCF_END;
    ld.param.u64 %rd1, [out];
    cvta.to.global.u64 %rd2, %rd1;
    mul.wide.u32 %rd3, %r2, 4;
    add.u64 %rd2, %rd2, %rd3;
    mov.b32 %f1, 0x{bits:08x};
    st.global.f32 [%rd2], %f1;
UCF_END:
    ret;
}}
"#
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ucf_types::ShaderProgram;

    #[test]
    fn ptx_contains_entry_and_no_cuda_c() {
        let program = ShaderProgram::dispatch_fill("ucf_dispatch", 1.0);
        let ptx = emit_ptx(&program).expect("ptx");
        let text = String::from_utf8(ptx).expect("utf8");
        assert!(text.contains(".entry ucf_dispatch"));
        assert!(!text.contains("#include"));
        assert!(!text.contains("__global__"));
    }
}
