use ucf_types::{ShaderOp, ShaderProgram};

/// Emit PTX bytes for the given UCF program. The driver JITs PTX to SASS on module load.
pub fn emit_ptx(program: &ShaderProgram) -> Result<Vec<u8>, String> {
    for op in &program.ops {
        match op {
            ShaderOp::MemCopy => continue,
            ShaderOp::IotaFill { value } => {
                let text = emit_iota_fill(&program.entry, *value);
                let mut bytes = text.into_bytes();
                bytes.push(0);
                return Ok(bytes);
            }
            ShaderOp::MatMul => {
                let text = emit_matmul(&program.entry);
                let mut bytes = text.into_bytes();
                bytes.push(0);
                return Ok(bytes);
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

fn emit_matmul(entry: &str) -> String {
    format!(
        r#"
.version 7.0
.target sm_75
.address_size 64

.visible .entry {entry}(
    .param .u64 a,
    .param .u64 b,
    .param .u64 out,
    .param .u32 m,
    .param .u32 n,
    .param .u32 k
)
{{
    .reg .pred %p<4>;
    .reg .b32 %r<16>;
    .reg .b64 %rd<16>;
    .reg .f32 %f<8>;

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r10, %r2, %r3, %r4;

    ld.param.u32 %r5, [m];
    ld.param.u32 %r6, [n];
    mul.lo.u32 %r7, %r5, %r6;
    setp.ge.u32 %p1, %r10, %r7;
    @%p1 bra UCF_END;

    rem.u32 %r9, %r10, %r6;
    div.u32 %r8, %r10, %r6;

    ld.param.u32 %r11, [k];
    mov.f32 %f1, 0f00000000;
    mov.u32 %r12, 0;

UCF_LOOP:
    setp.ge.u32 %p2, %r12, %r11;
    @%p2 bra UCF_STORE;

    mad.lo.u32 %r13, %r8, %r11, %r12;
    mad.lo.u32 %r14, %r12, %r6, %r9;

    ld.param.u64 %rd1, [a];
    cvta.to.global.u64 %rd2, %rd1;
    mul.wide.u32 %rd3, %r13, 4;
    add.u64 %rd2, %rd2, %rd3;
    ld.global.f32 %f2, [%rd2];

    ld.param.u64 %rd4, [b];
    cvta.to.global.u64 %rd5, %rd4;
    mul.wide.u32 %rd6, %r14, 4;
    add.u64 %rd5, %rd5, %rd6;
    ld.global.f32 %f3, [%rd5];

    fma.rn.f32 %f1, %f2, %f3, %f1;
    add.u32 %r12, %r12, 1;
    bra UCF_LOOP;

UCF_STORE:
    ld.param.u64 %rd7, [out];
    cvta.to.global.u64 %rd8, %rd7;
    mul.wide.u32 %rd9, %r10, 4;
    add.u64 %rd8, %rd8, %rd9;
    st.global.f32 [%rd8], %f1;
UCF_END:
    ret;
}}
"#
    )
}
