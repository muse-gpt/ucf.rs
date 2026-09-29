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
            ShaderOp::Rgba8Denoise => {
                let text = emit_rgba8_denoise(&program.entry);
                let mut bytes = text.into_bytes();
                bytes.push(0);
                return Ok(bytes);
            }
            ShaderOp::Attention => {
                let text = emit_attention(&program.entry);
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

fn emit_rgba8_denoise(entry: &str) -> String {
    // Horizontal low-pass: out.rgb = (pix + left_or_self) / 2, out.a = pix.a
    format!(
        r#"
.version 7.0
.target sm_75
.address_size 64

.visible .entry {entry}(
    .param .u64 src,
    .param .u64 out,
    .param .u32 width,
    .param .u32 height
)
{{
    .reg .pred %p<4>;
    .reg .b32 %r<32>;
    .reg .b64 %rd<24>;

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r10, %r2, %r3, %r4;

    ld.param.u32 %r5, [width];
    ld.param.u32 %r6, [height];
    mul.lo.u32 %r7, %r5, %r6;
    setp.ge.u32 %p1, %r10, %r7;
    @%p1 bra UCF_END;

    rem.u32 %r8, %r10, %r5;

    ld.param.u64 %rd1, [src];
    cvta.to.global.u64 %rd2, %rd1;
    mul.wide.u32 %rd3, %r10, 4;
    add.u64 %rd4, %rd2, %rd3;
    ld.global.u8 %r11, [%rd4];
    add.u64 %rd5, %rd4, 1;
    ld.global.u8 %r12, [%rd5];
    add.u64 %rd6, %rd4, 2;
    ld.global.u8 %r13, [%rd6];
    add.u64 %rd7, %rd4, 3;
    ld.global.u8 %r14, [%rd7];

    mov.u32 %r15, %r10;
    setp.eq.u32 %p2, %r8, 0;
    @%p2 bra UCF_LEFT_DONE;
    sub.u32 %r15, %r10, 1;
UCF_LEFT_DONE:
    mul.wide.u32 %rd8, %r15, 4;
    add.u64 %rd9, %rd2, %rd8;
    ld.global.u8 %r16, [%rd9];
    add.u64 %rd10, %rd9, 1;
    ld.global.u8 %r17, [%rd10];
    add.u64 %rd11, %rd9, 2;
    ld.global.u8 %r18, [%rd11];

    add.u32 %r19, %r11, %r16;
    shr.u32 %r19, %r19, 1;
    add.u32 %r20, %r12, %r17;
    shr.u32 %r20, %r20, 1;
    add.u32 %r21, %r13, %r18;
    shr.u32 %r21, %r21, 1;

    ld.param.u64 %rd12, [out];
    cvta.to.global.u64 %rd13, %rd12;
    add.u64 %rd14, %rd13, %rd3;
    st.global.u8 [%rd14], %r19;
    add.u64 %rd15, %rd14, 1;
    st.global.u8 [%rd15], %r20;
    add.u64 %rd16, %rd14, 2;
    st.global.u8 [%rd16], %r21;
    add.u64 %rd17, %rd14, 3;
    st.global.u8 [%rd17], %r14;
UCF_END:
    ret;
}}
"#
    )
}

fn emit_attention(entry: &str) -> String {
    // Out[b,h,i,d] = sum_j (sum_dd Q[b,h,i,dd]*K[b,h,j,dd]) * V[b,h,j,d]
    format!(
        r#"
.version 7.0
.target sm_75
.address_size 64

.visible .entry {entry}(
    .param .u64 q,
    .param .u64 k,
    .param .u64 v,
    .param .u64 out,
    .param .u32 batch,
    .param .u32 heads,
    .param .u32 seq,
    .param .u32 dim
)
{{
    .reg .pred %p<8>;
    .reg .b32 %r<40>;
    .reg .b64 %rd<24>;
    .reg .f32 %f<8>;

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r10, %r2, %r3, %r4;

    ld.param.u32 %r11, [batch];
    ld.param.u32 %r12, [heads];
    ld.param.u32 %r13, [seq];
    ld.param.u32 %r14, [dim];
    mul.lo.u32 %r15, %r11, %r12;
    mul.lo.u32 %r15, %r15, %r13;
    mul.lo.u32 %r15, %r15, %r14;
    setp.ge.u32 %p1, %r10, %r15;
    @%p1 bra UCF_END;

    rem.u32 %r20, %r10, %r14;
    div.u32 %r16, %r10, %r14;
    rem.u32 %r21, %r16, %r13;
    div.u32 %r16, %r16, %r13;
    rem.u32 %r22, %r16, %r12;
    div.u32 %r23, %r16, %r12;

    mov.f32 %f1, 0f00000000;
    mov.u32 %r24, 0;

UCF_J:
    setp.ge.u32 %p2, %r24, %r13;
    @%p2 bra UCF_STORE;

    mov.f32 %f2, 0f00000000;
    mov.u32 %r25, 0;

UCF_DD:
    setp.ge.u32 %p3, %r25, %r14;
    @%p3 bra UCF_ACC;

    mad.lo.u32 %r26, %r23, %r12, %r22;
    mad.lo.u32 %r26, %r26, %r13, %r21;
    mad.lo.u32 %r26, %r26, %r14, %r25;

    mad.lo.u32 %r27, %r23, %r12, %r22;
    mad.lo.u32 %r27, %r27, %r13, %r24;
    mad.lo.u32 %r27, %r27, %r14, %r25;

    ld.param.u64 %rd1, [q];
    cvta.to.global.u64 %rd2, %rd1;
    mul.wide.u32 %rd3, %r26, 4;
    add.u64 %rd2, %rd2, %rd3;
    ld.global.f32 %f3, [%rd2];

    ld.param.u64 %rd4, [k];
    cvta.to.global.u64 %rd5, %rd4;
    mul.wide.u32 %rd6, %r27, 4;
    add.u64 %rd5, %rd5, %rd6;
    ld.global.f32 %f4, [%rd5];

    fma.rn.f32 %f2, %f3, %f4, %f2;
    add.u32 %r25, %r25, 1;
    bra UCF_DD;

UCF_ACC:
    mad.lo.u32 %r28, %r23, %r12, %r22;
    mad.lo.u32 %r28, %r28, %r13, %r24;
    mad.lo.u32 %r28, %r28, %r14, %r20;

    ld.param.u64 %rd7, [v];
    cvta.to.global.u64 %rd8, %rd7;
    mul.wide.u32 %rd9, %r28, 4;
    add.u64 %rd8, %rd8, %rd9;
    ld.global.f32 %f5, [%rd8];

    fma.rn.f32 %f1, %f2, %f5, %f1;
    add.u32 %r24, %r24, 1;
    bra UCF_J;

UCF_STORE:
    ld.param.u64 %rd10, [out];
    cvta.to.global.u64 %rd11, %rd10;
    mul.wide.u32 %rd12, %r10, 4;
    add.u64 %rd11, %rd11, %rd12;
    st.global.f32 [%rd11], %f1;
UCF_END:
    ret;
}}
"#
    )
}
